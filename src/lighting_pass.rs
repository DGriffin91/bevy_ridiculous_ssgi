use bevy::app::prelude::*;
use bevy::asset::{load_internal_asset, Handle};
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::core_pipeline::{
    deferred::{
        copy_lighting_id::DeferredLightingIdDepthTexture, DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT,
    },
    prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass},
    tonemapping::{DebandDither, Tonemapping},
};
use bevy::ecs::{prelude::*, query::QueryItem};
use bevy::pbr::irradiance_volume::IrradianceVolume;
use bevy::pbr::{
    MeshPipeline, MeshPipelineKey, MeshViewBindGroup, RenderViewLightProbes,
    ScreenSpaceAmbientOcclusionSettings, ShadowFilteringMethod, ViewFogUniformOffset,
    ViewLightProbesUniformOffset, ViewLightsUniformOffset,
};
use bevy::prelude::EnvironmentMapLight;
use bevy::render::render_graph::RenderLabel;
use bevy::render::texture::BevyDefault;
use bevy::render::{
    extract_component::{
        ComponentUniforms, ExtractComponent, ExtractComponentPlugin, UniformComponentPlugin,
    },
    render_asset::RenderAssets,
    render_graph::{NodeRunError, RenderGraphContext, ViewNode, ViewNodeRunner},
    render_resource::{self, Operations, PipelineCache, RenderPassDescriptor},
    renderer::{RenderContext, RenderDevice},
    texture::Image,
    view::{ViewTarget, ViewUniformOffset},
    Render, RenderSet,
};

use bevy::render::{
    render_graph::RenderGraphApp, render_resource::*, view::ExtractedView, RenderApp,
};

use crate::bind_group_utils::{
    fsampler_layout_entry, ftexture_layout_entry, linear_sampler, nearest_sampler,
};
use crate::copy_frame::PrevFrameTexture;
use crate::ssgi_resolve::SSGIResolveTextures;
use crate::{
    image, resource, shader_def_uint, BlueNoise, BLUE_NOISE_DIMS, BLUE_NOISE_ENTRY_N,
    BLUE_NOISE_GROUP_N,
};

pub struct CustomDeferredPbrLightingPlugin;

pub const DEFERRED_LIGHTING_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(2039845798734502984);

pub const DEFAULT_PBR_DEFERRED_LIGHTING_PASS_ID: u8 = 1;

/// Component with a `depth_id` for specifying which corresponding materials should be rendered by this specific PBR deferred lighting pass.
/// Will be automatically added to entities with the [`DeferredPrepass`] component that don't already have a [`PbrDeferredLightingDepthId`].
#[derive(Component, Clone, Copy, ExtractComponent, ShaderType)]
pub struct PbrDeferredLightingDepthId {
    depth_id: u32,

    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
    _webgl2_padding_0: f32,
    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
    _webgl2_padding_1: f32,
    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
    _webgl2_padding_2: f32,
}

impl PbrDeferredLightingDepthId {
    pub fn new(value: u8) -> PbrDeferredLightingDepthId {
        PbrDeferredLightingDepthId {
            depth_id: value as u32,

            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            _webgl2_padding_0: 0.0,
            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            _webgl2_padding_1: 0.0,
            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            _webgl2_padding_2: 0.0,
        }
    }

    pub fn set(&mut self, value: u8) {
        self.depth_id = value as u32;
    }

    pub fn get(&self) -> u8 {
        self.depth_id as u8
    }
}

impl Default for PbrDeferredLightingDepthId {
    fn default() -> Self {
        PbrDeferredLightingDepthId {
            depth_id: DEFAULT_PBR_DEFERRED_LIGHTING_PASS_ID as u32,

            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            _webgl2_padding_0: 0.0,
            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            _webgl2_padding_1: 0.0,
            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            _webgl2_padding_2: 0.0,
        }
    }
}

const SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(2039485723049597);
impl Plugin for CustomDeferredPbrLightingPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SHADER_HANDLE,
            "../assets/shaders/deferred_lighting.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins((
            ExtractComponentPlugin::<PbrDeferredLightingDepthId>::default(),
            UniformComponentPlugin::<PbrDeferredLightingDepthId>::default(),
        ))
        .add_systems(PostUpdate, insert_deferred_lighting_pass_id_component);

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<DeferredLightingLayout>>()
            .add_systems(
                Render,
                (prepare_deferred_lighting_pipelines.in_set(RenderSet::Prepare),),
            )
            .add_render_graph_node::<ViewNodeRunner<DeferredOpaquePass3dPbrLightingNode>>(
                Core3d,
                DeferredOpaquePass3dPbrLightingLabel,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::StartMainPass,
                    DeferredOpaquePass3dPbrLightingLabel,
                    Node3d::MainOpaquePass,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<DeferredLightingLayout>();
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct DeferredOpaquePass3dPbrLightingLabel;
#[derive(Default)]
pub struct DeferredOpaquePass3dPbrLightingNode;

impl ViewNode for DeferredOpaquePass3dPbrLightingNode {
    type ViewQuery = (
        &'static ViewUniformOffset,
        &'static ViewLightsUniformOffset,
        &'static ViewFogUniformOffset,
        &'static ViewLightProbesUniformOffset,
        &'static MeshViewBindGroup,
        &'static ViewTarget,
        &'static DeferredLightingIdDepthTexture,
        &'static DeferredLightingPipeline,
        &'static PrevFrameTexture,
        // todo webgl &'static DisocclusionTextures,
        &'static SSGIResolveTextures,
    );

    fn run(
        &self,
        _graph_context: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            view_uniform_offset,
            view_lights_offset,
            view_fog_offset,
            view_light_probes_offset,
            mesh_view_bind_group,
            target,
            deferred_lighting_id_depth_texture,
            deferred_lighting_pipeline,
            prev_frame_tex,
            ssgi_resolve,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let deferred_lighting_layout = world.resource::<DeferredLightingLayout>();
        let images = world.resource::<RenderAssets<Image>>();

        let Some(pipeline) =
            pipeline_cache.get_render_pipeline(deferred_lighting_pipeline.pipeline_id)
        else {
            return Ok(());
        };

        let deferred_lighting_pass_id =
            world.resource::<ComponentUniforms<PbrDeferredLightingDepthId>>();
        let Some(deferred_lighting_pass_id_binding) =
            deferred_lighting_pass_id.uniforms().binding()
        else {
            return Ok(());
        };

        let blue_noise_tex = image!(images, &resource!(world, BlueNoise).0);
        let nearest_sampler = nearest_sampler(render_context.render_device());
        let linear_sampler = linear_sampler(render_context.render_device());

        let bind_group_1 = render_context.render_device().create_bind_group(
            "deferred_lighting_layout_group_1",
            &deferred_lighting_layout.bind_group_layout_1,
            &BindGroupEntries::with_indices((
                (0, deferred_lighting_pass_id_binding),
                (1, &prev_frame_tex.texture.default_view),
                (5, &nearest_sampler),
                (6, &linear_sampler),
                // todo webgl (8, &disocclusion_textures.output.default_view),
                (BLUE_NOISE_ENTRY_N, &blue_noise_tex.texture_view),
                // Use write since it's the one  resolve would have just written to
                (12, &ssgi_resolve.write.default_view),
            )),
        );

        //Operations {
        //    load: match camera_3d.clear_color {
        //        ClearColorConfig::Default => {
        //            LoadOp::Clear(world.resource::<ClearColor>().0.into())
        //        }
        //        ClearColorConfig::Custom(color) => LoadOp::Clear(color.into()),
        //        ClearColorConfig::None => LoadOp::Load,
        //    },
        //    store: true,
        //}

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("deferred_lighting_pass"),
            color_attachments: &[Some(target.get_color_attachment())],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &deferred_lighting_id_depth_texture.texture.default_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(
            0,
            &mesh_view_bind_group.value,
            &[
                view_uniform_offset.offset,
                view_lights_offset.offset,
                view_fog_offset.offset,
                **view_light_probes_offset,
            ],
        );
        render_pass.set_bind_group(1, &bind_group_1, &[]);
        render_pass.draw(0..3, 0..1);
        Ok(())
    }
}

impl FromWorld for DeferredLightingLayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            Some("deferred_lighting_layout"),
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: render_resource::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(PbrDeferredLightingDepthId::min_size()),
                    },
                    count: None,
                },
                ftexture_layout_entry(1, TextureViewDimension::D2), // Prev frame
                fsampler_layout_entry(5),                           // Nearest Sampler
                fsampler_layout_entry(6),                           // Linear Sampler
                // todo webgl ftexture_layout_entry(8, TextureViewDimension::D2), // Disocclusion
                ftexture_layout_entry(BLUE_NOISE_ENTRY_N, TextureViewDimension::D2Array), // Blue Noise
                ftexture_layout_entry(12, TextureViewDimension::D2), // SSGI Resolve
            ],
        );
        let mesh_pipeline = world.resource::<MeshPipeline>().clone();

        #[cfg(not(all(feature = "file_watcher")))]
        let shader = SHADER_HANDLE;
        #[cfg(all(feature = "file_watcher"))]
        let shader = {
            let asset_server = world.resource_mut::<bevy::prelude::AssetServer>();
            asset_server.load("shaders/deferred_lighting.wgsl")
        };

        Self {
            mesh_pipeline,
            bind_group_layout_1: layout,
            shader,
        }
    }
}

#[derive(Resource)]
pub struct DeferredLightingLayout {
    mesh_pipeline: MeshPipeline,
    bind_group_layout_1: BindGroupLayout,
    pub shader: Handle<Shader>,
}

#[derive(Component)]
pub struct DeferredLightingPipeline {
    pub pipeline_id: CachedRenderPipelineId,
}

impl SpecializedRenderPipeline for DeferredLightingLayout {
    type Key = MeshPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();

        // Let the shader code know that it's running in a deferred pipeline.
        shader_defs.push("DEFERRED_LIGHTING_PIPELINE".into());

        #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
        shader_defs.push("WEBGL2".into());

        if key.contains(MeshPipelineKey::TONEMAP_IN_SHADER) {
            shader_defs.push("TONEMAP_IN_SHADER".into());

            let method = key.intersection(MeshPipelineKey::TONEMAP_METHOD_RESERVED_BITS);

            if method == MeshPipelineKey::TONEMAP_METHOD_NONE {
                shader_defs.push("TONEMAP_METHOD_NONE".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_REINHARD {
                shader_defs.push("TONEMAP_METHOD_REINHARD".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE {
                shader_defs.push("TONEMAP_METHOD_REINHARD_LUMINANCE".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_ACES_FITTED {
                shader_defs.push("TONEMAP_METHOD_ACES_FITTED ".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_AGX {
                shader_defs.push("TONEMAP_METHOD_AGX".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM {
                shader_defs.push("TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_BLENDER_FILMIC {
                shader_defs.push("TONEMAP_METHOD_BLENDER_FILMIC".into());
            } else if method == MeshPipelineKey::TONEMAP_METHOD_TONY_MC_MAPFACE {
                shader_defs.push("TONEMAP_METHOD_TONY_MC_MAPFACE".into());
            }

            // Debanding is tied to tonemapping in the shader, cannot run without it.
            if key.contains(MeshPipelineKey::DEBAND_DITHER) {
                shader_defs.push("DEBAND_DITHER".into());
            }
        }

        if key.contains(MeshPipelineKey::SCREEN_SPACE_AMBIENT_OCCLUSION) {
            shader_defs.push("SCREEN_SPACE_AMBIENT_OCCLUSION".into());
        }

        if key.contains(MeshPipelineKey::ENVIRONMENT_MAP) {
            shader_defs.push("ENVIRONMENT_MAP".into());
        }

        if key.contains(MeshPipelineKey::NORMAL_PREPASS) {
            shader_defs.push("NORMAL_PREPASS".into());
        }

        if key.contains(MeshPipelineKey::DEPTH_PREPASS) {
            shader_defs.push("DEPTH_PREPASS".into());
        }

        if key.contains(MeshPipelineKey::MOTION_VECTOR_PREPASS) {
            shader_defs.push("MOTION_VECTOR_PREPASS".into());
        }

        // Always true, since we're in the deferred lighting pipeline
        shader_defs.push("DEFERRED_PREPASS".into());

        let shadow_filter_method =
            key.intersection(MeshPipelineKey::SHADOW_FILTER_METHOD_RESERVED_BITS);
        if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2 {
            shader_defs.push("SHADOW_FILTER_METHOD_HARDWARE_2X2".into());
        } else if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_CASTANO_13 {
            shader_defs.push("SHADOW_FILTER_METHOD_CASTANO_13".into());
        } else if shadow_filter_method == MeshPipelineKey::SHADOW_FILTER_METHOD_JIMENEZ_14 {
            shader_defs.push("SHADOW_FILTER_METHOD_JIMENEZ_14".into());
        }

        #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
        shader_defs.push("SIXTEEN_BYTE_ALIGNMENT".into());

        shader_defs.extend_from_slice(&[
            shader_def_uint!(BLUE_NOISE_GROUP_N),
            shader_def_uint!(BLUE_NOISE_ENTRY_N),
            shader_def_uint!(BLUE_NOISE_DIMS),
        ]);

        RenderPipelineDescriptor {
            label: Some("deferred_lighting_pipeline".into()),
            layout: vec![
                self.mesh_pipeline.get_view_layout(key.into()).clone(),
                self.bind_group_layout_1.clone(),
            ],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                entry_point: "vertex".into(),
                buffers: Vec::new(),
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: if key.contains(MeshPipelineKey::HDR) {
                        ViewTarget::TEXTURE_FORMAT_HDR
                    } else {
                        TextureFormat::bevy_default()
                    },
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: DEFERRED_LIGHTING_PASS_ID_DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: CompareFunction::Equal,
                stencil: StencilState {
                    front: StencilFaceState::IGNORE,
                    back: StencilFaceState::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
                bias: DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: MultisampleState::default(),
            push_constant_ranges: vec![],
        }
    }
}

pub fn insert_deferred_lighting_pass_id_component(
    mut commands: Commands,
    views: Query<Entity, (With<DeferredPrepass>, Without<PbrDeferredLightingDepthId>)>,
) {
    for entity in views.iter() {
        commands
            .entity(entity)
            .insert(PbrDeferredLightingDepthId::default());
    }
}

pub fn prepare_deferred_lighting_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<DeferredLightingLayout>>,
    deferred_lighting_layout: Res<DeferredLightingLayout>,
    views: Query<
        (
            Entity,
            &ExtractedView,
            Option<&Tonemapping>,
            Option<&DebandDither>,
            Option<&ShadowFilteringMethod>,
            Option<&ScreenSpaceAmbientOcclusionSettings>,
            (
                Has<NormalPrepass>,
                Has<DepthPrepass>,
                Has<MotionVectorPrepass>,
            ),
            Has<RenderViewLightProbes<EnvironmentMapLight>>,
            Has<RenderViewLightProbes<IrradianceVolume>>,
        ),
        With<DeferredPrepass>,
    >,
) {
    for (
        entity,
        view,
        tonemapping,
        dither,
        shadow_filter_method,
        ssao,
        (normal_prepass, depth_prepass, motion_vector_prepass),
        has_environment_maps,
        has_irradiance_volumes,
    ) in &views
    {
        let mut view_key = MeshPipelineKey::from_hdr(view.hdr);

        if normal_prepass {
            view_key |= MeshPipelineKey::NORMAL_PREPASS;
        }

        if depth_prepass {
            view_key |= MeshPipelineKey::DEPTH_PREPASS;
        }

        if motion_vector_prepass {
            view_key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
        }

        // Always true, since we're in the deferred lighting pipeline
        view_key |= MeshPipelineKey::DEFERRED_PREPASS;

        if !view.hdr {
            if let Some(tonemapping) = tonemapping {
                view_key |= MeshPipelineKey::TONEMAP_IN_SHADER;
                view_key |= match tonemapping {
                    Tonemapping::None => MeshPipelineKey::TONEMAP_METHOD_NONE,
                    Tonemapping::Reinhard => MeshPipelineKey::TONEMAP_METHOD_REINHARD,
                    Tonemapping::ReinhardLuminance => {
                        MeshPipelineKey::TONEMAP_METHOD_REINHARD_LUMINANCE
                    }
                    Tonemapping::AcesFitted => MeshPipelineKey::TONEMAP_METHOD_ACES_FITTED,
                    Tonemapping::AgX => MeshPipelineKey::TONEMAP_METHOD_AGX,
                    Tonemapping::SomewhatBoringDisplayTransform => {
                        MeshPipelineKey::TONEMAP_METHOD_SOMEWHAT_BORING_DISPLAY_TRANSFORM
                    }
                    Tonemapping::TonyMcMapface => MeshPipelineKey::TONEMAP_METHOD_TONY_MC_MAPFACE,
                    Tonemapping::BlenderFilmic => MeshPipelineKey::TONEMAP_METHOD_BLENDER_FILMIC,
                };
            }
            if let Some(DebandDither::Enabled) = dither {
                view_key |= MeshPipelineKey::DEBAND_DITHER;
            }
        }

        if ssao.is_some() {
            view_key |= MeshPipelineKey::SCREEN_SPACE_AMBIENT_OCCLUSION;
        }

        // We don't need to check to see whether the environment map is loaded
        // because [`gather_light_probes`] already checked that for us before
        // adding the [`RenderViewEnvironmentMaps`] component.
        if has_environment_maps {
            view_key |= MeshPipelineKey::ENVIRONMENT_MAP;
        }

        if has_irradiance_volumes {
            view_key |= MeshPipelineKey::IRRADIANCE_VOLUME;
        }

        match shadow_filter_method.unwrap_or(&ShadowFilteringMethod::default()) {
            ShadowFilteringMethod::Hardware2x2 => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2;
            }
            ShadowFilteringMethod::Castano13 => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_CASTANO_13;
            }
            ShadowFilteringMethod::Jimenez14 => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_JIMENEZ_14;
            }
        }

        let pipeline_id =
            pipelines.specialize(&pipeline_cache, &deferred_lighting_layout, view_key);

        commands
            .entity(entity)
            .insert(DeferredLightingPipeline { pipeline_id });
    }
}
