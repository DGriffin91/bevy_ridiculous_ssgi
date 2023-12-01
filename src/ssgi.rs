use bevy::app::prelude::*;
use bevy::asset::{load_internal_asset, Handle};
use bevy::prelude::*;

use bevy::core_pipeline::core_3d;
use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::ecs::query::QueryItem;
use bevy::prelude::Image;
use bevy::reflect::Reflect;
use bevy::render::camera::ExtractedCamera;
use bevy::render::extract_component::{ExtractComponent, ExtractComponentPlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::texture::{CachedTexture, TextureCache};
use bevy::render::{
    render_graph::{NodeRunError, RenderGraphContext, ViewNode, ViewNodeRunner},
    render_resource::{Operations, PipelineCache, RenderPassDescriptor},
    renderer::{RenderContext, RenderDevice},
    view::ViewUniformOffset,
    Render, RenderSet,
};

use bevy::render::{render_graph::RenderGraphApp, render_resource::*, RenderApp};
use bevy_inspector_egui::InspectorOptions;

use crate::bind_group_utils::{
    fsampler_layout_entry, ftexture_layout_entry, globals_binding, globals_layout_entry,
    linear_sampler, nearest_sampler, uniform_buffer, uniform_layout_entry, utexture_layout_entry,
    view_binding, view_layout_entry,
};
use crate::copy_frame::PrevFrameTexture;
use crate::prepass_downsample::{DownsampleNode, PrepassDownsampleTextures};
use crate::{image, resource, shader_def_uint, BlueNoise, BLUE_NOISE_DIMS, BLUE_NOISE_ENTRY_N};
use bevy_inspector_egui::prelude::ReflectInspectorOptions;

#[derive(Component, ExtractComponent, Clone, Reflect, InspectorOptions)]
#[reflect(Component, InspectorOptions)]
pub struct SSGIPass {
    /// Proportion of the render target resolution. Should be a multiple of 2, [2..=16], 2 is quite slow
    #[inspector(min = 2, max = 32)]
    pub render_scale: u32,
    /// Must be a square multiple of 4
    #[inspector(min = 4, max = 32)]
    pub cascade_0_directions: u32,
    /// [2..=6]
    #[inspector(min = 2, max = 8)]
    pub cascade_count: u32,
    pub jitter_probe_position: bool,
    pub jitter_probe_direction: bool,
    /// How many frames before the noise pattern repeats
    /// If too high, flickering & temporal noise will be visible.
    /// If too low noise from aliasing will be visible when things are moving.
    pub noise_frame_period: u32,
    /// How much differences in depth affect interpolation between probes when combining cascades
    #[inspector(min = 0.0)]
    pub distance_rejection: f32,
    /// How much differences in normals affect interpolation between probes when combining cascades
    #[inspector(min = 0.0)]
    pub normal_rejection: f32,
    /// Controls the amount of light distance falloff. 0.0 to disable falloff
    pub falloff: f32,
    pub square_falloff: bool,
    /// SSGI Brightness
    #[inspector(min = 0.0, max = 100.0)]
    pub brightness: f32,
    /// Non-PBR rough specular with simple fresnel
    #[inspector(min = 0.0, max = 100.0)]
    pub rough_specular: f32,
    /// Non-PBR rough specular fresnel sharpness
    #[inspector(min = 0.1, max = 100.0)]
    pub rough_specular_sharpness: f32,
    /// How much light we accept from things pointing away from our position
    /// Allows us to still sample the color even if we hit something from the backside
    /// This is needed for top down where only the tops of things are visible
    /// 2.0 is needed for steep top down
    #[inspector(min = -2.0, max = 2.0)]
    pub backside_illumination: f32,
    /// Minimum mip to use for depth samples
    #[inspector(min = 0.0, max = 5.0)]
    pub depth_mip_min: f32,
    /// Minimum mip to use for normal, motion vector, color, samples
    #[inspector(min = 0.0, max = 5.0)]
    pub mip_min: f32,
    /// Maximum mip to use for normal, motion vector, color, samples
    /// Max is used for ray march steps further away from the ray origin
    #[inspector(min = 0.0, max = 5.0)]
    pub mip_max: f32,
    #[inspector(min = 0.0, max = 1.0)]
    /// How much cascade intervals overlap.
    pub interval_overlap: f32,
    /// Min distance each interval travels, the distance is screen space,
    /// but the unit only applies for cascade 0, higher cascades scale up non-linearly
    #[inspector(min = 1.0, max = 1000.0)]
    pub cascade_0_dist: f32,
    // If this is false defined there will **less** ray march steps and it will be faster
    // but there can be more light leaking / inconsistencies
    pub divide_steps_by_square_of_cascade_exp: bool,
    // How much the raw horizon occlusion is used. Leave at 0.0 for bitmask occlusion;
    #[inspector(min = 0.0, max = 100.0)]
    pub horizon_occlusion: f32,
}

impl Default for SSGIPass {
    fn default() -> Self {
        SSGIPass {
            render_scale: 4,
            cascade_0_directions: 16,
            cascade_count: 6,
            jitter_probe_position: true,
            jitter_probe_direction: true,
            noise_frame_period: 8,
            distance_rejection: 2.0,
            normal_rejection: 100.0,
            falloff: 1.0,
            square_falloff: false,
            brightness: 1.0,
            rough_specular: 1.0,
            rough_specular_sharpness: 3.0,
            backside_illumination: 0.0,
            depth_mip_min: 0.0,
            mip_min: 2.0,
            mip_max: 4.0,
            interval_overlap: 1.0,
            cascade_0_dist: 21.0,
            divide_steps_by_square_of_cascade_exp: true,
            horizon_occlusion: 0.0,
        }
    }
}

impl SSGIPass {
    pub fn key(&self) -> SSGIPipelineKey {
        SSGIPipelineKey {
            jitter_probe_position: self.jitter_probe_position,
            jitter_probe_direction: self.jitter_probe_direction,
            noise_frame_period: self.noise_frame_period,
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct SSGIPipelineKey {
    pub jitter_probe_position: bool,
    pub jitter_probe_direction: bool,
    pub noise_frame_period: u32,
}
impl SSGIPipelineKey {
    pub fn shader_defs(&self, shader_defs: &mut Vec<ShaderDefVal>) {
        shader_defs.extend_from_slice(&[ShaderDefVal::UInt(
            "NOISE_FRAME_PERIOD".to_string(),
            self.noise_frame_period,
        )]);
        if self.jitter_probe_position {
            shader_defs.push("JITTER_PROBE_POSITION".into());
        }
        if self.jitter_probe_direction {
            shader_defs.push("JITTER_PROBE_DIRECTION".into());
        }
    }
}

pub const SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(845720938457230948);

pub const CASCADE_FORMAT: TextureFormat = TextureFormat::Rgba32Uint;

#[derive(Component, Clone, Copy, ShaderType, Debug, Default)]
pub struct SSGIConfig {
    cas_w: u32,
    cas_h: u32,
    cascade_n: u32,
    directions: u32,
    cas_0_directions: u32,
    cas_0_render_scale: u32,
    cascade_count: u32,
    render_scale: u32,
    distance_rejection: f32,
    normal_rejection: f32,
    falloff: f32,
    square_falloff: u32,
    brightness: f32,
    backside_illumination: f32,
    depth_mip_min: f32,
    mip_min: f32,
    mip_max: f32,
    interval_overlap: f32,
    cascade_0_dist: f32,
    divide_steps_by_square_of_cascade_exp: u32,
    horizon_occlusion: f32,
    _webgl2_padding_1: f32,
    _webgl2_padding_2: f32,
    _webgl2_padding_3: f32,
}

pub struct SSGISamplePlugin;
impl Plugin for SSGISamplePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SHADER_HANDLE,
            "../assets/shaders/ssgi.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<SSGIPass>()
            .add_plugins(ExtractComponentPlugin::<SSGIPass>::default());

        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(Render, prepare_textures.in_set(RenderSet::PrepareResources))
            .init_resource::<SpecializedRenderPipelines<SSGILayout>>()
            .add_systems(Render, (prepare_pipelines.in_set(RenderSet::Prepare),))
            .add_render_graph_node::<ViewNodeRunner<SSGIOpaquePass3dPbrLightingNode>>(
                core_3d::graph::NAME,
                SSGI_PASS,
            )
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[
                    DownsampleNode::NAME,
                    SSGI_PASS,
                    core_3d::graph::node::START_MAIN_PASS,
                ],
            );
    }

    fn finish(&self, app: &mut App) {
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<SSGILayout>();
    }
}

pub const SSGI_PASS: &str = "ssgi_pass_3d";
#[derive(Default)]
pub struct SSGIOpaquePass3dPbrLightingNode;

impl ViewNode for SSGIOpaquePass3dPbrLightingNode {
    type ViewQuery = (
        &'static ViewUniformOffset,
        &'static SSGIPipeline,
        &'static PrevFrameTexture,
        &'static PrepassDownsampleTextures,
        &'static SSGITextures,
        &'static SSGIPass,
        // todo webgl &'static DisocclusionTextures,
    );

    fn run(
        &self,
        _graph_context: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            view_uniform_offset,
            ssgi_pipelines,
            prev_frame_tex,
            prepass_downsample_texture,
            ssgi_textures,
            ssgi_pass,
            // todo webgl disocclusion_textures,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let ssgi_lighting_layout = world.resource::<SSGILayout>();
        let images = world.resource::<RenderAssets<Image>>();

        let Some(ssgi_pipeline) =
            pipeline_cache.get_render_pipeline(ssgi_pipelines.ssgi_pipeline_id)
        else {
            return Ok(());
        };

        let blue_noise_tex = image!(images, &resource!(world, BlueNoise).0);
        let nearest_sampler = nearest_sampler(render_context.render_device());
        let linear_sampler = linear_sampler(render_context.render_device());

        for cascade_n in (0..ssgi_pass.cascade_count as usize).rev() {
            let scale = 1 << cascade_n;

            let directions = ssgi_pass.cascade_0_directions * (1 << (cascade_n * 1));

            let cascade_n_u32 = cascade_n as u32;
            let config = SSGIConfig {
                cas_w: ssgi_textures.data_textures1[cascade_n].texture.width()
                    / ssgi_pass.cascade_0_directions,
                cas_h: ssgi_textures.data_textures1[cascade_n].texture.height(),
                cascade_n: cascade_n_u32,
                directions,
                cas_0_directions: ssgi_pass.cascade_0_directions,
                cas_0_render_scale: ssgi_pass.render_scale,
                cascade_count: ssgi_pass.cascade_count,
                render_scale: ssgi_pass.render_scale * scale,
                distance_rejection: ssgi_pass.distance_rejection,
                normal_rejection: ssgi_pass.normal_rejection,
                falloff: ssgi_pass.falloff,
                square_falloff: ssgi_pass.square_falloff as u32,
                brightness: ssgi_pass.brightness,
                backside_illumination: ssgi_pass.backside_illumination,
                depth_mip_min: ssgi_pass.depth_mip_min,
                mip_min: ssgi_pass.mip_min,
                mip_max: ssgi_pass.mip_max,
                interval_overlap: ssgi_pass.interval_overlap,
                cascade_0_dist: ssgi_pass.cascade_0_dist,
                divide_steps_by_square_of_cascade_exp: ssgi_pass
                    .divide_steps_by_square_of_cascade_exp
                    as u32,
                horizon_occlusion: ssgi_pass.horizon_occlusion,
                _webgl2_padding_1: 0.0,
                _webgl2_padding_2: 0.0,
                _webgl2_padding_3: 0.0,
            };

            let uniform = uniform_buffer(config, render_context, "SSGI Config Uniform");
            {
                let mut cas_read_tex_index = cascade_n + 1;

                if cas_read_tex_index >= ssgi_pass.cascade_count as usize {
                    cas_read_tex_index = 0; // Wont be used in this, just as placeholder binding
                }

                let bind_group_1 = render_context.render_device().create_bind_group(
                    "ssgi_lighting_layout_group_1",
                    &ssgi_lighting_layout.bind_group_layout,
                    &BindGroupEntries::with_indices((
                        (0, view_binding(world)),
                        (9, globals_binding(world)),
                        (101, &prev_frame_tex.texture.default_view),
                        (102, &prepass_downsample_texture.normals.default_view),
                        (103, &prepass_downsample_texture.depth.default_view),
                        (104, &prepass_downsample_texture.motion.default_view),
                        (105, &nearest_sampler),
                        (106, &linear_sampler),
                        // todo webgl (8, &disocclusion_textures.output.default_view),
                        (109, uniform.as_entire_binding()),
                        (BLUE_NOISE_ENTRY_N, &blue_noise_tex.texture_view),
                        (
                            110,
                            &ssgi_textures.data_textures1[cas_read_tex_index].default_view,
                        ),
                        (
                            111,
                            &ssgi_textures.data_textures2[cas_read_tex_index].default_view,
                        ),
                    )),
                );

                let attachments = [
                    Some(RenderPassColorAttachment {
                        view: &ssgi_textures.data_textures1[cascade_n].default_view,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                    Some(RenderPassColorAttachment {
                        view: &ssgi_textures.data_textures2[cascade_n].default_view,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                ];

                run_pass(
                    render_context,
                    "ssgi_lighting_pass",
                    &attachments,
                    ssgi_pipeline,
                    view_uniform_offset,
                    bind_group_1,
                );
            }
        }

        Ok(())
    }
}

fn run_pass(
    render_context: &mut RenderContext,
    pass_name: &str,
    attachments: &[Option<RenderPassColorAttachment<'_>>],
    pipeline: &RenderPipeline,
    view_uniform_offset: &ViewUniformOffset,
    bind_group: BindGroup,
) {
    let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some(pass_name),
        color_attachments: &attachments,
        depth_stencil_attachment: None,
    });

    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
    render_pass.draw(0..3, 0..1);
}

impl FromWorld for SSGILayout {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ssgi_lighting_layout"),
            entries: &[
                view_layout_entry(0),
                globals_layout_entry(9),
                ftexture_layout_entry(101, TextureViewDimension::D2), // Prev frame
                ftexture_layout_entry(102, TextureViewDimension::D2), // Prepass Downsample Normals
                ftexture_layout_entry(103, TextureViewDimension::D2), // Prepass Downsample Depth
                ftexture_layout_entry(104, TextureViewDimension::D2), // Prepass Downsample Motion
                fsampler_layout_entry(105),                           // Nearest Sampler
                fsampler_layout_entry(106),                           // Linear Sampler
                // todo webgl ftexture_layout_entry(108, TextureViewDimension::D2), // Disocclusion
                uniform_layout_entry(109, SSGIConfig::min_size()),
                ftexture_layout_entry(BLUE_NOISE_ENTRY_N, TextureViewDimension::D2Array), // Blue Noise
                utexture_layout_entry(110, TextureViewDimension::D2), // Higher Cascade Data Texture 1
                utexture_layout_entry(111, TextureViewDimension::D2), // Higher Cascade Data Texture 2
            ],
        });

        #[cfg(not(all(feature = "file_watcher")))]
        let shader = SHADER_HANDLE;
        #[cfg(all(feature = "file_watcher"))]
        let shader = {
            let asset_server = world.resource_mut::<bevy::prelude::AssetServer>();
            asset_server.load("shaders/ssgi.wgsl")
        };

        Self {
            bind_group_layout: layout,
            ssgi_shader: shader,
        }
    }
}

#[derive(Resource)]
pub struct SSGILayout {
    bind_group_layout: BindGroupLayout,
    pub ssgi_shader: Handle<Shader>,
}

#[derive(Component)]
pub struct SSGIPipeline {
    pub ssgi_pipeline_id: CachedRenderPipelineId,
}

impl SpecializedRenderPipeline for SSGILayout {
    type Key = SSGIPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();

        #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
        shader_defs.push("WEBGL2".into());

        #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
        shader_defs.push("SIXTEEN_BYTE_ALIGNMENT".into());

        key.shader_defs(&mut shader_defs);

        shader_defs.extend_from_slice(&[
            ShaderDefVal::UInt("BLUE_NOISE_GROUP_N".to_string(), 0),
            shader_def_uint!(BLUE_NOISE_ENTRY_N),
            shader_def_uint!(BLUE_NOISE_DIMS),
        ]);

        // Always true, since we're in the deferred lighting pipeline
        shader_defs.push("DEFERRED_PREPASS".into());

        RenderPipelineDescriptor {
            label: Some("ssgi_lighting_pipeline".into()),
            layout: vec![self.bind_group_layout.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.ssgi_shader.clone(),
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![
                    Some(ColorTargetState {
                        format: CASCADE_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: CASCADE_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            push_constant_ranges: vec![],
        }
    }
}

pub fn prepare_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SSGILayout>>,
    ssgi_lighting_layout: Res<SSGILayout>,
    views: Query<(Entity, &SSGIPass)>,
) {
    for (entity, ssgi_pass) in &views {
        let ssgi_pipeline_id: CachedRenderPipelineId =
            pipelines.specialize(&pipeline_cache, &ssgi_lighting_layout, ssgi_pass.key());
        commands
            .entity(entity)
            .insert(SSGIPipeline { ssgi_pipeline_id });
    }
}

#[derive(Component, Clone)]
pub struct SSGITextures {
    pub data_textures1: Vec<CachedTexture>,
    pub data_textures2: Vec<CachedTexture>,
}

fn prepare_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, &SSGIPass)>,
) {
    for (entity, camera, ssgi_pass) in &views {
        if let Some(physical_viewport_size) = camera.physical_viewport_size {
            let mut data_textures1 = Vec::new();
            let mut data_textures2 = Vec::new();

            let (cas_0_w, cas_0_h) = (
                physical_viewport_size.x / ssgi_pass.render_scale,
                physical_viewport_size.y / ssgi_pass.render_scale,
            );

            for cascade_n in 0..ssgi_pass.cascade_count {
                let scale = 1 << cascade_n;
                let directions = ssgi_pass.cascade_0_directions * (1 << (cascade_n * 1));

                let mut texture_descriptor = TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        depth_or_array_layers: 1,
                        width: ((cas_0_w / scale) * 4).max(1),
                        height: ((cas_0_h / scale) * directions / 4).max(1),
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: CASCADE_FORMAT,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                };

                texture_descriptor.label = Some("CascadeDataTexture1");
                data_textures1.push(texture_cache.get(&render_device, texture_descriptor.clone()));
                texture_descriptor.label = Some("CascadeDataTexture2");
                data_textures2.push(texture_cache.get(&render_device, texture_descriptor.clone()));
            }

            commands.entity(entity).insert(SSGITextures {
                data_textures1,
                data_textures2,
            });
        }
    }
}
