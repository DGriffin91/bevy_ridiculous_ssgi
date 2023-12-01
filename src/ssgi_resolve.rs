use bevy::{
    asset::load_internal_asset,
    core::FrameCount,
    core_pipeline::{core_3d, fullscreen_vertex_shader::fullscreen_shader_vertex_state},
    prelude::*,
    render::{
        camera::ExtractedCamera,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
        render_resource::{
            BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor, CachedRenderPipelineId,
            ColorTargetState, ColorWrites, Extent3d, FragmentState, MultisampleState, Operations,
            PipelineCache, PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, ShaderDefVal, ShaderType, SpecializedRenderPipeline,
            SpecializedRenderPipelines, TextureDescriptor, TextureDimension, TextureFormat,
            TextureUsages, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        texture::{CachedTexture, TextureCache},
        view::{ExtractedView, ViewUniformOffset},
        Render, RenderApp, RenderSet,
    },
};
use bevy_inspector_egui::{inspector_options::ReflectInspectorOptions, InspectorOptions};

use crate::{
    bind_group_utils::{
        fsampler_layout_entry, ftexture_layout_entry, globals_binding, globals_layout_entry,
        linear_sampler, uniform_buffer, uniform_layout_entry, utexture_layout_entry, view_binding,
        view_layout_entry,
    },
    image,
    prepass_downsample::PrepassDownsampleTextures,
    resource, shader_def_uint,
    ssgi::{SSGIPass, SSGIPipelineKey, SSGITextures},
    ssgi_generate_sh::{SSGIGenerateSHNode, SSGISHTextures},
    BlueNoise, BLUE_NOISE_DIMS, BLUE_NOISE_ENTRY_N,
};

const SH_RESOLVE_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

pub const SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(523704598327409748);

#[derive(Component, ExtractComponent, Clone, Reflect, InspectorOptions)]
#[reflect(Component, InspectorOptions)]
pub struct SSGIResolve {
    /// How much differences in position affect interpolation between probes when resolving to full resolution
    #[inspector(min = 0.0)]
    pub distance_rejection: f32,
    /// How much differences in normals affect interpolation between probes when resolving to full resolution
    #[inspector(min = 0.0)]
    pub normal_rejection: f32,
    /// How much to blend in previous reprojected resolved frame. 1.0 for only using the current frame,
    /// lower numbers uses more of the previous accumulation
    #[inspector(min = 0.05, max = 1.0)]
    hysteresis: f32,
}

impl Default for SSGIResolve {
    fn default() -> Self {
        SSGIResolve {
            distance_rejection: 2.0,
            normal_rejection: 100.0,
            hysteresis: 0.1,
        }
    }
}

#[derive(Component, Clone, Copy, ShaderType, Debug, Default)]
struct SSGIResolveConfig {
    cas_w: u32,
    cas_h: u32,
    directions: u32,
    render_scale: u32,
    cascade_count: u32,
    distance_rejection: f32,
    normal_rejection: f32,
    hysteresis: f32,
    rough_specular: f32,
    rough_specular_sharpness: f32,
    _webgl2_padding_1: f32,
    _webgl2_padding_2: f32,
}

pub struct SSGIResolvePlugin;
impl Plugin for SSGIResolvePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SHADER_HANDLE,
            "../assets/shaders/ssgi_resolve.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<SSGIResolve>()
            .add_plugins(ExtractComponentPlugin::<SSGIResolve>::default());
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(Render, prepare_textures.in_set(RenderSet::PrepareResources))
            .init_resource::<SpecializedRenderPipelines<SSGIResolveLayout>>()
            .add_render_graph_node::<SSGIResolveNode>(core_3d::graph::NAME, SSGIResolveNode::NAME)
            .add_systems(Render, (prepare_pipelines.in_set(RenderSet::Prepare),))
            .add_render_graph_edges(
                core_3d::graph::NAME,
                &[
                    SSGIGenerateSHNode::NAME,
                    SSGIResolveNode::NAME,
                    core_3d::graph::node::START_MAIN_PASS,
                ],
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };
        render_app.init_resource::<SSGIResolveLayout>();
    }
}

pub struct SSGIResolveNode {
    query: QueryState<
        (
            &'static ViewUniformOffset,
            &'static SSGITextures,
            &'static SSGIResolveTextures,
            &'static SSGIResolvePipeline,
            &'static SSGISHTextures,
            &'static PrepassDownsampleTextures,
            &'static SSGIPass,
            &'static SSGIResolve,
            // todo webgl &'static DisocclusionTextures,
            // todo webgl &'static DynamicUniformIndex<DisocclusionUniforms>,
        ),
        With<ExtractedView>,
    >,
}

impl FromWorld for SSGIResolveNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl SSGIResolveNode {
    pub const NAME: &'static str = "SSGIResolve";
}

impl Node for SSGIResolveNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        graph_context: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let view_entity = graph_context.view_entity();

        let Ok((
            view_uniform_offset,
            ssgi_textures,
            resolve_textures,
            resolve_pipeline,
            sh_texture,
            prepass_downsample_texture,
            ssgi_pass,
            ssgi_resolve,
            // todo webgl disocclusion_textures,
            // todo webgl disocclusion_uniform_index,
        )) = self.query.get_manual(world, view_entity)
        else {
            return Ok(());
        };

        let ssgi_sh_pipeline = world.resource::<SSGIResolveLayout>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) = pipeline_cache.get_render_pipeline(resolve_pipeline.pipeline_id)
        else {
            return Ok(());
        };
        let linear_sampler = linear_sampler(render_context.render_device());

        let images = world.resource::<RenderAssets<Image>>();
        let blue_noise_tex = image!(images, &resource!(world, BlueNoise).0);

        // todo webgl let disocclusion_uniforms = world.resource::<ComponentUniforms<DisocclusionUniforms>>();
        //let Some(disocclusion_uniforms) = disocclusion_uniforms.binding() else {
        //    return Ok(());
        //};

        let config = SSGIResolveConfig {
            cas_w: ssgi_textures.data_textures1[0].texture.width(),
            cas_h: ssgi_textures.data_textures1[0].texture.height(),
            directions: ssgi_pass.cascade_0_directions,
            render_scale: ssgi_pass.render_scale,
            cascade_count: ssgi_pass.cascade_count,
            rough_specular: ssgi_pass.rough_specular,
            rough_specular_sharpness: ssgi_pass.rough_specular_sharpness,
            distance_rejection: ssgi_resolve.distance_rejection,
            normal_rejection: ssgi_resolve.normal_rejection,
            hysteresis: ssgi_resolve.hysteresis,
            _webgl2_padding_1: 0.0,
            _webgl2_padding_2: 0.0,
        };

        let uniform = uniform_buffer(config, render_context, "SSGI Resolve Config Uniform");
        let bind_group = render_context.render_device().create_bind_group(
            "ssgi_resolve_bind_group",
            &ssgi_sh_pipeline.layout,
            &BindGroupEntries::with_indices((
                (0, view_binding(world)),
                (9, globals_binding(world)),
                (101, &ssgi_textures.data_textures1[0].default_view),
                (102, &prepass_downsample_texture.normals.default_view),
                (103, &prepass_downsample_texture.depth.default_view),
                (104, &prepass_downsample_texture.motion.default_view),
                // Use write since it's the one ssgi_generate_sh would have just written to
                (105, &sh_texture.write.default_view),
                (106, &resolve_textures.read.default_view),
                //(107, &disocclusion_textures.output.default_view),
                (108, &linear_sampler),
                (109, uniform.as_entire_binding()),
                (110, &sh_texture.pos_write.default_view),
                //(111, disocclusion_uniforms),
                (BLUE_NOISE_ENTRY_N, &blue_noise_tex.texture_view),
            )),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("ssgi_resolve_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &resolve_textures.write.default_view,
                resolve_target: None,
                ops: Operations::default(),
            })],
            depth_stencil_attachment: None,
        });
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(
            0,
            &bind_group,
            &[
                view_uniform_offset.offset,
                //disocclusion_uniform_index.index(),
            ],
        );
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

#[derive(Resource)]
pub struct SSGIResolveLayout {
    pub layout: BindGroupLayout,
    pub shader: Handle<Shader>,
}

#[derive(Component)]
struct SSGIResolvePipeline {
    pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for SSGIResolveLayout {
    fn from_world(world: &mut World) -> Self {
        let entries = vec![
            view_layout_entry(0),
            globals_layout_entry(9),
            utexture_layout_entry(101, TextureViewDimension::D2),
            ftexture_layout_entry(102, TextureViewDimension::D2), // Prepass Downsample Normals
            ftexture_layout_entry(103, TextureViewDimension::D2), // Prepass Downsample Depth
            ftexture_layout_entry(104, TextureViewDimension::D2), // Prepass Downsample Motion
            utexture_layout_entry(105, TextureViewDimension::D2), // SH Texture
            ftexture_layout_entry(106, TextureViewDimension::D2), // Read Resolve
            //ftexture_layout_entry(107, TextureViewDimension::D2), // Disocclusion
            fsampler_layout_entry(108), // Linear Sampler
            // Disocclusion Parameters
            //BindGroupLayoutEntry {
            //    binding: 111,
            //    ty: BindingType::Buffer {
            //        ty: BufferBindingType::Uniform,
            //        has_dynamic_offset: true,
            //        min_binding_size: Some(DisocclusionUniforms::min_size()),
            //    },
            //    visibility: ShaderStages::FRAGMENT,
            //    count: None,
            //},
            uniform_layout_entry(109, SSGIResolveConfig::min_size()),
            ftexture_layout_entry(110, TextureViewDimension::D2), // Pos / Reflection Texture
            ftexture_layout_entry(BLUE_NOISE_ENTRY_N, TextureViewDimension::D2Array), // Blue Noise
        ];

        #[cfg(not(all(feature = "file_watcher")))]
        let shader = SHADER_HANDLE;
        #[cfg(all(feature = "file_watcher"))]
        let shader = {
            let asset_server = world.resource_mut::<bevy::prelude::AssetServer>();
            asset_server.load("shaders/ssgi_resolve.wgsl")
        };

        let layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("ssgi_resolve_bind_group_layout"),
                    entries: &entries,
                });

        Self { layout, shader }
    }
}

impl SpecializedRenderPipeline for SSGIResolveLayout {
    type Key = SSGIPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut shader_defs = Vec::new();

        shader_defs.extend_from_slice(&[
            ShaderDefVal::UInt("BLUE_NOISE_GROUP_N".to_string(), 0),
            shader_def_uint!(BLUE_NOISE_ENTRY_N),
            shader_def_uint!(BLUE_NOISE_DIMS),
        ]);

        key.shader_defs(&mut shader_defs);

        RenderPipelineDescriptor {
            label: Some("ssgi_resolve_pipeline".into()),
            layout: vec![self.layout.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format: SH_RESOLVE_FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            push_constant_ranges: vec![],
        }
    }
}

#[derive(Component)]
pub struct SSGIResolveTextures {
    pub read: CachedTexture,
    pub write: CachedTexture,
}

fn prepare_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, &ExtractedView), With<SSGIResolve>>,
    frame_count: Res<FrameCount>,
) {
    for (entity, camera, _view) in &views {
        if let Some(physical_viewport_size) = camera.physical_viewport_size {
            let mut texture_descriptor = TextureDescriptor {
                label: None,
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: physical_viewport_size.x,
                    height: physical_viewport_size.y,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SH_RESOLVE_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            };

            texture_descriptor.label = Some("ssgi_resolve_a");
            let ssgi_resolve_texture_a =
                texture_cache.get(&render_device, texture_descriptor.clone());
            texture_descriptor.label = Some("ssgi_resolve_b");
            let ssgi_resolve_texture_b =
                texture_cache.get(&render_device, texture_descriptor.clone());

            let textures = if frame_count.0 % 2 == 0 {
                SSGIResolveTextures {
                    write: ssgi_resolve_texture_a,
                    read: ssgi_resolve_texture_b,
                }
            } else {
                SSGIResolveTextures {
                    write: ssgi_resolve_texture_b,
                    read: ssgi_resolve_texture_a,
                }
            };
            commands.entity(entity).insert(textures);
        }
    }
}

pub fn prepare_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SSGIResolveLayout>>,
    layout: Res<SSGIResolveLayout>,
    views: Query<(Entity, &SSGIPass)>,
) {
    for (entity, ssgi_pass) in &views {
        let pipeline_id: CachedRenderPipelineId =
            pipelines.specialize(&pipeline_cache, &layout, ssgi_pass.key());
        commands
            .entity(entity)
            .insert(SSGIResolvePipeline { pipeline_id });
    }
}
