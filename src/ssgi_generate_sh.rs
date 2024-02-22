use bevy::{
    asset::load_internal_asset,
    core::FrameCount,
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        fullscreen_vertex_shader::fullscreen_shader_vertex_state,
    },
    prelude::*,
    render::{
        camera::ExtractedCamera,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::RenderAssets,
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel},
        render_resource::{
            BindGroupEntries, BindGroupLayout, CachedRenderPipelineId, ColorTargetState,
            ColorWrites, Extent3d, FragmentState, MultisampleState, Operations, PipelineCache,
            PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor,
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
        ftexture_layout_entry, globals_binding, globals_layout_entry, uniform_buffer,
        uniform_layout_entry, utexture_layout_entry, view_binding, view_layout_entry,
    },
    image,
    prepass_downsample::PrepassDownsampleTextures,
    resource, shader_def_uint,
    ssgi::{SSGILabel, SSGIPass, SSGIPipelineKey, SSGITextures},
    BlueNoise, BLUE_NOISE_DIMS, BLUE_NOISE_ENTRY_N,
};

const SH_DATA_FORMAT: TextureFormat = TextureFormat::Rgba32Uint;
const SH_HISTORY_POS_FORMAT: TextureFormat = TextureFormat::Rgba32Float;
pub const SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(40958237405983745);

#[derive(Component, ExtractComponent, Clone, Reflect, InspectorOptions)]
#[reflect(Component, InspectorOptions)]
pub struct SSGIGenerateSH {
    /// How much to blend in previous reprojected probes. 1.0 for only using the current frame,
    /// lower numbers uses more of the previous accumulation
    #[inspector(min = 0.05, max = 1.0)]
    hysteresis: f32,
}

impl Default for SSGIGenerateSH {
    fn default() -> Self {
        SSGIGenerateSH { hysteresis: 0.2 }
    }
}

#[derive(Component, Clone, Copy, ShaderType, Debug, Default)]
struct SSGIGenerateSHConfig {
    cas_w: u32,
    cas_h: u32,
    directions: u32,
    render_scale: u32,
    cascade_count: u32,
    hysteresis: f32,
    _webgl2_padding_1: f32,
    _webgl2_padding_2: f32,
}

pub struct SSGIGenerateSHPlugin;
impl Plugin for SSGIGenerateSHPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            SHADER_HANDLE,
            "../assets/shaders/ssgi_generate_sh.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<SSGIGenerateSH>()
            .add_plugins(ExtractComponentPlugin::<SSGIGenerateSH>::default());
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(Render, prepare_textures.in_set(RenderSet::PrepareResources))
            .init_resource::<SpecializedRenderPipelines<SSGIGenerateSHLayout>>()
            .add_systems(Render, (prepare_pipelines.in_set(RenderSet::Prepare),))
            .add_render_graph_node::<SSGIGenerateSHNode>(Core3d, SSGIGenerateSHLabel)
            .add_render_graph_edges(
                Core3d,
                (SSGILabel, SSGIGenerateSHLabel, Node3d::StartMainPass),
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };
        render_app.init_resource::<SSGIGenerateSHLayout>();
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct SSGIGenerateSHLabel;

pub struct SSGIGenerateSHNode {
    query: QueryState<
        (
            &'static ViewUniformOffset,
            &'static SSGITextures,
            &'static SSGISHTextures,
            &'static SSGIGenerateSHPipeline,
            &'static PrepassDownsampleTextures,
            &'static SSGIPass,
            &'static SSGIGenerateSH,
        ),
        With<ExtractedView>,
    >,
}

impl FromWorld for SSGIGenerateSHNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl SSGIGenerateSHNode {
    pub const NAME: &'static str = "GenerateSSGISH";
}

impl Node for SSGIGenerateSHNode {
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
            sh_texture,
            pipeline,
            prepass_downsample_texture,
            ssgi_pass,
            ssgi_generate_sh,
        )) = self.query.get_manual(world, view_entity)
        else {
            return Ok(());
        };

        let ssgi_sh_pipeline = world.resource::<SSGIGenerateSHLayout>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline.pipeline_id) else {
            return Ok(());
        };

        let images = world.resource::<RenderAssets<Image>>();
        let blue_noise_tex = image!(images, &resource!(world, BlueNoise).0);

        let config = SSGIGenerateSHConfig {
            cas_w: ssgi_textures.data_textures1[0].texture.width(),
            cas_h: ssgi_textures.data_textures1[0].texture.height(),
            directions: ssgi_pass.cascade_0_directions,
            render_scale: ssgi_pass.render_scale,
            cascade_count: ssgi_pass.cascade_count,
            hysteresis: ssgi_generate_sh.hysteresis,
            ..default()
        };
        let uniform = uniform_buffer(config, render_context, "SSGI Generate SH Config Uniform");
        let bind_group = render_context.render_device().create_bind_group(
            "ssgi_generate_sh_bind_group",
            &ssgi_sh_pipeline.layout,
            &BindGroupEntries::with_indices((
                (0, view_binding(world)),
                (9, globals_binding(world)),
                (101, &ssgi_textures.data_textures1[0].default_view),
                (111, &ssgi_textures.data_textures2[0].default_view),
                (102, &prepass_downsample_texture.normals.default_view),
                (103, &prepass_downsample_texture.depth.default_view),
                (104, &prepass_downsample_texture.motion.default_view),
                (105, &sh_texture.read.default_view),
                (106, &sh_texture.pos_read.default_view),
                (109, uniform.as_entire_binding()),
                (BLUE_NOISE_ENTRY_N, &blue_noise_tex.texture_view),
            )),
        );

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("ssgi_generate_sh_pass"),
            color_attachments: &[
                Some(RenderPassColorAttachment {
                    view: &sh_texture.write.default_view,
                    resolve_target: None,
                    ops: Operations::default(),
                }),
                Some(RenderPassColorAttachment {
                    view: &sh_texture.pos_write.default_view,
                    resolve_target: None,
                    ops: Operations::default(),
                }),
            ],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}

#[derive(Resource)]
pub struct SSGIGenerateSHLayout {
    pub layout: BindGroupLayout,
    pub shader: Handle<Shader>,
}

#[derive(Component)]
struct SSGIGenerateSHPipeline {
    pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for SSGIGenerateSHLayout {
    fn from_world(world: &mut World) -> Self {
        let entries = vec![
            view_layout_entry(0),
            globals_layout_entry(9),
            utexture_layout_entry(101, TextureViewDimension::D2),
            utexture_layout_entry(111, TextureViewDimension::D2),
            ftexture_layout_entry(102, TextureViewDimension::D2), // Prepass Downsample Normals
            ftexture_layout_entry(103, TextureViewDimension::D2), // Prepass Downsample Depth
            ftexture_layout_entry(104, TextureViewDimension::D2), // Prepass Downsample Motion
            utexture_layout_entry(105, TextureViewDimension::D2), // Prev SH
            ftexture_layout_entry(106, TextureViewDimension::D2), // Pos Read
            uniform_layout_entry(109, SSGIGenerateSHConfig::min_size()),
            ftexture_layout_entry(BLUE_NOISE_ENTRY_N, TextureViewDimension::D2Array), // Blue Noise
        ];

        let layout = world
            .resource::<RenderDevice>()
            .create_bind_group_layout(Some("ssgi_generate_sh_bind_group_layout"), &entries);

        #[cfg(not(all(feature = "file_watcher")))]
        let shader = SHADER_HANDLE;
        #[cfg(all(feature = "file_watcher"))]
        let shader = {
            let asset_server = world.resource_mut::<bevy::prelude::AssetServer>();
            asset_server.load("shaders/ssgi_generate_sh.wgsl")
        };

        Self { layout, shader }
    }
}

impl SpecializedRenderPipeline for SSGIGenerateSHLayout {
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
            label: Some("ssgi_generate_sh_pipeline".into()),
            layout: vec![self.layout.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: "fragment".into(),
                targets: vec![
                    Some(ColorTargetState {
                        format: SH_DATA_FORMAT,
                        blend: None,
                        write_mask: ColorWrites::ALL,
                    }),
                    Some(ColorTargetState {
                        format: SH_HISTORY_POS_FORMAT,
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

#[derive(Component)]
pub struct SSGISHTextures {
    pub read: CachedTexture,
    pub write: CachedTexture,
    pub pos_read: CachedTexture,
    pub pos_write: CachedTexture,
}

fn prepare_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, &ExtractedView, &SSGIPass), With<SSGIGenerateSH>>,
    frame_count: Res<FrameCount>,
) {
    for (entity, camera, _view, ssgi_pass) in &views {
        if let Some(physical_viewport_size) = camera.physical_viewport_size {
            let mut texture_descriptor = TextureDescriptor {
                label: None,
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: physical_viewport_size.x / ssgi_pass.render_scale,
                    height: physical_viewport_size.y / ssgi_pass.render_scale,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SH_DATA_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            };
            let mut sh_history_pos_texture_descriptor = TextureDescriptor {
                label: None,
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: physical_viewport_size.x / ssgi_pass.render_scale,
                    height: physical_viewport_size.y / ssgi_pass.render_scale,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: SH_HISTORY_POS_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            };

            texture_descriptor.label = Some("ssgi_sh_a");
            let ssgi_sh_texture_a = texture_cache.get(&render_device, texture_descriptor.clone());
            texture_descriptor.label = Some("ssgi_sh_b");
            let ssgi_sh_texture_b = texture_cache.get(&render_device, texture_descriptor.clone());

            sh_history_pos_texture_descriptor.label = Some("ssgi_sh_history_pos_a");
            let ssgi_sh_history_pos_texture_a =
                texture_cache.get(&render_device, sh_history_pos_texture_descriptor.clone());
            sh_history_pos_texture_descriptor.label = Some("ssgi_sh_history_pos_b");
            let ssgi_sh_history_pos_texture_b =
                texture_cache.get(&render_device, sh_history_pos_texture_descriptor.clone());

            let textures = if frame_count.0 % 2 == 0 {
                SSGISHTextures {
                    write: ssgi_sh_texture_a,
                    read: ssgi_sh_texture_b,
                    pos_write: ssgi_sh_history_pos_texture_a,
                    pos_read: ssgi_sh_history_pos_texture_b,
                }
            } else {
                SSGISHTextures {
                    write: ssgi_sh_texture_b,
                    read: ssgi_sh_texture_a,
                    pos_write: ssgi_sh_history_pos_texture_b,
                    pos_read: ssgi_sh_history_pos_texture_a,
                }
            };
            commands.entity(entity).insert(textures);
        }
    }
}

pub fn prepare_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<SSGIGenerateSHLayout>>,
    layout: Res<SSGIGenerateSHLayout>,
    views: Query<(Entity, &SSGIPass)>,
) {
    for (entity, ssgi_pass) in &views {
        let pipeline_id: CachedRenderPipelineId =
            pipelines.specialize(&pipeline_cache, &layout, ssgi_pass.key());
        commands
            .entity(entity)
            .insert(SSGIGenerateSHPipeline { pipeline_id });
    }
}
