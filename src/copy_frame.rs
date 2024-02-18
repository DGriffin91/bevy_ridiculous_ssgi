use crate::bind_group_utils::linear_sampler;
use bevy::{
    asset::load_internal_asset,
    core_pipeline::{
        core_3d::graph::{Core3d, Node3d},
        fullscreen_vertex_shader::fullscreen_shader_vertex_state,
    },
    prelude::*,
    render::{
        camera::ExtractedCamera,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel},
        render_resource::{
            BindGroupEntries, BindGroupLayout, BindGroupLayoutEntry, BindingType,
            CachedRenderPipelineId, ColorTargetState, ColorWrites, Extent3d, FragmentState,
            MultisampleState, Operations, PipelineCache, PrimitiveState, RenderPassColorAttachment,
            RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, Sampler,
            SamplerBindingType, ShaderStages, TextureAspect, TextureDescriptor, TextureDimension,
            TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDescriptor,
            TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        texture::{CachedTexture, TextureCache},
        view::{ExtractedView, ViewTarget},
        Render, RenderApp, RenderSet,
    },
};

const DOWNSAMPLE_COLOR_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

#[derive(Component, ExtractComponent, Clone)]
pub struct CopyFrame {
    mip_levels: u8,
}

impl Default for CopyFrame {
    fn default() -> Self {
        CopyFrame { mip_levels: 5 }
    }
}

const SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(23059847523049077);
pub struct CopyFramePlugin;
impl Plugin for CopyFramePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, SHADER_HANDLE, "copy_frame.wgsl", Shader::from_wgsl);
        app.add_plugins(ExtractComponentPlugin::<CopyFrame>::default());
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(Render, prepare_textures.in_set(RenderSet::PrepareResources))
            .add_render_graph_node::<FrameCopyNode>(Core3d, FrameCopyLabel)
            .add_render_graph_edges(
                Core3d,
                (Node3d::MainOpaquePass, FrameCopyLabel, Node3d::EndMainPass),
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };
        render_app.init_resource::<CopyFramePipeline>();
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct FrameCopyLabel;

pub struct FrameCopyNode {
    query: QueryState<
        (
            &'static ViewTarget,
            &'static PrevFrameTexture,
            &'static CopyFrame,
        ),
        With<ExtractedView>,
    >,
}

impl FromWorld for FrameCopyNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl FrameCopyNode {
    pub const NAME: &'static str = "copy_frame";
}

impl Node for FrameCopyNode {
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

        let Ok((view_target, prev_frame_tex, copy_frame)) =
            self.query.get_manual(world, view_entity)
        else {
            return Ok(());
        };

        if !view_target.is_hdr() {
            println!("view_target is not HDR");
            return Ok(());
        }

        let copy_frame_pipeline = world.resource::<CopyFramePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(pipeline) = pipeline_cache.get_render_pipeline(copy_frame_pipeline.pipeline_id)
        else {
            return Ok(());
        };

        let mip_levels = copy_frame.mip_levels as u32;

        #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
        {
            // For WebGL we can't read and write to the same texture at the same time,
            // so we first create the mips with separate textures, then copy them to
            // the single texture that has multiple mip levels
            run_pass(
                render_context,
                copy_frame_pipeline,
                view_target
                    .main_texture()
                    .create_view(&TextureViewDescriptor {
                        label: Some("MIP_SRC"),
                        format: Some(DOWNSAMPLE_COLOR_FORMAT),
                        dimension: Some(TextureViewDimension::D2),
                        ..default()
                    }),
                prev_frame_tex.temp_texture[0].default_view.clone(),
                pipeline,
            );
            for i in 0..mip_levels - 2 {
                run_pass(
                    render_context,
                    copy_frame_pipeline,
                    prev_frame_tex.temp_texture[i as usize].default_view.clone(),
                    prev_frame_tex.temp_texture[i as usize + 1]
                        .default_view
                        .clone(),
                    pipeline,
                );
            }
            for i in 0..mip_levels - 1 {
                run_pass(
                    render_context,
                    copy_frame_pipeline,
                    prev_frame_tex.temp_texture[i as usize].default_view.clone(),
                    prev_frame_tex
                        .texture
                        .texture
                        .create_view(&TextureViewDescriptor {
                            label: Some("MIP_DST"),
                            format: Some(DOWNSAMPLE_COLOR_FORMAT),
                            dimension: Some(TextureViewDimension::D2),
                            aspect: TextureAspect::All,
                            base_mip_level: i + 1,
                            mip_level_count: Some(1),
                            base_array_layer: 0,
                            array_layer_count: Some(1),
                        }),
                    pipeline,
                );
            }
        }

        #[cfg(not(all(feature = "webgl", target_arch = "wasm32")))]
        {
            run_pass(
                render_context,
                copy_frame_pipeline,
                view_target
                    .main_texture()
                    .create_view(&TextureViewDescriptor {
                        label: Some("MIP_SRC"),
                        format: Some(DOWNSAMPLE_COLOR_FORMAT),
                        dimension: Some(TextureViewDimension::D2),
                        ..default()
                    }),
                prev_frame_tex
                    .texture
                    .texture
                    .create_view(&TextureViewDescriptor {
                        label: Some("MIP_DST"),
                        format: Some(DOWNSAMPLE_COLOR_FORMAT),
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::All,
                        base_mip_level: 0,
                        mip_level_count: Some(1),
                        base_array_layer: 0,
                        array_layer_count: Some(1),
                    }),
                pipeline,
            );
            for i in 0..mip_levels - 1 {
                run_pass(
                    render_context,
                    copy_frame_pipeline,
                    prev_frame_tex
                        .texture
                        .texture
                        .create_view(&TextureViewDescriptor {
                            label: Some("MIP_SRC"),
                            format: Some(DOWNSAMPLE_COLOR_FORMAT),
                            dimension: Some(TextureViewDimension::D2),
                            aspect: TextureAspect::All,
                            base_mip_level: i,
                            mip_level_count: Some(1),
                            base_array_layer: 0,
                            array_layer_count: Some(1),
                        }),
                    prev_frame_tex
                        .texture
                        .texture
                        .create_view(&TextureViewDescriptor {
                            label: Some("MIP_DST"),
                            format: Some(DOWNSAMPLE_COLOR_FORMAT),
                            dimension: Some(TextureViewDimension::D2),
                            aspect: TextureAspect::All,
                            base_mip_level: i + 1,
                            mip_level_count: Some(1),
                            base_array_layer: 0,
                            array_layer_count: Some(1),
                        }),
                    pipeline,
                );
            }
        }

        Ok(())
    }
}

fn run_pass(
    render_context: &mut RenderContext,
    copy_frame_pipeline: &CopyFramePipeline,
    src_view: TextureView,
    dst_view: TextureView,
    pipeline: &RenderPipeline,
) {
    let bind_group = render_context.render_device().create_bind_group(
        "post_process_bind_group",
        &copy_frame_pipeline.layout,
        // It's important for this to match the BindGroupLayout defined in the PostProcessPipeline
        &BindGroupEntries::sequential((&src_view, &copy_frame_pipeline.sampler)),
    );
    let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("mip_chain_pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &dst_view,
            resolve_target: None,
            ops: Operations::default(),
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(0, &bind_group, &[]);
    render_pass.draw(0..3, 0..1);
}

#[derive(Resource)]
struct CopyFramePipeline {
    layout: BindGroupLayout,
    sampler: Sampler,
    pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for CopyFramePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ];

        let layout = world
            .resource::<RenderDevice>()
            .create_bind_group_layout(Some("copy_frame_bind_group_layout"), &entries);

        let sampler = linear_sampler(render_device);

        let pipeline_id =
            world
                .resource_mut::<PipelineCache>()
                .queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some("copy_frame_pipeline".into()),
                    layout: vec![layout.clone()],
                    vertex: fullscreen_shader_vertex_state(),
                    fragment: Some(FragmentState {
                        shader: SHADER_HANDLE,
                        shader_defs: vec![],
                        entry_point: "fragment".into(),
                        targets: vec![Some(ColorTargetState {
                            format: DOWNSAMPLE_COLOR_FORMAT,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        })],
                    }),
                    primitive: PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                    push_constant_ranges: vec![],
                });

        Self {
            layout,
            sampler,
            pipeline_id,
        }
    }
}

#[derive(Component)]
pub struct PrevFrameTexture {
    pub texture: CachedTexture,
    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
    temp_texture: Vec<CachedTexture>,
}

fn prepare_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, &ExtractedView, &CopyFrame)>,
) {
    for (entity, camera, _view, copy_frame) in &views {
        if let Some(physical_viewport_size) = camera.physical_viewport_size {
            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            let mut temp_texture_set = Vec::new();
            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            for i in 1..copy_frame.mip_levels as u32 {
                let scale = 1 << i;
                let mut texture_descriptor = TextureDescriptor {
                    label: None,
                    size: Extent3d {
                        depth_or_array_layers: 1,
                        width: physical_viewport_size.x / scale,
                        height: physical_viewport_size.y / scale,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format: DOWNSAMPLE_COLOR_FORMAT,
                    usage: TextureUsages::RENDER_ATTACHMENT
                        | TextureUsages::TEXTURE_BINDING
                        | TextureUsages::COPY_DST,
                    view_formats: &[],
                };
                texture_descriptor.label = Some("temp_prev_frame_texture");
                temp_texture_set
                    .push(texture_cache.get(&render_device, texture_descriptor.clone()));
            }

            let mut texture_descriptor = TextureDescriptor {
                label: None,
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: physical_viewport_size.x,
                    height: physical_viewport_size.y,
                },
                mip_level_count: copy_frame.mip_levels as u32,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: DOWNSAMPLE_COLOR_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            };

            texture_descriptor.label = Some("prev_frame_texture");
            let prev_frame_texture = texture_cache.get(&render_device, texture_descriptor.clone());

            commands.entity(entity).insert(PrevFrameTexture {
                texture: prev_frame_texture,
                #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
                temp_texture: temp_texture_set,
            });
        }
    }
}
