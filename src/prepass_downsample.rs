use bevy::{
    asset::load_internal_asset,
    core::FrameCount,
    core_pipeline::{
        core_3d, fullscreen_vertex_shader::fullscreen_shader_vertex_state,
        prepass::ViewPrepassTextures,
    },
    prelude::*,
    render::{
        camera::ExtractedCamera,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_graph::{Node, NodeRunError, RenderGraphApp, RenderGraphContext},
        render_resource::{
            BindGroupEntries, BindGroupLayout, BindGroupLayoutDescriptor, CachedRenderPipelineId,
            ColorTargetState, ColorWrites, Extent3d, FragmentState, MultisampleState, Operations,
            PipelineCache, PrimitiveState, RenderPassColorAttachment, RenderPassDescriptor,
            RenderPipelineDescriptor, Sampler, TextureAspect, TextureDescriptor, TextureDimension,
            TextureFormat, TextureUsages, TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderContext, RenderDevice},
        texture::{CachedTexture, TextureCache},
        view::{ExtractedView, ViewTarget, ViewUniformOffset},
        Render, RenderApp, RenderSet,
    },
};

use crate::bind_group_utils::{
    dtexture_layout_entry, fsampler_layout_entry, ftexture_layout_entry, globals_binding,
    globals_layout_entry, nearest_sampler, utexture_layout_entry, view_binding, view_layout_entry,
};

#[cfg(all(feature = "webgl", target_arch = "wasm32"))]
const DOWNSAMPLE_NORMALS_FORMAT: TextureFormat = TextureFormat::Rg16Float;
#[cfg(not(all(feature = "webgl", target_arch = "wasm32")))]
const DOWNSAMPLE_NORMALS_FORMAT: TextureFormat = TextureFormat::Rg16Unorm;
const DOWNSAMPLE_DEPTH_FORMAT: TextureFormat = TextureFormat::R32Float;
const DOWNSAMPLE_MOTION_FORMAT: TextureFormat = TextureFormat::Rg16Float;

#[derive(Component, ExtractComponent, Clone)]
pub struct PrepassDownsample {
    mip_levels: u8,
}

impl Default for PrepassDownsample {
    fn default() -> Self {
        PrepassDownsample { mip_levels: 5 }
    }
}

/// Makes a copies of the prepass normals, depth, and motion vectors with mips.
pub struct PrepassDownsamplePlugin {
    node_order: Vec<&'static str>,
}

impl Default for PrepassDownsamplePlugin {
    fn default() -> Self {
        PrepassDownsamplePlugin {
            node_order: [
                core_3d::graph::node::END_PREPASSES,
                DownsampleNode::NAME,
                core_3d::graph::node::START_MAIN_PASS,
            ]
            .to_vec(),
        }
    }
}

const CONVERT_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(163429348570394285);
const DOWNSAMPLE_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(329046523092834572);
impl Plugin for PrepassDownsamplePlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            CONVERT_SHADER_HANDLE,
            "prepass_convert.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            DOWNSAMPLE_SHADER_HANDLE,
            "prepass_downsample.wgsl",
            Shader::from_wgsl
        );
        app.add_plugins(ExtractComponentPlugin::<PrepassDownsample>::default());
        let Ok(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(Render, prepare_textures.in_set(RenderSet::PrepareResources))
            .add_render_graph_node::<DownsampleNode>(core_3d::graph::NAME, DownsampleNode::NAME)
            .add_render_graph_edges(core_3d::graph::NAME, &self.node_order);
    }

    fn finish(&self, app: &mut App) {
        let render_app = match app.get_sub_app_mut(RenderApp) {
            Ok(render_app) => render_app,
            Err(_) => return,
        };
        render_app.init_resource::<PrepassDownsamplePipeline>();
    }
}

pub struct DownsampleNode {
    query: QueryState<
        (
            &'static ViewUniformOffset,
            &'static ViewTarget,
            &'static ViewPrepassTextures,
            &'static PrepassDownsampleTextures,
            &'static PrepassDownsample,
        ),
        With<ExtractedView>,
    >,
}

impl FromWorld for DownsampleNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl DownsampleNode {
    pub const NAME: &'static str = "copy_frame";
}

impl Node for DownsampleNode {
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
            view_target,
            prepass_textures,
            downsample_textures,
            prepass_downsample,
        )) = self.query.get_manual(world, view_entity)
        else {
            return Ok(());
        };

        if !view_target.is_hdr() {
            println!("view_target is not HDR");
            return Ok(());
        }

        let copy_frame_pipeline = world.resource::<PrepassDownsamplePipeline>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let Some(convert_pipeline) =
            pipeline_cache.get_render_pipeline(copy_frame_pipeline.convert_pipeline_id)
        else {
            return Ok(());
        };
        let Some(downsample_pipeline) =
            pipeline_cache.get_render_pipeline(copy_frame_pipeline.downsample_pipeline_id)
        else {
            return Ok(());
        };

        let deferred_binding = prepass_textures.deferred.as_ref().unwrap();
        let depth_binding = prepass_textures.depth.as_ref().unwrap();
        let depth_view = depth_binding.texture.create_view(&TextureViewDescriptor {
            label: Some("prepass_depth"),
            aspect: TextureAspect::DepthOnly,
            ..default()
        });

        let motion_bindings = prepass_textures.motion_vectors.as_ref().unwrap();
        let mip_levels = prepass_downsample.mip_levels as u32;

        {
            let (depth_dst_view, normals_dst_view, motion_dst_view) = {
                (
                    downsample_textures
                        .depth
                        .texture
                        .create_view(&TextureViewDescriptor {
                            label: Some("MIP_DST_DEPTH"),
                            format: Some(DOWNSAMPLE_DEPTH_FORMAT),
                            dimension: Some(TextureViewDimension::D2),
                            aspect: TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: Some(1),
                            base_array_layer: 0,
                            array_layer_count: Some(1),
                        }),
                    downsample_textures
                        .normals
                        .texture
                        .create_view(&TextureViewDescriptor {
                            label: Some("MIP_DST_DEPTH"),
                            format: Some(DOWNSAMPLE_NORMALS_FORMAT),
                            dimension: Some(TextureViewDimension::D2),
                            aspect: TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: Some(1),
                            base_array_layer: 0,
                            array_layer_count: Some(1),
                        }),
                    downsample_textures
                        .motion
                        .texture
                        .create_view(&TextureViewDescriptor {
                            label: Some("MIP_DST_DEPTH"),
                            format: Some(DOWNSAMPLE_MOTION_FORMAT),
                            dimension: Some(TextureViewDimension::D2),
                            aspect: TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: Some(1),
                            base_array_layer: 0,
                            array_layer_count: Some(1),
                        }),
                )
            };
            let bind_group = render_context.render_device().create_bind_group(
                "post_process_bind_group",
                &copy_frame_pipeline.convert_layout,
                // It's important for this to match the BindGroupLayout defined in the PostProcessPipeline
                &BindGroupEntries::with_indices((
                    (0, view_binding(world)),
                    (9, globals_binding(world)),
                    (101, &deferred_binding.default_view),
                    // todo webgl (102, &normal_binding.default_view),
                    (103, &depth_view),
                    (104, &motion_bindings.default_view),
                )),
            );
            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("depth_normals_convert_pass"),
                color_attachments: &[
                    Some(RenderPassColorAttachment {
                        view: &depth_dst_view,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                    Some(RenderPassColorAttachment {
                        view: &normals_dst_view,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                    Some(RenderPassColorAttachment {
                        view: &motion_dst_view,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                ],
                depth_stencil_attachment: None,
            });
            render_pass.set_render_pipeline(convert_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[view_uniform_offset.offset]);
            render_pass.draw(0..3, 0..1);
        }

        #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
        {
            let depth_dst_view = downsample_textures.temp_depth.default_view.clone();
            let normals_dst_view = downsample_textures.temp_normals.default_view.clone();
            let motion_dst_view = downsample_textures.temp_motion.default_view.clone();

            let depth_src_view = downsample_textures.depth.default_view.clone();
            let normals_src_view = downsample_textures.normals.default_view.clone();
            let motion_src_view = downsample_textures.motion.default_view.clone();

            let bind_group = render_context.render_device().create_bind_group(
                "post_process_bind_group",
                &copy_frame_pipeline.downsample_layout,
                // It's important for this to match the BindGroupLayout defined in the PostProcessPipeline
                &BindGroupEntries::sequential((
                    &depth_src_view,
                    &copy_frame_pipeline.sampler1,
                    &normals_src_view,
                    &copy_frame_pipeline.sampler2,
                    &motion_src_view,
                    &copy_frame_pipeline.sampler3,
                )),
            );
            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("depth_normals_mip_chain_pass"),
                color_attachments: &[
                    Some(RenderPassColorAttachment {
                        view: &depth_dst_view,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                    Some(RenderPassColorAttachment {
                        view: &normals_dst_view,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                    Some(RenderPassColorAttachment {
                        view: &motion_dst_view,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                ],
                depth_stencil_attachment: None,
            });
            render_pass.set_render_pipeline(downsample_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        for i in 0..mip_levels - 1 {
            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            let depth_src_view = downsample_textures.temp_depth.default_view.clone();
            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            let normals_src_view = downsample_textures.temp_normals.default_view.clone();
            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            let motion_src_view = downsample_textures.temp_motion.default_view.clone();

            #[cfg(not(all(feature = "webgl", target_arch = "wasm32")))]
            let depth_src_view =
                downsample_textures
                    .depth
                    .texture
                    .create_view(&TextureViewDescriptor {
                        label: Some("MIP_SRC_DEPTH"),
                        format: Some(DOWNSAMPLE_DEPTH_FORMAT),
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::All,
                        base_mip_level: i,
                        mip_level_count: Some(1),
                        base_array_layer: 0,
                        array_layer_count: Some(1),
                    });
            #[cfg(not(all(feature = "webgl", target_arch = "wasm32")))]
            let normals_src_view =
                downsample_textures
                    .normals
                    .texture
                    .create_view(&TextureViewDescriptor {
                        label: Some("MIP_SRC_NORMALS"),
                        format: Some(DOWNSAMPLE_NORMALS_FORMAT),
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::All,
                        base_mip_level: i,
                        mip_level_count: Some(1),
                        base_array_layer: 0,
                        array_layer_count: Some(1),
                    });
            #[cfg(not(all(feature = "webgl", target_arch = "wasm32")))]
            let motion_src_view =
                downsample_textures
                    .motion
                    .texture
                    .create_view(&TextureViewDescriptor {
                        label: Some("MIP_SRC_MOTION"),
                        format: Some(DOWNSAMPLE_MOTION_FORMAT),
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::All,
                        base_mip_level: i,
                        mip_level_count: Some(1),
                        base_array_layer: 0,
                        array_layer_count: Some(1),
                    });

            let depth_dst_view =
                downsample_textures
                    .depth
                    .texture
                    .create_view(&TextureViewDescriptor {
                        label: Some("MIP_DST_DEPTH"),
                        format: Some(DOWNSAMPLE_DEPTH_FORMAT),
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::All,
                        base_mip_level: i + 1,
                        mip_level_count: Some(1),
                        base_array_layer: 0,
                        array_layer_count: Some(1),
                    });
            let normals_dst_view =
                downsample_textures
                    .normals
                    .texture
                    .create_view(&TextureViewDescriptor {
                        label: Some("MIP_DST_DEPTH"),
                        format: Some(DOWNSAMPLE_NORMALS_FORMAT),
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::All,
                        base_mip_level: i + 1,
                        mip_level_count: Some(1),
                        base_array_layer: 0,
                        array_layer_count: Some(1),
                    });
            let motion_dst_view =
                downsample_textures
                    .motion
                    .texture
                    .create_view(&TextureViewDescriptor {
                        label: Some("MIP_DST_MOTION"),
                        format: Some(DOWNSAMPLE_MOTION_FORMAT),
                        dimension: Some(TextureViewDimension::D2),
                        aspect: TextureAspect::All,
                        base_mip_level: i + 1,
                        mip_level_count: Some(1),
                        base_array_layer: 0,
                        array_layer_count: Some(1),
                    });

            let bind_group = render_context.render_device().create_bind_group(
                "post_process_bind_group",
                &copy_frame_pipeline.downsample_layout,
                // It's important for this to match the BindGroupLayout defined in the PostProcessPipeline
                &BindGroupEntries::sequential((
                    &depth_src_view,
                    &copy_frame_pipeline.sampler1,
                    &normals_src_view,
                    &copy_frame_pipeline.sampler2,
                    &motion_src_view,
                    &copy_frame_pipeline.sampler3,
                )),
            );
            let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
                label: Some("depth_normals_mip_chain_pass"),
                color_attachments: &[
                    Some(RenderPassColorAttachment {
                        view: &depth_dst_view,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                    Some(RenderPassColorAttachment {
                        view: &normals_dst_view,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                    Some(RenderPassColorAttachment {
                        view: &motion_dst_view,
                        resolve_target: None,
                        ops: Operations::default(),
                    }),
                ],
                depth_stencil_attachment: None,
            });
            render_pass.set_render_pipeline(downsample_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
        //}

        Ok(())
    }
}

#[derive(Component, Clone)]
pub struct PrepassDownsampleTextures {
    pub normals: CachedTexture,
    pub depth: CachedTexture,
    pub motion: CachedTexture,
    // On webgl2 we can't read from and write to mips of the same texture so we need to create temporary ones
    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
    pub temp_normals: CachedTexture,
    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
    pub temp_depth: CachedTexture,
    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
    pub temp_motion: CachedTexture,
    pub histry_depth: CachedTexture,
}

#[derive(Resource)]
struct PrepassDownsamplePipeline {
    convert_layout: BindGroupLayout,
    downsample_layout: BindGroupLayout,
    sampler1: Sampler,
    sampler2: Sampler,
    sampler3: Sampler,
    convert_pipeline_id: CachedRenderPipelineId,
    downsample_pipeline_id: CachedRenderPipelineId,
}

impl FromWorld for PrepassDownsamplePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let sampler1 = nearest_sampler(render_device);
        let sampler2 = nearest_sampler(render_device);
        let sampler3 = nearest_sampler(render_device);

        let entries = vec![
            view_layout_entry(0),
            globals_layout_entry(9),
            utexture_layout_entry(101, TextureViewDimension::D2),
            // todo webgl ftexture_layout_entry(102, TextureViewDimension::D2),
            dtexture_layout_entry(103, TextureViewDimension::D2),
            ftexture_layout_entry(104, TextureViewDimension::D2),
        ];

        let convert_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("copy_frame_bind_group_layout"),
                    entries: &entries,
                });

        #[allow(unused_mut)]
        let mut shader_defs = Vec::new();

        #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
        shader_defs.push("WEBGL2".into());

        let convert_pipeline_id =
            world
                .resource_mut::<PipelineCache>()
                .queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some("prepass_downsample_convert_pipeline".into()),
                    layout: vec![convert_layout.clone()],
                    vertex: fullscreen_shader_vertex_state(),
                    fragment: Some(FragmentState {
                        shader: CONVERT_SHADER_HANDLE,
                        shader_defs,
                        entry_point: "fragment".into(),
                        targets: vec![
                            Some(ColorTargetState {
                                format: DOWNSAMPLE_DEPTH_FORMAT,
                                blend: None,
                                write_mask: ColorWrites::ALL,
                            }),
                            Some(ColorTargetState {
                                format: DOWNSAMPLE_NORMALS_FORMAT,
                                blend: None,
                                write_mask: ColorWrites::ALL,
                            }),
                            Some(ColorTargetState {
                                format: DOWNSAMPLE_MOTION_FORMAT,
                                blend: None,
                                write_mask: ColorWrites::ALL,
                            }),
                        ],
                    }),
                    primitive: PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                    push_constant_ranges: vec![],
                });

        let entries = vec![
            ftexture_layout_entry(0, TextureViewDimension::D2),
            fsampler_layout_entry(1),
            ftexture_layout_entry(2, TextureViewDimension::D2),
            fsampler_layout_entry(3),
            ftexture_layout_entry(4, TextureViewDimension::D2),
            fsampler_layout_entry(5),
        ];

        let downsample_layout =
            world
                .resource::<RenderDevice>()
                .create_bind_group_layout(&BindGroupLayoutDescriptor {
                    label: Some("copy_frame_bind_group_layout"),
                    entries: &entries,
                });

        let downsample_pipeline_id =
            world
                .resource_mut::<PipelineCache>()
                .queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some("prepass_downsample_pipeline".into()),
                    layout: vec![downsample_layout.clone()],
                    vertex: fullscreen_shader_vertex_state(),
                    fragment: Some(FragmentState {
                        shader: DOWNSAMPLE_SHADER_HANDLE,
                        shader_defs: vec![],
                        entry_point: "fragment".into(),
                        targets: vec![
                            Some(ColorTargetState {
                                format: DOWNSAMPLE_DEPTH_FORMAT,
                                blend: None,
                                write_mask: ColorWrites::ALL,
                            }),
                            Some(ColorTargetState {
                                format: DOWNSAMPLE_NORMALS_FORMAT,
                                blend: None,
                                write_mask: ColorWrites::ALL,
                            }),
                            Some(ColorTargetState {
                                format: DOWNSAMPLE_MOTION_FORMAT,
                                blend: None,
                                write_mask: ColorWrites::ALL,
                            }),
                        ],
                    }),
                    primitive: PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                    push_constant_ranges: vec![],
                });

        Self {
            convert_layout,
            downsample_layout,
            sampler1,
            sampler2,
            sampler3,
            convert_pipeline_id,
            downsample_pipeline_id,
        }
    }
}

fn prepare_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, &PrepassDownsample)>,
    frame_count: Res<FrameCount>,
) {
    for (entity, camera, prepass_downsample) in &views {
        if let Some(physical_viewport_size) = camera.physical_viewport_size {
            let mut depth_texture_descriptor = TextureDescriptor {
                label: None,
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: physical_viewport_size.x,
                    height: physical_viewport_size.y,
                },
                mip_level_count: prepass_downsample.mip_levels as u32,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: DOWNSAMPLE_DEPTH_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            };
            let mut normals_texture_descriptor = TextureDescriptor {
                label: None,
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: physical_viewport_size.x,
                    height: physical_viewport_size.y,
                },
                mip_level_count: prepass_downsample.mip_levels as u32,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: DOWNSAMPLE_NORMALS_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            };
            let mut motion_texture_descriptor = TextureDescriptor {
                label: None,
                size: Extent3d {
                    depth_or_array_layers: 1,
                    width: physical_viewport_size.x,
                    height: physical_viewport_size.y,
                },
                mip_level_count: prepass_downsample.mip_levels as u32,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: DOWNSAMPLE_MOTION_FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC
                    | TextureUsages::COPY_DST,
                view_formats: &[],
            };

            normals_texture_descriptor.label = Some("PrepassDownsampleNormalsTexture");
            depth_texture_descriptor.label = Some("PrepassDownsampleDepthTextureA");
            motion_texture_descriptor.label = Some("PrepassDownsampleMotionTexture");
            let normals_texture =
                texture_cache.get(&render_device, normals_texture_descriptor.clone());
            let depth_texture_a =
                texture_cache.get(&render_device, depth_texture_descriptor.clone());
            depth_texture_descriptor.label = Some("PrepassDownsampleDepthTextureB");
            let depth_texture_b =
                texture_cache.get(&render_device, depth_texture_descriptor.clone());
            let motion_texture =
                texture_cache.get(&render_device, motion_texture_descriptor.clone());
            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            {
                normals_texture_descriptor.mip_level_count = 1;
                depth_texture_descriptor.mip_level_count = 1;
                motion_texture_descriptor.mip_level_count = 1;
                normals_texture_descriptor.label = Some("TempPrepassDownsampleNormalsTexture");
                depth_texture_descriptor.label = Some("TempPrepassDownsampleDepthTexture");
                motion_texture_descriptor.label = Some("TempPrepassDownsampleMotionTexture");
            }
            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            let temp_normals_texture =
                texture_cache.get(&render_device, normals_texture_descriptor.clone());
            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            let temp_depth_texture =
                texture_cache.get(&render_device, depth_texture_descriptor.clone());
            #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
            let temp_motion_texture =
                texture_cache.get(&render_device, motion_texture_descriptor.clone());

            let textures = if frame_count.0 % 2 == 0 {
                PrepassDownsampleTextures {
                    normals: normals_texture,
                    depth: depth_texture_a,
                    histry_depth: depth_texture_b,
                    motion: motion_texture,
                    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
                    temp_normals: temp_normals_texture,
                    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
                    temp_depth: temp_depth_texture,
                    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
                    temp_motion: temp_motion_texture,
                }
            } else {
                PrepassDownsampleTextures {
                    normals: normals_texture,
                    depth: depth_texture_b,
                    histry_depth: depth_texture_a,
                    motion: motion_texture,
                    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
                    temp_normals: temp_normals_texture,
                    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
                    temp_depth: temp_depth_texture,
                    #[cfg(all(feature = "webgl", target_arch = "wasm32"))]
                    temp_motion: temp_motion_texture,
                }
            };

            commands.entity(entity).insert(textures);
        }
    }
}
