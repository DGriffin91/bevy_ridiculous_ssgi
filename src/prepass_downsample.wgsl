@group(0) @binding(0) var depth_prepass_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_sampler: sampler;
@group(0) @binding(2) var normal_prepass_texture: texture_2d<f32>;
@group(0) @binding(3) var normal_sampler: sampler;
@group(0) @binding(4) var motion_prepass_texture: texture_2d<f32>;
@group(0) @binding(5) var motion_sampler: sampler;

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct FragmentOutput {
    @location(0) depth: f32,
    @location(1) normals: vec4<f32>,
    @location(2) motion: vec2<f32>,
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    out.depth = textureSample(depth_prepass_texture, depth_sampler, in.uv).x;
    out.normals = textureSample(normal_prepass_texture, normal_sampler, in.uv);
    out.motion = textureSample(motion_prepass_texture, normal_sampler, in.uv).xy;
    return out;
}
