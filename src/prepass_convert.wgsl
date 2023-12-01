#import bevy_pbr::pbr_deferred_functions::pbr_input_from_deferred_gbuffer
#import bevy_pbr::pbr_deferred_types::unpack_unorm3x4_plus_unorm_20_
#import bevy_pbr::utils::{octahedral_encode, octahedral_decode}

@group(0) @binding(101) var deferred_prepass_texture: texture_2d<u32>;
//@group(0) @binding(102) var normal_prepass_texture: texture_2d<f32>;
@group(0) @binding(103) var depth_prepass_texture: texture_depth_2d;
@group(0) @binding(104) var motion_prepass_texture: texture_2d<f32>;

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

struct FragmentOutput {
    @location(0) depth: f32,
    @location(1) normals: vec2<f32>,
    @location(2) motion: vec2<f32>,
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    var frag_coord = vec4(in.position.xy, 0.0, 0.0);

    let deferred_data = textureLoad(deferred_prepass_texture, vec2<i32>(frag_coord.xy), 0);

#ifdef WEBGL2
    frag_coord.z = unpack_unorm3x4_plus_unorm_20_(deferred_data.b).w;
#else
    frag_coord.z = textureLoad(depth_prepass_texture, vec2<i32>(in.position.xy), 0);
#endif

    var pbr_input = pbr_input_from_deferred_gbuffer(frag_coord, deferred_data);

    out.depth = frag_coord.z;
//#ifdef WEBGL2
    out.normals = octahedral_encode(pbr_input.N);
//#else
//    out.normals = octahedral_encode(textureLoad(normal_prepass_texture, vec2<i32>(in.position.xy), 0).xyz * 2.0 - 1.0);
//#endif
    out.motion = textureLoad(motion_prepass_texture, vec2<i32>(in.position.xy), 0).xy;
    return out;
}
