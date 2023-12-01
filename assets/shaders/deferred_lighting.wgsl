#import bevy_pbr::{
    prepass_utils,
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
    pbr_functions,
    pbr_deferred_functions::pbr_input_from_deferred_gbuffer,
    pbr_deferred_types::unpack_unorm3x4_plus_unorm_20_,
    mesh_view_bindings::{deferred_prepass_texture, view, globals},
}

// ---------------------------------------
// ---------------------------------------
// ---------------------------------------
// ---------------------------------------
// ---------------------------------------

#import "shaders/sampling.wgsl" as sampling
#import "shaders/sampling.wgsl"::TAU
#import bevy_pbr::view_transformations as vt
#import "shaders/rgb9e5.wgsl"::{vec3_to_rgb9e5_, rgb9e5_to_vec3_}
#import "shaders/xyz8e5.wgsl"::{vec3_to_xyz8e5_, xyz8e5_to_vec3_}
#import "shaders/util.wgsl" as util
#import bevy_pbr::utils::{octahedral_encode, octahedral_decode}

#import bevy_pbr::{
    pbr_types::PbrInput,
    utils::PI,
}

#import bevy_pbr::mesh_view_bindings as view_bindings
#import bevy_pbr::prepass_utils::prepass_motion_vector

@group(1) @binding(0) var<uniform> depth_id: PbrDeferredLightingDepthId;
@group(1) @binding(1) var prev_frame_tex: texture_2d<f32>;
@group(1) @binding(5) var nearest_sampler: sampler;
@group(1) @binding(6) var linear_sampler: sampler;
@group(1) @binding(12) var ssgi_resolve: texture_2d<f32>;

// ---------------------------------------
// ---------------------------------------
// ---------------------------------------
// ---------------------------------------
// ---------------------------------------

#ifdef SCREEN_SPACE_AMBIENT_OCCLUSION
#import bevy_pbr::mesh_view_bindings::screen_space_ambient_occlusion_texture
#import bevy_pbr::gtao_utils::gtao_multibounce
#endif

struct FullscreenVertexOutput {
    @builtin(position)
    position: vec4<f32>,
    @location(0)
    uv: vec2<f32>,
};

struct PbrDeferredLightingDepthId {
    depth_id: u32, // limited to u8
#ifdef SIXTEEN_BYTE_ALIGNMENT
    // WebGL2 structs must be 16 byte aligned.
    _webgl2_padding_0: f32,
    _webgl2_padding_1: f32,
    _webgl2_padding_2: f32,
#endif
}

// https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve
fn rcp(x: f32) -> f32 { return 1.0 / x; }
fn max3(x: vec3<f32>) -> f32 { return max(x.r, max(x.g, x.b)); }
fn tonemap(color: vec3<f32>) -> vec3<f32> { return color * rcp(max3(color) + 1.0); }
fn reverse_tonemap(color: vec3<f32>) -> vec3<f32> { return color * rcp(1.0 - max3(color)); }

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    // See the full screen vertex shader for explanation above for how this works.
    let uv = vec2<f32>(f32(vertex_index >> 1u), f32(vertex_index & 1u)) * 2.0;
    // Depth is stored as unorm, so we are dividing the u8 depth_id by 255.0 here.
    let clip_position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), f32(depth_id.depth_id) / 255.0, 1.0);

    return FullscreenVertexOutput(clip_position, uv);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    var frag_coord = vec4(in.position.xy, 0.0, 0.0);

    let deferred_data = textureLoad(deferred_prepass_texture, vec2<i32>(frag_coord.xy), 0);

#ifdef WEBGL2
    frag_coord.z = unpack_unorm3x4_plus_unorm_20_(deferred_data.b).w;
#else
#ifdef DEPTH_PREPASS
    frag_coord.z = prepass_utils::prepass_depth(in.position, 0u);
#endif
#endif

    var pbr_input = pbr_input_from_deferred_gbuffer(frag_coord, deferred_data);
    var output_color = vec4(0.0);

    // NOTE: Unlit bit not set means == 0 is true, so the true case is if lit
    if ((pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u) {

#ifdef SCREEN_SPACE_AMBIENT_OCCLUSION
        let ssao = textureLoad(screen_space_ambient_occlusion_texture, vec2<i32>(in.position.xy), 0i).r;
        let ssao_multibounce = gtao_multibounce(ssao, pbr_input.material.base_color.rgb);
        pbr_input.occlusion = min(pbr_input.occlusion, ssao_multibounce);
#endif // SCREEN_SPACE_AMBIENT_OCCLUSION

        output_color = pbr_functions::apply_pbr_lighting(pbr_input);
        

// ----------------------------------------------------
// ----------------------------------------------------
// ----------------------------------------------------
    let diffuse_color = pbr_input.material.base_color;// * (1.0 - pbr_input.material.metallic);
    //let indirect_light = read_cascade_radiance(pbr_input, pbr_input.N, pbr_input.frag_coord, pbr_input.world_position.xyz);
    let indirect_light = textureLoad(ssgi_resolve, vec2<i32>(frag_coord.xy), 0).rgb;
    
    output_color += vec4(diffuse_color.rgb * indirect_light, 0.0);
// ----------------------------------------------------
// ----------------------------------------------------
// ----------------------------------------------------

    } else {
        output_color = pbr_input.material.base_color;
    }

    
    output_color = pbr_functions::main_pass_post_lighting_processing(pbr_input, output_color);

    return output_color;
}
