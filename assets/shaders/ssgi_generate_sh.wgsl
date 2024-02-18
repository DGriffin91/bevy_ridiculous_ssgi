#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_pbr::view_transformations as vt
#import ssgi::common as common
#import ssgi::rgb9e5::{vec3_to_rgb9e5_, rgb9e5_to_vec3_}
#import ssgi::xyz8e5::{vec3_to_xyz8e5_, xyz8e5_to_vec3_}
#import bevy_pbr::mesh_view_bindings::{globals, view}
#import ssgi::sampling as sampling
#import ssgi::sampling::TAU
#import bevy_pbr::lighting::{specular, F_AB, Fd_Burley, perceptualRoughnessToRoughness}
#import bevy_pbr::utils::{octahedral_encode, octahedral_decode}

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

@group(0) @binding(101) var cascade_0_data1: texture_2d<u32>;
@group(0) @binding(111) var cascade_0_data2: texture_2d<u32>;
@group(0) @binding(102) var prepass_downsample_normals: texture_2d<f32>;
@group(0) @binding(103) var prepass_downsample_depth: texture_2d<f32>;
@group(0) @binding(104) var prepass_downsample_motion: texture_2d<f32>;
@group(0) @binding(105) var prev_sh_texture: texture_2d<u32>;
@group(0) @binding(106) var prev_pos_texture: texture_2d<f32>;
@group(0) @binding(109) var<uniform> config: SSGIGenerateSHConfig;

struct FragmentOutput {
    @location(0) sh: vec4<u32>,
    @location(1) pos: vec4<f32>,
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    let icas_coord = vec2<i32>(in.position.xy);
    let icas_0_size = vec2<i32>(textureDimensions(cascade_0_data1).xy);
    let sh_res = vec2<f32>(textureDimensions(prev_sh_texture).xy);
    let view_z_dir = vt::direction_view_to_world(vec3(0.0, 0.0, -1.0));
    
    var frag_coord = vec4<f32>(common::frag_coord_for_cas(0u, icas_coord, config.render_scale), 0.0, 0.0);
    var ifrag_coord = vec2<i32>(frag_coord.xy);

    var frag_coord_no_jitter = vec4(in.uv * view.viewport.zw, 0.0, 0.0);
    var ifrag_coord_no_jitter = vec2<i32>(frag_coord_no_jitter.xy);

    frag_coord.z = max(textureLoad(prepass_downsample_depth, vec2<i32>(frag_coord.xy), 0).x, sampling::F32_EPSILON);
    frag_coord_no_jitter.z = max(textureLoad(prepass_downsample_depth, vec2<i32>(frag_coord_no_jitter.xy), 0).x, sampling::F32_EPSILON);

    let uv = frag_coord.xy / view.viewport.zw;
    let uv_no_jitter = in.uv;

    let world_position = vt::position_ndc_to_world(vec3(vt::uv_to_ndc(uv), frag_coord.z));
    let world_position_no_jitter = vt::position_ndc_to_world(vec3(vt::uv_to_ndc(uv_no_jitter), frag_coord_no_jitter.z));
    let V = normalize(view.world_position.xyz - world_position);

    // For spec
    let N = octahedral_decode(textureLoad(prepass_downsample_normals, vec2<i32>(frag_coord.xy), 0).xy);
    // //var spec_color = vec3(0.0);
    // let perceptual_roughness = 0.5;
    // //let perceptual_roughness = pbr.material.perceptual_roughness;
    // let roughness = perceptualRoughnessToRoughness(perceptual_roughness);
    // let NdotV = max(dot(N, V), 0.0001);
    // let f_ab = F_AB(perceptual_roughness, NdotV);
    // //let F0 = 0.16 * pbr.material.reflectance * pbr.material.reflectance * (1.0 - pbr.material.metallic) + pbr.material.base_color.rgb * pbr.material.metallic;
    // let F0 = 0.16 * 0.5 * 0.5 * (1.0 - 0.0) + vec3(0.5) * 0.0;
    // let R = reflect(-V, N);
    // ---


    let directions_div_4 = vec2(4, i32(config.directions) / 4);
    let fdirections = f32(config.directions);

    var sh0 = vec3(0.0);
    var sh1 = vec3(0.0);
    var sh2 = vec3(0.0);
    var sh3 = vec3(0.0);

    var spec = vec3(0.0);

    let phase_offset = common::get_phase_noise_offset(fdirections);

    for (var i = 0u; i < config.directions; i += 1u) {
        let phase = fract(f32(i) / fdirections);
        let phase_with_offset = fract(phase + phase_offset);
        var phi = TAU * phase_with_offset;
        let ss_dir = vec2(cos(phi), sin(phi));

        var ws_dir = common::ss_dir_to_ws_dir(world_position, uv, ss_dir, frag_coord.z);

        let i = i32(phase * fdirections);

        let ofs = vec2(i % 4, i / 4);

        let coords0 = (icas_coord + vec2(0, 0)) * directions_div_4 + ofs;

        var c1 = vec4(0u);
        var c2 = vec4(0u);
        c1 = textureLoad(cascade_0_data1, clamp(coords0, vec2(0), icas_0_size - 1), 0);
        c2 = textureLoad(cascade_0_data2, clamp(coords0, vec2(0), icas_0_size - 1), 0);

        let dir1 = common::reconstruct_dir_to_sample(V, ws_dir, 0.111);
        let dir2 = common::reconstruct_dir_to_sample(V, ws_dir, 0.222);
        let dir3 = common::reconstruct_dir_to_sample(V, ws_dir, 0.333);
        let dir4 = common::reconstruct_dir_to_sample(V, ws_dir, 0.444);
        let dir5 = common::reconstruct_dir_to_sample(V, ws_dir, 0.555);
        let dir6 = common::reconstruct_dir_to_sample(V, ws_dir, 0.666);
        let dir7 = common::reconstruct_dir_to_sample(V, ws_dir, 0.777);
        let dir8 = common::reconstruct_dir_to_sample(V, ws_dir, 0.888);

        // The dot(N, dir) here helps with light leaks
        var gather1 = rgb9e5_to_vec3_(c1.x) * saturate(dot(N, dir1) * 8.0); 
        var gather2 = rgb9e5_to_vec3_(c1.y) * saturate(dot(N, dir2) * 8.0); 
        var gather3 = rgb9e5_to_vec3_(c1.z) * saturate(dot(N, dir3) * 8.0); 
        var gather4 = rgb9e5_to_vec3_(c1.w) * saturate(dot(N, dir4) * 8.0); 
        var gather5 = rgb9e5_to_vec3_(c2.x) * saturate(dot(N, dir5) * 8.0); 
        var gather6 = rgb9e5_to_vec3_(c2.y) * saturate(dot(N, dir6) * 8.0); 
        var gather7 = rgb9e5_to_vec3_(c2.z) * saturate(dot(N, dir7) * 8.0); 
        var gather8 = rgb9e5_to_vec3_(c2.w) * saturate(dot(N, dir8) * 8.0); 

        sh0 += gather1;
        sh0 += gather2;
        sh0 += gather3;
        sh0 += gather4;
        sh0 += gather5;
        sh0 += gather6;
        sh0 += gather7;
        sh0 += gather8;

        sh1 += gather1 * dir1.x;
        sh1 += gather2 * dir2.x;
        sh1 += gather3 * dir3.x;
        sh1 += gather4 * dir4.x;
        sh1 += gather5 * dir5.x;
        sh1 += gather6 * dir6.x;
        sh1 += gather7 * dir7.x;
        sh1 += gather8 * dir8.x;

        sh2 += gather1 * dir1.y;
        sh2 += gather2 * dir2.y;
        sh2 += gather3 * dir3.y;
        sh2 += gather4 * dir4.y;
        sh2 += gather5 * dir5.y;
        sh2 += gather6 * dir6.y;
        sh2 += gather7 * dir7.y;
        sh2 += gather8 * dir8.y;

        sh3 += gather1 * dir1.z;
        sh3 += gather2 * dir2.z;
        sh3 += gather3 * dir3.z;
        sh3 += gather4 * dir4.z;
        sh3 += gather5 * dir5.y;
        sh3 += gather6 * dir6.y;
        sh3 += gather7 * dir7.y;
        sh3 += gather8 * dir8.y;
        
        // For spec
        //spec += sampling::bevy_light(roughness, NdotV, N, V, R, F0, f_ab, gather1, dir1, 1.0);
        //spec += sampling::bevy_light(roughness, NdotV, N, V, R, F0, f_ab, gather2, dir2, 1.0);
        //spec += sampling::bevy_light(roughness, NdotV, N, V, R, F0, f_ab, gather3, dir3, 1.0);
        //spec += sampling::bevy_light(roughness, NdotV, N, V, R, F0, f_ab, gather4, dir4, 1.0);
        //spec += sampling::bevy_light(roughness, NdotV, N, V, R, F0, f_ab, gather5, dir5, 1.0);
        //spec += sampling::bevy_light(roughness, NdotV, N, V, R, F0, f_ab, gather6, dir6, 1.0);
        //spec += sampling::bevy_light(roughness, NdotV, N, V, R, F0, f_ab, gather7, dir7, 1.0);
        //spec += sampling::bevy_light(roughness, NdotV, N, V, R, F0, f_ab, gather8, dir8, 1.0);
    }


    sh0 /= fdirections;
    sh1 /= fdirections;
    sh2 /= fdirections;
    sh3 /= fdirections;

    // P. 47 https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/gdc2018-precomputedgiobalilluminationinfrostbite.pdf
    let Y0 = 0.282095; // sqrt(1/fourPI)
    let Y1 = 0.488603; // sqrt(3/fourPI)

    let A0 = 0.886227; // pi/sqrt(fourPI)
    let A1 = 1.023326; // sqrt(pi/3
    
    let AY0 = 0.25;
    let AY1 = 0.5;

    sh0 *= AY0;
    sh1 *= AY1;
    sh2 *= AY1;
    sh3 *= AY1;

    // Decode with:
    //out += sh0 
    //     + sh1 * N.x
    //     + sh2 * N.y
    //     + sh3 * N.z;

    
    let closest_motion_vector = textureLoad(prepass_downsample_motion, ifrag_coord, 0).xy;
    let history_uv = uv - closest_motion_vector;
    let closest_motion_vector_no_jitter = textureLoad(prepass_downsample_motion, ifrag_coord_no_jitter, 0).xy;
    let history_uv_no_jitter = uv_no_jitter - closest_motion_vector_no_jitter;

    var prev_pos_spec = textureLoad(prev_pos_texture, icas_coord, 0);
    var prev_pos = prev_pos_spec.xyz;
    var prev_spec = rgb9e5_to_vec3_(bitcast<u32>(prev_pos_spec.w));
    var closest_ws_pos_dist = distance(prev_pos, world_position_no_jitter) * 0.9;
    var closest_prev_pos = prev_pos;

    var offset = vec2(0);
    var closest_offset = vec2(0);
//    for (var x = -2; x <= 2; x += 1) {
//        for (var y = -2; y <= 2; y += 1) {
//            let offset = vec2(x, y);
//            prev_pos = textureLoad(prev_pos_texture, vec2<i32>(history_uv_no_jitter * sh_res) + offset, 0).xyz;
//
//            let dist = distance(prev_pos, world_position_no_jitter);
//            if dist < closest_ws_pos_dist {
//                closest_ws_pos_dist = dist;
//                closest_offset = offset;
//                closest_prev_pos = prev_pos;
//            }
//        } 
//    }

    let prev_sh = textureLoad(prev_sh_texture, vec2<i32>(history_uv_no_jitter * sh_res) + closest_offset, 0);
    
    let prev_sh0 = rgb9e5_to_vec3_(prev_sh.x);
    let prev_sh1 = xyz8e5_to_vec3_(prev_sh.y);
    let prev_sh2 = xyz8e5_to_vec3_(prev_sh.z);
    let prev_sh3 = xyz8e5_to_vec3_(prev_sh.w);
    
    let reprojection_fail = f32(any(history_uv <= vec2(0.0)) || any(history_uv >= vec2(1.0)));

    let hysteresis = mix(config.hysteresis, saturate(config.hysteresis + 0.4), reprojection_fail);

    sh0 = mix(prev_sh0, sh0, hysteresis);
    sh1 = mix(prev_sh1, sh1, hysteresis);
    sh2 = mix(prev_sh2, sh2, hysteresis);
    sh3 = mix(prev_sh3, sh3, hysteresis);


    spec = mix(prev_spec, spec, hysteresis);

    closest_prev_pos = mix(closest_prev_pos, world_position_no_jitter, hysteresis);
    out.pos = vec4(closest_prev_pos, bitcast<f32>(vec3_to_rgb9e5_(spec))); 

    out.sh = vec4(vec3_to_rgb9e5_(sh0), vec3_to_xyz8e5_(sh1), vec3_to_xyz8e5_(sh2), vec3_to_xyz8e5_(sh3));
    //out.sh = vec4(vec3_to_rgb9e5_(vec3(1.0)), 0u, 0u, 0u);

    return out;
}