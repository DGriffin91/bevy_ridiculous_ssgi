#import bevy_pbr::utils::{octahedral_encode, octahedral_decode}

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput
#import bevy_pbr::view_transformations as vt
#import ssgi::common as common
#import ssgi::rgb9e5::{vec3_to_rgb9e5_, rgb9e5_to_vec3_}
#import ssgi::xyz8e5::{vec3_to_xyz8e5_, xyz8e5_to_vec3_}
#import bevy_pbr::mesh_view_bindings::{globals, view}
#import ssgi::sampling as sampling
#import ssgi::sampling::TAU
#import bevy_pbr::lighting::{specular, F_AB, Fd_Burley, perceptualRoughnessToRoughness}

struct DisocclusionUniform {
    inverse_view_proj: mat4x4<f32>, // not jittered
    prev_inverse_view_proj: mat4x4<f32>, // not jittered
    velocity_disocclusion: f32,
    depth_disocclusion_px_radius: f32,
    normals_disocclusion_scale: f32,
};

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
};

@group(0) @binding(101) var cascade_0_data: texture_2d<u32>;
@group(0) @binding(102) var prepass_downsample_normals: texture_2d<f32>;
@group(0) @binding(103) var prepass_downsample_depth: texture_2d<f32>;
@group(0) @binding(104) var prepass_downsample_motion: texture_2d<f32>;
@group(0) @binding(105) var cascade_0_sh_data: texture_2d<u32>;
@group(0) @binding(106) var prev_resolve: texture_2d<f32>;
//@group(0) @binding(107) var disocclusion_texture: texture_2d<f32>;
@group(0) @binding(108) var linear_sampler: sampler;
@group(0) @binding(109) var<uniform> config: SSGIResolveConfig;
@group(0) @binding(110) var pos_refl: texture_2d<f32>;
//@group(0) @binding(111) var<uniform> duni: DisocclusionUniform;

/// Convert a ndc space position to world space
// todo webgl
//fn position_history_ndc_to_world(ndc_pos: vec3<f32>) -> vec3<f32> {
//    let world_pos = duni.prev_inverse_view_proj * vec4(ndc_pos, 1.0);
//    return world_pos.xyz / world_pos.w;
//}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {    
    var frag_coord = vec4(in.position.xy, 0.0, 0.0);
    var ifrag_coord = vec2<i32>(frag_coord.xy);

    frag_coord.z = max(textureLoad(prepass_downsample_depth, ifrag_coord, 0).x, sampling::F32_EPSILON);
    let N = octahedral_decode(textureLoad(prepass_downsample_normals, ifrag_coord, 0).xy);

    let world_position = vt::position_ndc_to_world(vec3(vt::uv_to_ndc(in.uv), frag_coord.z));

    var out = vec4(0.0);
    //let d = textureLoad(disocclusion_texture, vec2<i32>(frag_coord.xy), 0);
    //let two_of_three = min(min(max(d.x, d.y), max(d.y, d.z)), max(d.x, d.z));
    //let all_three = max(max(d.x, d.y), d.z);
    let closest_motion_vector = textureLoad(prepass_downsample_motion, ifrag_coord, 0).xy;
    let history_frag_coord = vec2<i32>((in.uv - closest_motion_vector) * view.viewport.zw);
    let history_uv = in.uv - closest_motion_vector;
    let reprojection_fail = any(history_uv <= 0.0) || any(history_uv >= 1.0); //TODO webgl max( , saturate(two_of_three * 3.0))

    out = vec4(read_cascade_radiance(world_position, N, frag_coord, select(history_uv, in.uv.xy, reprojection_fail)), 1.0);
    
    let frender_scale = f32(config.render_scale);
    let cas_coord = vec2(
        (frag_coord.x - frender_scale * 0.5) / frender_scale,
        (frag_coord.y - frender_scale * 0.5) / frender_scale,
    );




    var offset = vec2(0);
    var closest_offset = vec2(0);
/*
    let history_normals = textureLoad(data_history_normals, history_frag_coord, 0).xyz;
    var closest_normal_dot = dot(history_normals, N) + 0.01;

    let history_depth = max(textureLoad(history_depth_texture, history_frag_coord, 0).x, sampling::F32_EPSILON);
    let history_world_position = position_history_ndc_to_world(vec3(vt::uv_to_ndc(history_uv), history_depth));
    let ws_dist = distance(history_world_position, world_position);

    var closest_ws_pos_dist = ws_dist;
    var closest_color_dist = 0.001;
    var prev_frame = vec4(0.0);
    var total_weight = 0.0;
    for (var x = -2; x <= 2; x += 1) {
        for (var y = -2; y <= 2; y += 1) {
            offset = vec2(x, y);
            let frag_offset = vec2<f32>(offset) / view.viewport.zw;
            let history_normals = textureLoad(data_history_normals, history_frag_coord + offset, 0).xyz;
            let history_depth = max(textureLoad(history_depth_texture, history_frag_coord + offset, 0).x, sampling::F32_EPSILON);
            let history_world_position = position_history_ndc_to_world(vec3(vt::uv_to_ndc(history_uv + frag_offset), history_depth));
            let ws_dist = distance(history_world_position, world_position);
            let ndot = dot(history_normals, N);
            //let prev_frame = textureSampleLevel(prev_resolve, linear_sampler, history_uv + frag_offset, 0.0);
            if ndot > closest_normal_dot {
                closest_offset = offset;
                closest_normal_dot = ndot;
            }
            //if ws_dist < closest_ws_pos_dist {
            //    closest_ws_pos_dist = ws_dist;
            //    closest_offset = offset;
            //}
            //let weight = 1.0/ws_dist;
            //let color = textureLoad(prev_resolve, history_frag_coord + offset, 0);
            //prev_frame += color * weight;
            //total_weight += weight;
            //let rec709 = vec3<f32>(0.2126, 0.7152, 0.0722);
            //let color_dist = distance(dot(prev_frame.rgb, rec709), dot(out.rgb, rec709));
            //if color_dist < closest_color_dist {
            //    closest_color_dist = color_dist;
            //    closest_offset = offset;
            //}
        }
    }
*/
    //prev_frame /= max(total_weight, 1.0);


    //let prev_frame = textureSampleLevel(prev_resolve, linear_sampler, history_uv + vec2<f32>(closest_offset) / view.viewport.zw, 0.0);
    let prev_frame = texture_sample_bicubic_catmull_rom(prev_resolve, linear_sampler, history_uv, view.viewport.zw);
    let hysteresis = mix(config.hysteresis, saturate(config.hysteresis + 0.4), f32(reprojection_fail));
    let blend = mix(clamp(prev_frame.rgb, vec3(0.0), vec3(10000.0)), out.rgb, hysteresis);
    out = vec4(blend, out.a);

    return out;
}

fn read_cascade_radiance(world_position: vec3<f32>, N: vec3<f32>, frag_coord: vec4<f32>, history_uv: vec2<f32>) -> vec3<f32> {
    var ufrag_coord = vec2<u32>(frag_coord.xy);
    
    var pixel_radius = sampling::world_space_pixel_radius(-vt::depth_ndc_to_view_z(frag_coord.z));
    // limit minimum pixel radius for things really close to the camera
    pixel_radius = max(pixel_radius, 0.001); 

    let V = normalize(view.world_position.xyz - world_position);
    let R = reflect(-V, N);
    let NdotV = max(dot(N, V), 0.0001);

    // TODO Non-PBR
    var fresnel = pow(clamp(1.0 - NdotV, 0.0, 1.0) * config.rough_specular, config.rough_specular_sharpness);

    let frender_scale = f32(config.render_scale);

    var out = vec3(0.0);

    let cas_coord = vec2(
        (frag_coord.x - frender_scale * 0.5) / frender_scale,
        (frag_coord.y - frender_scale * 0.5) / frender_scale,
    );
    let icas_coord = vec2<i32>(cas_coord);
    let fd = abs(cas_coord - ceil(cas_coord));
    let id = 1.0 - fd;

    var aa = fd.x * fd.y;
    var ba = id.x * fd.y;
    var ab = fd.x * id.y;
    var bb = id.x * id.y;

    common::weight_bilinear(
        &aa, &ba, &ab, &bb, 
        N, world_position, frag_coord.z, 
        config.normal_rejection, config.distance_rejection, 
        icas_coord, 0u, config.render_scale, 
        prepass_downsample_normals, prepass_downsample_depth,
        pixel_radius,
    );

    var sh0 = vec3(0.0);
    var sh1 = vec3(0.0);
    var sh2 = vec3(0.0);
    var sh3 = vec3(0.0);
    var c1 = vec4(0u);

    // TODO specular should be seperate so it's not multiplied by the albedo
    c1 = textureLoad(cascade_0_sh_data, icas_coord + vec2(0, 0), 0);
    sh0 = rgb9e5_to_vec3_(c1.x);
    sh1 = xyz8e5_to_vec3_(c1.y);
    sh2 = xyz8e5_to_vec3_(c1.z);
    sh3 = xyz8e5_to_vec3_(c1.w);
    out += (sh0 + sh1 * R.x + sh2 * R.y + sh3 * R.z) * aa * fresnel;
    out += (sh0 + sh1 * N.x + sh2 * N.y + sh3 * N.z) * aa;
    c1 = textureLoad(cascade_0_sh_data, icas_coord + vec2(1, 0), 0);
    sh0 = rgb9e5_to_vec3_(c1.x);
    sh1 = xyz8e5_to_vec3_(c1.y);
    sh2 = xyz8e5_to_vec3_(c1.z);
    sh3 = xyz8e5_to_vec3_(c1.w);
    out += (sh0 + sh1 * R.x + sh2 * R.y + sh3 * R.z) * ba * fresnel;
    out += (sh0 + sh1 * N.x + sh2 * N.y + sh3 * N.z) * ba;
    c1 = textureLoad(cascade_0_sh_data, icas_coord + vec2(0, 1), 0);
    sh0 = rgb9e5_to_vec3_(c1.x);
    sh1 = xyz8e5_to_vec3_(c1.y);
    sh2 = xyz8e5_to_vec3_(c1.z);
    sh3 = xyz8e5_to_vec3_(c1.w);
    out += (sh0 + sh1 * R.x + sh2 * R.y + sh3 * R.z) * ab * fresnel;
    out += (sh0 + sh1 * N.x + sh2 * N.y + sh3 * N.z) * ab;
    c1 = textureLoad(cascade_0_sh_data, icas_coord + vec2(1, 1), 0);
    sh0 = rgb9e5_to_vec3_(c1.x);
    sh1 = xyz8e5_to_vec3_(c1.y);
    sh2 = xyz8e5_to_vec3_(c1.z);
    sh3 = xyz8e5_to_vec3_(c1.w);
    out += (sh0 + sh1 * R.x + sh2 * R.y + sh3 * R.z) * bb * fresnel;
    out += (sh0 + sh1 * N.x + sh2 * N.y + sh3 * N.z) * bb;
    //out = xyz8e5_to_vec3_(textureLoad(cascade_0_sh_data, icas_coord + vec2(0, 0), 0).x);

    // For spec - Looks Blocky
    //let white_frame_noise = sampling::white_frame_noise(789u);
    //let samples = 4u;
    //let distance = 10.0;
    //var weight = 0.1;
    //var sum = vec3(0.0);
    //for (var i = 0u; i <= samples; i += 1u) {
    //    var rand = vec2(
    //        fract(sampling::blue_noise_for_pixel(ufrag_coord, globals.frame_count + i) + white_frame_noise.x),
    //        fract(sampling::blue_noise_for_pixel(ufrag_coord, globals.frame_count + i + samples) + white_frame_noise.y),
    //    );
    //    let offset = vec2<i32>((rand * 2.0 - 1.0) * distance);
    //    let samp_color = rgb9e5_to_vec3_(bitcast<u32>(textureLoad(pos_refl, icas_coord + offset, 0).w));    
    //    let gather_frag_coord = vec2<i32>(common::frag_coord_for_cas(0u, (icas_coord + offset), config.render_scale));
    //    let gather_normal = octahedral_decode(textureLoad(prepass_downsample_normals, gather_frag_coord, 0).xy);
    //    let gather_depth = textureLoad(prepass_downsample_depth, gather_frag_coord, 0).x;
    //    var w = 1.0;
    //    w *= max(pow(dot(N, gather_normal), config.normal_rejection), 0.001);
    //    w *= max(1.0 - saturate(distance(gather_depth, frag_coord.z) * config.distance_rejection), 0.0);
    //    w *= distance * 2.0 - length(vec2<f32>(offset));
    //    sum += samp_color * w;
    //    weight += w;
    //}
    //sum /= weight;
    //out += sum;
    
    // For spec - Shows artifacts on the edges
    //var s1 = vec3(0.0);
    //let history_frag = vec2<i32>(history_uv * view.viewport.zw / frender_scale);
    //s1 = rgb9e5_to_vec3_(bitcast<u32>(textureLoad(pos_refl, icas_coord + vec2(0, 0), 0).w));
    //out += s1 * aa * 1.0;
    //s1 = rgb9e5_to_vec3_(bitcast<u32>(textureLoad(pos_refl, icas_coord + vec2(1, 0), 0).w));
    //out += s1 * ba * 1.0;
    //s1 = rgb9e5_to_vec3_(bitcast<u32>(textureLoad(pos_refl, icas_coord + vec2(0, 1), 0).w));
    //out += s1 * ab * 1.0;
    //s1 = rgb9e5_to_vec3_(bitcast<u32>(textureLoad(pos_refl, icas_coord + vec2(1, 1), 0).w));
    //out += s1 * bb * 1.0;

    return clamp(out, vec3(0.0), vec3(10000.0));
}

// 5-sample Catmull-Rom filtering
// Catmull-Rom filtering: https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
// Ignoring corners: https://www.activision.com/cdn/research/Dynamic_Temporal_Antialiasing_and_Upsampling_in_Call_of_Duty_v4.pdf#page=68
// Technically we should renormalize the weights since we're skipping the corners, but it's basically the same result
fn texture_sample_bicubic_catmull_rom(tex: texture_2d<f32>, tex_sampler: sampler, uv: vec2<f32>, texture_size: vec2<f32>) -> vec4<f32> {
    let texel_size = 1.0 / texture_size;
    let sample_position = uv * texture_size;
    let tex_pos1 = floor(sample_position - 0.5) + 0.5;
    let f = sample_position - tex_pos1;

    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    var w12 = w1 + w2;
    var offset12 = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    var tex_pos0 = tex_pos1 - 1.0;
    var tex_pos3 = tex_pos1 + 2.0;
    var tex_pos12 = tex_pos1 + offset12;

    tex_pos0 /= texture_size;
    tex_pos3 /= texture_size;
    tex_pos12 /= texture_size;

    var result = vec4(0.0);
    //result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos0.x, tex_pos0.y), 0.0) * w0.x * w0.y;
    result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos12.x, tex_pos0.y), 0.0) * w12.x * w0.y;
    //result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos3.x, tex_pos0.y), 0.0) * w3.x * w0.y;

    result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos0.x, tex_pos12.y), 0.0) * w0.x * w12.y;
    result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos12.x, tex_pos12.y), 0.0) * w12.x * w12.y;
    result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos3.x, tex_pos12.y), 0.0) * w3.x * w12.y;

    //result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos0.x, tex_pos3.y), 0.0) * w0.x * w3.y;
    result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos12.x, tex_pos3.y), 0.0) * w12.x * w3.y;
    //result += textureSampleLevel(tex, tex_sampler, vec2(tex_pos3.x, tex_pos3.y), 0.0) * w3.x * w3.y;

    return result;
}