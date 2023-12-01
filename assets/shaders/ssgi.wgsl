#import bevy_pbr::{
    prepass_utils,
    mesh_view_bindings::{view, globals},
}
#import bevy_pbr::utils::{octahedral_encode, octahedral_decode}

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput


#import bevy_pbr::{
    pbr_types::PbrInput,
    utils::PI,
}

#import bevy_pbr::mesh_view_bindings as view_bindings
#import bevy_pbr::view_transformations as vt
#import bevy_pbr::prepass_utils::prepass_motion_vector
#import ssgi::sampling as sampling
#import ssgi::sampling::TAU
#import ssgi::rgb9e5::{vec3_to_rgb9e5_, rgb9e5_to_vec3_}
#import ssgi::common as common

struct SSGIConfig {
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

@group(0) @binding(101) var prev_frame_tex: texture_2d<f32>;
@group(0) @binding(102) var prepass_downsample_normals: texture_2d<f32>;
@group(0) @binding(103) var prepass_downsample_depth: texture_2d<f32>;
@group(0) @binding(104) var prepass_downsample_motion: texture_2d<f32>;
@group(0) @binding(105) var nearest_sampler: sampler;
@group(0) @binding(106) var linear_sampler: sampler;
// todo webgl @group(0) @binding(108) var disocclusion_texture: texture_2d<f32>;
@group(0) @binding(109) var<uniform> config: SSGIConfig;
@group(0) @binding(110) var higher_cascade_data1: texture_2d<u32>;
@group(0) @binding(111) var higher_cascade_data2: texture_2d<u32>;

struct FragmentOutput {
    @location(0) data1: vec4<u32>,
    @location(1) data2: vec4<u32>,
}



@fragment
fn fragment(in: FullscreenVertexOutput) -> FragmentOutput {
    var out = vec4(0.0);

    let uposition = vec2<u32>(in.position.xy);
    let cas_xy = uposition / vec2(4u, config.directions / 4u);

    var frag_coord = vec4<f32>(common::frag_coord_for_cas(config.cascade_n, vec2<i32>(cas_xy), config.cas_0_render_scale), 0.0, 0.0);
    
    frag_coord.z = textureLoad(prepass_downsample_depth, vec2<i32>(frag_coord.xy), 0).x;
    let normal = octahedral_decode(textureLoad(prepass_downsample_normals, vec2<i32>(frag_coord.xy), 0).xy);

    let ws_pos = vt::position_ndc_to_world(vec3(vt::uv_to_ndc(in.uv), frag_coord.z));

    var pixel_radius = sampling::world_space_pixel_radius(-vt::depth_ndc_to_view_z(frag_coord.z));
    // limit minimum pixel radius for things really close to the camera
    pixel_radius = max(pixel_radius, 0.001); 

    return ssgi(frag_coord, cas_xy, uposition, ws_pos, normal);
}

// ----------------------------------------------------
// ----------------------------------------------------
// ----------------------------------------------------
// ----------------------------------------------------
// ----------------------------------------------------
// ----------------------------------------------------
// ----------------------------------------------------
// ----------------------------------------------------
// ----------------------------------------------------
// ----------------------------------------------------
// ----------------------------------------------------
// ----------------------------------------------------

fn ssgi(frag_coord_in: vec4<f32>, cas_xy: vec2<u32>, uposition: vec2<u32>, world_position: vec3<f32>, normal: vec3<f32>) -> FragmentOutput {
    var out: FragmentOutput;

    var color = vec3(0.0);

    let fcascade = f32(config.cascade_n);

    let cascade_exp = f32(1u << config.cascade_n);
    let last_cascade_exp = f32(1u << u32(max(i32(config.cascade_n) - 1, 0)));
    var interval_dist = cascade_exp * config.cascade_0_dist * sqrt(f32(config.cascade_n + 1u));
    
    var prev_interval_dist = last_cascade_exp * config.cascade_0_dist * sqrt(fcascade);
    prev_interval_dist = select(prev_interval_dist, 1.0, config.cascade_n == 0u);
    let prev_interval_dist_skip = prev_interval_dist * (1.0 - config.interval_overlap);

    let texel_size = 1.0 / view.viewport.zw;
    let frag_coord = frag_coord_in;
    let screen_uv = frag_coord.xy * texel_size;
    let ufrag_coord = vec2<u32>(frag_coord_in.xy);

    let ws_pos = world_position;

    let fdirections = f32(config.directions);
    let interval_length = interval_dist - prev_interval_dist_skip;

    let cascade_t = fcascade / f32(config.cascade_count);
    var fsteps = interval_length / f32(config.cas_0_render_scale / 2u);

    if config.divide_steps_by_square_of_cascade_exp == 1u {
        fsteps /= sqrt(cascade_exp);
    } else {
        fsteps /= cascade_exp;
    }

    let steps = u32(fsteps);
    var step_size = interval_length / fsteps;

    
    let phase = fract(f32(uposition.x % 4u + uposition.y % config.directions * 4u) / fdirections);
    let phase_with_offset = fract(phase + common::get_phase_noise_offset(f32(config.cas_0_directions)));
    var phi = TAU * phase_with_offset;
    let ss_dir = vec2(cos(phi), sin(phi));
    let V = normalize(view.world_position.xyz - world_position);

    var pixel_radius = sampling::world_space_pixel_radius(-vt::depth_ndc_to_view_z(frag_coord.z));
    // limit minimum pixel radius for things really close to the camera
    pixel_radius = max(pixel_radius, 0.001); 
    
    

    var samp_frag_coord = frag_coord.xy + ss_dir * prev_interval_dist_skip;

    let view_z_dir = vt::direction_view_to_world(vec3(0.0, 0.0, -1.0));

    var march_color = vec3(0.0);
    var march_brightness = 0.0;
    var march_hit_angle = 0.0;
    var max_occluded_angle = 0.0;

    var march_gather1 = vec3(0.0);
    var march_gather2 = vec3(0.0);
    var march_gather3 = vec3(0.0);
    var march_gather4 = vec3(0.0);
    var march_gather5 = vec3(0.0);
    var march_gather6 = vec3(0.0);
    var march_gather7 = vec3(0.0);
    var march_gather8 = vec3(0.0);

    var march_gather_weight = 0.0;
    var bitmask = 0u;
    let bitmask_steps = 32.0;

    // TODO needs more testing, for scaling light contribution with distance so really close things don't contribute disproportionately
    // Makes light contribution independant of screen res
    var pixel_radius_factor = pixel_radius * 200.0;
    for (var s = 0u; s < steps; s += 1u) {

        let fs = f32(s);

        samp_frag_coord += ss_dir * step_size;
        let samp_screen_uv = samp_frag_coord * texel_size;

        if (samp_screen_uv.x <= 0.0 || samp_screen_uv.y <= 0.0 || samp_screen_uv.x >= 1.0 || samp_screen_uv.y >= 1.0) {
            break;
        }

        let uv_dist = distance(samp_screen_uv, screen_uv);
        let ss_dist = distance(samp_frag_coord, frag_coord.xy);

        var mip = clamp(fcascade - 1.0, config.mip_min, config.mip_max);
        var depth_mip = clamp(fcascade - 1.0, config.depth_mip_min, config.mip_max);

        let closest_motion_vector = textureSampleLevel(prepass_downsample_motion, nearest_sampler, samp_screen_uv, mip).xy;
        let history_uv = samp_screen_uv - closest_motion_vector;

        if (history_uv.x <= 0.0 || history_uv.y <= 0.0 || history_uv.x >= 1.0 || history_uv.y >= 1.0) {
            break;
        }

        var samp_depth = textureSampleLevel(prepass_downsample_depth, nearest_sampler, samp_screen_uv, depth_mip).x;

        let samp_ndc = vec3(vt::uv_to_ndc(samp_screen_uv), max(samp_depth, 0.00000001));
        var samp_ws_pos = vt::position_ndc_to_world(samp_ndc);

        let to_sample = samp_ws_pos - ws_pos;
        let dir_to_sample = normalize(to_sample);
        
        let hit_angle = saturate(dot(V, dir_to_sample) * 0.5 + 0.5);

        //let visible = hit_angle > max_occluded_angle;
        var samp_bitmask = 1u << u32(round(bitmask_steps * hit_angle));
        let visible = (bitmask & samp_bitmask) == 0u || hit_angle > max_occluded_angle;
        max_occluded_angle = max(max_occluded_angle, hit_angle);

        let inside_current_interval = distance(frag_coord.xy, samp_frag_coord) > prev_interval_dist;

        if visible {
            // Don't contribute light on the overlap
            if inside_current_interval {
                let samp_color = textureSampleLevel(prev_frame_tex, nearest_sampler, history_uv, mip).xyz;
                let dist = length(to_sample);
                let samp_normal = octahedral_decode(textureSampleLevel(prepass_downsample_normals, nearest_sampler, samp_screen_uv, mip).xy);
                var hit_facing_sample = saturate(dot(samp_normal, -dir_to_sample) + config.backside_illumination);

                var falloff_dist = dist * config.falloff;
                falloff_dist = select(falloff_dist, falloff_dist * dist, config.square_falloff == 1u);

                var dist_falloff = saturate(1.0 / (1.0 + falloff_dist));
                let ws_d = distance(view.world_position.xyz, world_position);

                
                var horizon_occlusion = 1.0 - saturate(saturate(max_occluded_angle - hit_angle) * config.horizon_occlusion);
                var color = hit_facing_sample * samp_color * dist_falloff * config.brightness * pixel_radius_factor * horizon_occlusion;


                march_gather1 += color * common::angle_dist(0.111, 0.111, hit_angle);
                march_gather2 += color * common::angle_dist(0.222, 0.111, hit_angle);
                march_gather3 += color * common::angle_dist(0.333, 0.111, hit_angle);
                march_gather4 += color * common::angle_dist(0.444, 0.111, hit_angle);
                march_gather5 += color * common::angle_dist(0.555, 0.111, hit_angle);
                march_gather6 += color * common::angle_dist(0.666, 0.111, hit_angle);
                march_gather7 += color * common::angle_dist(0.777, 0.111, hit_angle);
                march_gather8 += color * common::angle_dist(0.888, 0.111, hit_angle);

            }

            bitmask |= samp_bitmask;
        }
        march_gather_weight += select(1.0, 1.0, inside_current_interval);
    }
    march_gather_weight = max(1.0, march_gather_weight);

    march_gather1 /= march_gather_weight;
    march_gather2 /= march_gather_weight;
    march_gather3 /= march_gather_weight;
    march_gather4 /= march_gather_weight;
    march_gather5 /= march_gather_weight;
    march_gather6 /= march_gather_weight;
    march_gather7 /= march_gather_weight;
    march_gather8 /= march_gather_weight;

    var gather1 = vec3(0.0);
    var gather2 = vec3(0.0);
    var gather3 = vec3(0.0);
    var gather4 = vec3(0.0);
    var gather5 = vec3(0.0);
    var gather6 = vec3(0.0);
    var gather7 = vec3(0.0);
    var gather8 = vec3(0.0);

    // If we're not the highest cascase_n then sample from the next cascade up
    if config.cascade_n < config.cascade_count - 1u { // 

        let higher_directions = config.directions * 2u; // The next cascade will have 2x the directions of this one

        let cas_coord = vec2(
            (f32(cas_xy.x) - 0.5) / 2.0,
            (f32(cas_xy.y) - 0.5) / 2.0,
        );
        let icas_coord = vec2<i32>(cas_coord);
        var fd = abs(cas_coord - ceil(cas_coord));
        var id = 1.0 - fd;

        var aa = fd.x * fd.y;
        var ba = id.x * fd.y;
        var ab = fd.x * id.y;
        var bb = id.x * id.y;

        common::weight_bilinear(
            &aa, &ba, &ab, &bb, 
            normal, world_position, frag_coord.z, 
            config.normal_rejection, config.distance_rejection, 
            icas_coord, config.cascade_n + 1u, config.cas_0_render_scale, 
            prepass_downsample_normals, prepass_downsample_depth,
            pixel_radius,
        );

        let directions_ratio = higher_directions / config.directions;

        let dir_step = 1.0 / f32(higher_directions);

        let half_dir_ratio = i32(directions_ratio) / 2;

        let dims = vec2<i32>(textureDimensions(higher_cascade_data1).xy) - 1;
        var gather = vec4(0.0);
        var weight = 0.0;

        let directions_div_4 = vec2(4, i32(higher_directions) / 4);

        var cdata1 = vec4(0u);
        var cdata2 = vec4(0u);
        for (var i = -half_dir_ratio; i <= half_dir_ratio; i += 1) {
            let fi = f32(i);
            var higher_dir = fract(phase + dir_step * fi) * f32(higher_directions);
            var ihigher_dir = clamp(i32(higher_dir), 0, i32(higher_directions));

            let ofs = vec2(ihigher_dir % 4, ihigher_dir / 4);

            let coords0 = (icas_coord + vec2(0, 0)) * directions_div_4 + ofs;
            let coords1 = (icas_coord + vec2(1, 0)) * directions_div_4 + ofs;
            let coords2 = (icas_coord + vec2(0, 1)) * directions_div_4 + ofs;
            let coords3 = (icas_coord + vec2(1, 1)) * directions_div_4 + ofs;

            cdata1 = textureLoad(higher_cascade_data1, clamp(coords0, vec2(0), dims), 0);
            cdata2 = textureLoad(higher_cascade_data2, clamp(coords0, vec2(0), dims), 0);
            gather1 += rgb9e5_to_vec3_(cdata1.x) * aa;
            gather2 += rgb9e5_to_vec3_(cdata1.y) * aa;
            gather3 += rgb9e5_to_vec3_(cdata1.z) * aa;
            gather4 += rgb9e5_to_vec3_(cdata1.w) * aa;
            gather5 += rgb9e5_to_vec3_(cdata2.x) * aa;
            gather6 += rgb9e5_to_vec3_(cdata2.y) * aa;
            gather7 += rgb9e5_to_vec3_(cdata2.z) * aa;
            gather8 += rgb9e5_to_vec3_(cdata2.w) * aa;
            cdata1 = textureLoad(higher_cascade_data1, clamp(coords1, vec2(0), dims), 0);
            cdata2 = textureLoad(higher_cascade_data2, clamp(coords1, vec2(0), dims), 0);
            gather1 += rgb9e5_to_vec3_(cdata1.x) * ba;
            gather2 += rgb9e5_to_vec3_(cdata1.y) * ba;
            gather3 += rgb9e5_to_vec3_(cdata1.z) * ba;
            gather4 += rgb9e5_to_vec3_(cdata1.w) * ba;
            gather5 += rgb9e5_to_vec3_(cdata2.x) * ba;
            gather6 += rgb9e5_to_vec3_(cdata2.y) * ba;
            gather7 += rgb9e5_to_vec3_(cdata2.z) * ba;
            gather8 += rgb9e5_to_vec3_(cdata2.w) * ba;
            cdata1 = textureLoad(higher_cascade_data1, clamp(coords2, vec2(0), dims), 0);
            cdata2 = textureLoad(higher_cascade_data2, clamp(coords2, vec2(0), dims), 0);
            gather1 += rgb9e5_to_vec3_(cdata1.x) * ab;
            gather2 += rgb9e5_to_vec3_(cdata1.y) * ab;
            gather3 += rgb9e5_to_vec3_(cdata1.z) * ab;
            gather4 += rgb9e5_to_vec3_(cdata1.w) * ab;
            gather5 += rgb9e5_to_vec3_(cdata2.x) * ab;
            gather6 += rgb9e5_to_vec3_(cdata2.y) * ab;
            gather7 += rgb9e5_to_vec3_(cdata2.z) * ab;
            gather8 += rgb9e5_to_vec3_(cdata2.w) * ab;
            cdata1 = textureLoad(higher_cascade_data1, clamp(coords3, vec2(0), dims), 0);
            cdata2 = textureLoad(higher_cascade_data2, clamp(coords3, vec2(0), dims), 0);
            gather1 += rgb9e5_to_vec3_(cdata1.x) * bb;
            gather2 += rgb9e5_to_vec3_(cdata1.y) * bb;
            gather3 += rgb9e5_to_vec3_(cdata1.z) * bb;
            gather4 += rgb9e5_to_vec3_(cdata1.w) * bb;
            gather5 += rgb9e5_to_vec3_(cdata2.x) * bb;
            gather6 += rgb9e5_to_vec3_(cdata2.y) * bb;
            gather7 += rgb9e5_to_vec3_(cdata2.z) * bb;
            gather8 += rgb9e5_to_vec3_(cdata2.w) * bb;

            weight += 1.0;
        }
        //gather1 /= max(weight, 1.0);
        //gather2 /= max(weight, 1.0);
        //gather3 /= max(weight, 1.0);
        //gather4 /= max(weight, 1.0);
  }
    
    gather1 *= get_vis(bitmask, max_occluded_angle, 0.111, bitmask_steps);
    gather2 *= get_vis(bitmask, max_occluded_angle, 0.222, bitmask_steps);
    gather3 *= get_vis(bitmask, max_occluded_angle, 0.333, bitmask_steps);
    gather4 *= get_vis(bitmask, max_occluded_angle, 0.444, bitmask_steps);
    gather5 *= get_vis(bitmask, max_occluded_angle, 0.555, bitmask_steps);
    gather6 *= get_vis(bitmask, max_occluded_angle, 0.666, bitmask_steps);
    gather7 *= get_vis(bitmask, max_occluded_angle, 0.777, bitmask_steps);
    gather8 *= get_vis(bitmask, max_occluded_angle, 0.888, bitmask_steps);

    //if config.cascade_n == 5u {
        gather1 += march_gather1;
        gather2 += march_gather2;
        gather3 += march_gather3;
        gather4 += march_gather4;
        gather5 += march_gather5;
        gather6 += march_gather6;
        gather7 += march_gather7;
        gather8 += march_gather8;
    //}

    out.data1.x = vec3_to_rgb9e5_(gather1);
    out.data1.y = vec3_to_rgb9e5_(gather2);
    out.data1.z = vec3_to_rgb9e5_(gather3);
    out.data1.w = vec3_to_rgb9e5_(gather4);
    out.data2.x = vec3_to_rgb9e5_(gather5);
    out.data2.y = vec3_to_rgb9e5_(gather6);
    out.data2.z = vec3_to_rgb9e5_(gather7);
    out.data2.w = vec3_to_rgb9e5_(gather8);

    return out;
}

fn count_bits(val_in: u32) -> u32 {
    var val = val_in;
    // Counts the number of 1:s
    // https://www.baeldung.com/cs/integer-bitcount
    val = (val&0x55555555u)+((val>>1u)&0x55555555u);
    val = (val&0x33333333u)+((val>>2u)&0x33333333u);
    val = (val&0x0F0F0F0Fu)+((val>>4u)&0x0F0F0F0Fu);
    val = (val&0x00FF00FFu)+((val>>8u)&0x00FF00FFu);
    val = (val&0x0000FFFFu)+((val>>16u)&0x0000FFFFu);
    return val;
}

fn get_vis(bitmask: u32, max_occluded_angle: f32, angle: f32, bitmask_steps: f32) -> f32 {
    var uangle = u32(round(angle * bitmask_steps));
    var bitvis = count_bits(((bitmask >> uangle) & 31u)); // 4 bit wide
    bitvis += count_bits(((bitmask >> (uangle - 1u)) & 127u)); // 6 bit wide, if using this divide vis by twice as much
    var vis = f32(bitvis);
    vis = mix(vis, 0.0, saturate(config.horizon_occlusion * 0.01));

    //vis = select(vis, 0.0, angle > max_occluded_angle);
    vis = mix(vis, 0.0, saturate((angle - max_occluded_angle) * 30.0)); // similar to above but with a softer cutoff

    vis /= 3.0; // TODO this could be between /1.0 and /8.0

    // include raw angle occlusion
    vis += saturate(saturate(max_occluded_angle - angle) * config.horizon_occlusion);

    return 1.0 - saturate(vis); 
}