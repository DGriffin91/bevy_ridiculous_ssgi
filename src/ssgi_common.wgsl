#define_import_path ssgi::common

#import bevy_pbr::view_transformations as vt
#import ssgi::sampling as sampling
#import bevy_pbr::utils::{octahedral_encode, octahedral_decode}
#import bevy_pbr::mesh_view_bindings::view

#import bevy_pbr::{
    mesh_view_bindings::globals,
}

fn ss_dir_to_ws_dir(ws_pos: vec3<f32>, screen_uv: vec2<f32>, ss_dir: vec2<f32>, ndc_depth: f32) -> vec3<f32> {
    let ws_pos2 = vt::position_ndc_to_world(vec3(vt::uv_to_ndc(screen_uv + ss_dir), ndc_depth));
    let ws_dir = normalize(ws_pos2 - ws_pos);
    return ws_dir;
}

fn ss_dir_to_vs_dir(vs_pos: vec3<f32>, screen_uv: vec2<f32>, ss_dir: vec2<f32>, ndc_depth: f32) -> vec3<f32> {
    let vs_pos2 = vt::position_ndc_to_view(vec3(vt::uv_to_ndc(screen_uv + ss_dir), ndc_depth));
    let vs_dir = normalize(vs_pos2 - vs_pos);
    return vs_dir;
}

// Computes a normalized distance between angle and x, scaled by interval, and clamped between 0.0 and 1.0
// If x == angle, output will be 1, if x == angle - interval output will be 0
fn angle_dist(angle: f32, interval: f32, x: f32) -> f32 {
    return 1.0 - saturate(abs(x - angle) / interval);
}

fn directional_angle_dist(angle: f32, interval: f32, x: f32) -> f32 {
    return 1.0 - saturate((x - angle) / interval);
}

// TODO find more accurate method
fn reconstruct_dir_to_sample(view_z: vec3<f32>, ws_dir: vec3<f32>, angle: f32) -> vec3<f32> {
    let cos_angle = angle * 2.0 - 1.0;
    var s = mix(ws_dir, view_z * sign(cos_angle), abs(cos_angle));
    return normalize(s);
}

fn get_phase_noise_offset(fdirections: f32) -> f32 {
#ifdef JITTER_PROBE_DIRECTION
    var phase_noise = sampling::blue_noise_for_pixel(vec2(2u, 2u), globals.frame_count % #{NOISE_FRAME_PERIOD}u) * 2.0 - 1.0;
    let phase_offset = 1.0 / fdirections * 0.5 * phase_noise;
#else
    let phase_offset = 0.0;
#endif
    return phase_offset;
}

fn frag_coord_for_cas(cascade_n: u32, cas_xy: vec2<i32>, cas_0_render_scale: u32) -> vec2<f32> {
    let render_scale = f32(cas_0_render_scale * (1u << cascade_n));

    let frame = globals.frame_count;
#ifdef JITTER_PROBE_POSITION
    var frag_noise = sampling::blue_noise_for_pixel(vec2(4u, 9u), frame % #{NOISE_FRAME_PERIOD}u) * 2.0 - 1.0;
    frag_noise = frag_noise * 0.5;
#else
    var frag_noise = 0.0;
#endif

    let render_offset = f32(render_scale * (0.5 + frag_noise));
    var frag_coord = vec2(vec2<f32>(cas_xy) * render_scale + render_offset);
    return frag_coord;
}

fn weight_bilinear(
    aa: ptr<function, f32>, 
    ba: ptr<function, f32>, 
    ab: ptr<function, f32>, 
    bb: ptr<function, f32>, 
    normal: vec3<f32>, 
    world_position: vec3<f32>, 
    ndc_depth: f32, 
    normal_rejection: f32, 
    distance_rejection: f32, 
    icas_coord: vec2<i32>, 
    cascade_n: u32, 
    cas_0_render_scale: u32,
    prepass_downsample_normals: texture_2d<f32>,
    prepass_downsample_depth: texture_2d<f32>,
    pixel_radius: f32,
) {
    var gather_frag_coord: vec2<i32>;
    var gather_depth: f32;
    var gather_normal: vec3<f32>;
    var gather_uv: vec2<f32>;
    var gather_ws_pos: vec3<f32>;
    var coplanar: bool;
    var dist_reject: f32;
    let m_pixel_radius = pixel_radius * 1000.0;

    gather_frag_coord = vec2<i32>(frag_coord_for_cas(cascade_n, (icas_coord + vec2(0, 0)), cas_0_render_scale));
    gather_uv = vec2<f32>(gather_frag_coord) / view.viewport.zw;
    gather_normal = octahedral_decode(textureLoad(prepass_downsample_normals, gather_frag_coord, 0).xy);
    gather_depth = textureLoad(prepass_downsample_depth, gather_frag_coord, 0).x;
    gather_ws_pos = vt::position_ndc_to_world(vec3(vt::uv_to_ndc(gather_uv), gather_depth));
    coplanar = sampling::coplanar(world_position, gather_normal, gather_ws_pos, normal, 0.95, pixel_radius * 5.0);
    dist_reject = select(distance_rejection, distance_rejection * 0.25, coplanar);
    *aa *= max(pow(dot(normal, gather_normal), normal_rejection), 0.001);
    *aa *= max(1.0 - saturate(distance(gather_ws_pos, world_position) / m_pixel_radius * dist_reject), 0.0);


    gather_frag_coord = vec2<i32>(frag_coord_for_cas(cascade_n, (icas_coord + vec2(1, 0)), cas_0_render_scale));
    gather_uv = vec2<f32>(gather_frag_coord) / view.viewport.zw;
    gather_normal = octahedral_decode(textureLoad(prepass_downsample_normals, gather_frag_coord, 0).xy);
    gather_depth = textureLoad(prepass_downsample_depth, gather_frag_coord, 0).x;
    gather_ws_pos = vt::position_ndc_to_world(vec3(vt::uv_to_ndc(gather_uv), gather_depth));
    coplanar = sampling::coplanar(world_position, gather_normal, gather_ws_pos, normal, 0.95, pixel_radius * 5.0);
    dist_reject = select(distance_rejection, distance_rejection * 0.25, coplanar);
    *ba *= max(pow(dot(normal, gather_normal), normal_rejection), 0.001);
    *ba *= max(1.0 - saturate(distance(gather_ws_pos, world_position) / m_pixel_radius * dist_reject), 0.0);

    gather_frag_coord = vec2<i32>(frag_coord_for_cas(cascade_n, (icas_coord + vec2(0, 1)), cas_0_render_scale));
    gather_uv = vec2<f32>(gather_frag_coord) / view.viewport.zw;
    gather_normal = octahedral_decode(textureLoad(prepass_downsample_normals, gather_frag_coord, 0).xy);
    gather_depth = textureLoad(prepass_downsample_depth, gather_frag_coord, 0).x;
    gather_ws_pos = vt::position_ndc_to_world(vec3(vt::uv_to_ndc(gather_uv), gather_depth));
    coplanar = sampling::coplanar(world_position, gather_normal, gather_ws_pos, normal, 0.95, pixel_radius * 5.0);
    dist_reject = select(distance_rejection, distance_rejection * 0.25, coplanar);
    *ab *= max(pow(dot(normal, gather_normal), normal_rejection), 0.001);
    *ab *= max(1.0 - saturate(distance(gather_ws_pos, world_position) / m_pixel_radius * dist_reject), 0.0);

    gather_frag_coord = vec2<i32>(frag_coord_for_cas(cascade_n, (icas_coord + vec2(1, 1)), cas_0_render_scale));
    gather_uv = vec2<f32>(gather_frag_coord) / view.viewport.zw;
    gather_normal = octahedral_decode(textureLoad(prepass_downsample_normals, gather_frag_coord, 0).xy);
    gather_depth = textureLoad(prepass_downsample_depth, gather_frag_coord, 0).x;
    gather_ws_pos = vt::position_ndc_to_world(vec3(vt::uv_to_ndc(gather_uv), gather_depth));
    coplanar = sampling::coplanar(world_position, gather_normal, gather_ws_pos, normal, 0.95, pixel_radius * 5.0);
    dist_reject = select(distance_rejection, distance_rejection * 0.25, coplanar);
    *bb *= max(pow(dot(normal, gather_normal), normal_rejection), 0.001);
    *bb *= max(1.0 - saturate(distance(gather_ws_pos, world_position) / m_pixel_radius * dist_reject), 0.0);

    // Renormalize
    let sum = 1.0 / (*aa + *ba + *ab + *bb);
    *aa *= sum;
    *ba *= sum;
    *ab *= sum;
    *bb *= sum;
}