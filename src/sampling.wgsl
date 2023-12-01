#define_import_path ssgi::sampling

#import bevy_pbr::mesh_view_bindings::{globals, view}

@group(#{BLUE_NOISE_GROUP_N}) @binding(#{BLUE_NOISE_ENTRY_N})
var blue_noise_tex: texture_2d_array<f32>;
const BLUE_NOISE_TEX_DIMS = vec3<u32>(#{BLUE_NOISE_DIMS}u, #{BLUE_NOISE_DIMS}u, #{BLUE_NOISE_DIMS}u);

const PHI = 1.618033988749895; // Golden Ratio
const TAU = 6.28318530717958647692528676655900577;

const INV_TAU: f32 = 0.159154943;
const PHIMINUS1_: f32 = 0.61803398875;

const F32_EPSILON: f32 = 1.1920929E-7;
const F32_MAX: f32 = 3.402823466E+38;
const F16_MAX: f32 = 65504.0;

const U32_MAX: u32 = 0xFFFFFFFFu;

const PI: f32 = 3.141592653589793;
const HALF_PI: f32 = 1.57079632679;
const E: f32 = 2.718281828459045;

fn RGB_to_YCoCg(rgb: vec3<f32>) -> vec3<f32> {
    let y = (rgb.r / 4.0) + (rgb.g / 2.0) + (rgb.b / 4.0);
    let co = (rgb.r / 2.0) - (rgb.b / 2.0);
    let cg = (-rgb.r / 4.0) + (rgb.g / 2.0) - (rgb.b / 4.0);
    return vec3(y, co, cg);
}

fn YCoCg_to_RGB(ycocg: vec3<f32>) -> vec3<f32> {
    let r = ycocg.x + ycocg.y - ycocg.z;
    let g = ycocg.x + ycocg.z;
    let b = ycocg.x - ycocg.y - ycocg.z;
    return saturate(vec3(r, g, b));
}

fn uniform_sample_sphere(urand: vec2<f32>) -> vec3<f32> {
    let theta = 2.0 * PI * urand.y;
    let z = 1.0 - 2.0 * urand.x;
    let xy = sqrt(max(0.0, 1.0 - z * z));
    let sn = sin(theta);
    let cs = cos(theta);
    return vec3(cs * xy, sn * xy, z);
}

fn uniform_sample_disc(urand: vec2<f32>) -> vec3<f32> {
    let r = sqrt(urand.x);
    let theta = urand.y * TAU;

    let x = r * cos(theta);
    let y = r * sin(theta);

    return vec3(x, y, 0.0);
}

fn uniform_sample_circle(urand: f32) -> vec2<f32> {
    let theta = urand * TAU;

    let x = cos(theta);
    let y = sin(theta);

    return vec2(x, y);
}

fn uniform_sample_cone(urand: vec2<f32>, cos_theta_max: f32) -> vec3<f32> {
    let cos_theta = (1.0 - urand.x) + urand.x * cos_theta_max;
    let sin_theta = sqrt(saturate(1.0 - cos_theta * cos_theta));
    let phi = urand.y * TAU;
    return vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

fn cosine_sample_hemisphere(urand: vec2<f32>) -> vec3<f32> {
    let r = sqrt(urand.x);
    let theta = urand.y * TAU;

    let x = r * cos(theta);
    let y = r * sin(theta);

    return vec3(x, y, sqrt(max(0.0, 1.0 - urand.x)));
}

const M_PLASTIC = 1.32471795724474602596;

fn r2_sequence(i: u32) -> vec2<f32> {
    let a1 = 1.0 / M_PLASTIC;
    let a2 = 1.0 / (M_PLASTIC * M_PLASTIC);
    return fract(vec2(a1, a2) * f32(i) + 0.5);
}

fn blue_noise_for_pixel(px: vec2<u32>, layer: u32) -> f32 {
    return textureLoad(blue_noise_tex, px % BLUE_NOISE_TEX_DIMS.xy, i32(layer % BLUE_NOISE_TEX_DIMS.z), 0).x * 255.0 / 256.0 + 0.5 / 256.0;
}

fn fract_blue_noise_for_pixel(ufrag_coord: vec2<u32>, seed: u32, white_frame_noise: vec4<f32>) -> vec4<f32> {
    return vec4(
        fract(blue_noise_for_pixel(ufrag_coord, seed + 0u) + white_frame_noise.x),
        fract(blue_noise_for_pixel(ufrag_coord, seed + 1u) + white_frame_noise.y),
        fract(blue_noise_for_pixel(ufrag_coord, seed + 2u) + white_frame_noise.z),
        fract(blue_noise_for_pixel(ufrag_coord, seed + 3u) + white_frame_noise.w)
    );
}

fn fract_white_noise_for_pixel(ufrag_coord: vec2<u32>, seed: u32, white_frame_noise: vec4<f32>) -> vec4<f32> {
    return vec4(
        fract(hash_noise(ufrag_coord, seed + 0u) + white_frame_noise.x),
        fract(hash_noise(ufrag_coord, seed + 1u) + white_frame_noise.y),
        fract(hash_noise(ufrag_coord, seed + 2u) + white_frame_noise.z),
        fract(hash_noise(ufrag_coord, seed + 3u) + white_frame_noise.w)
    );
}

fn uhash(a: u32, b: u32) -> u32 { 
    var x = ((a * 1597334673u) ^ (b * 3812015801u));
    // from https://nullprogram.com/blog/2018/07/31/
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

fn unormf(n: u32) -> f32 { 
    return f32(n) * (1.0 / f32(0xffffffffu)); 
}

fn hash_noise(ufrag_coord: vec2<u32>, frame: u32) -> f32 {
    let urnd = uhash(ufrag_coord.x, (ufrag_coord.y << 11u) + frame);
    return unormf(urnd);
}

// Warning: only good for 4096 frames. Don't use with super long frame accumulation
fn white_frame_noise(seed: u32) -> vec4<f32> {
    return vec4(
        hash_noise(vec2(0u + seed), globals.frame_count + seed), 
        hash_noise(vec2(1u + seed), globals.frame_count + 4096u + seed),
        hash_noise(vec2(2u + seed), globals.frame_count + 8192u + seed),
        hash_noise(vec2(3u + seed), globals.frame_count + 12288u + seed)
    );
}

// https://blog.demofox.org/2022/01/01/interleaved-gradient-noise-a-different-kind-of-low-discrepancy-sequence
fn interleaved_gradient_noise(pixel_coordinates: vec2<f32>, frame_in: u32) -> f32 {
    let frame = f32(frame_in % 64u);
    let xy = pixel_coordinates + 5.588238 * frame;
    return fract(52.9829189 * fract(0.06711056 * xy.x + 0.00583715 * xy.y));
}

// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
fn build_orthonormal_basis(n: vec3<f32>) -> mat3x3<f32> {
    var b1: vec3<f32>;
    var b2: vec3<f32>;

    if (n.z < 0.0) {
        let a = 1.0 / (1.0 - n.z);
        let b = n.x * n.y * a;
        b1 = vec3(1.0 - n.x * n.x * a, -b, n.x);
        b2 = vec3(b, n.y * n.y * a - 1.0, -n.y);
    } else {
        let a = 1.0 / (1.0 + n.z);
        let b = -n.x * n.y * a;
        b1 = vec3(1.0 - n.x * n.x * a, b, -n.x);
        b2 = vec3(b, 1.0 - n.y * n.y * a, -n.y);
    }

    return mat3x3<f32>(
        b1.x, b2.x, n.x,
        b1.y, b2.y, n.y,
        b1.z, b2.z, n.z
    );
}

// https://github.com/NVIDIAGameWorks/RayTracingDenoiser/blob/3c881ae3075f7ca754e22177877335b82e16da5a/Shaders/Include/Common.hlsli#L124
fn world_space_pixel_radius(linear_depth: f32) -> f32 {
    // https://github.com/NVIDIAGameWorks/RayTracingDenoiser/blob/3c881ae3075f7ca754e22177877335b82e16da5a/Source/Sigma.cpp#L107
    let is_orthographic = view.projection[3].w == 1.0;
    let unproject = 1.0 / (0.5 * view.viewport.w * view.projection[1][1]);
    return unproject * select(linear_depth, 1.0, is_orthographic);
}


fn projection_is_orthographic() -> bool {
    return view.projection[3].w == 1.0;
}

fn coplanar(pos1: vec3<f32>, normal1: vec3<f32>, pos2: vec3<f32>, normal2: vec3<f32>, nor_epsilon: f32, ws_epsilon: f32) -> bool {
    if dot(normal1, normal2) > clamp(nor_epsilon, -1.0, 1.0) {
        let D1 = -dot(normal1, pos1);
        return abs(dot(normal1, pos2) + D1) < ws_epsilon;
    } else {
        return false;
    }
}



// Sampling Visible GGX Normals with Spherical Caps
// https://arxiv.org/pdf/2306.05044.pdf

// Helper function: sample the visible hemisphere from a spherical cap
fn sample_vndf_Hemisphere(u: vec2<f32>, wi: vec3<f32>) -> vec3<f32> {
	// sample a spherical cap in (-wi.z, 1]
	let phi = 2.0 * PI * u.x;
	let z = fma((1.0 - u.y), (1.0 + wi.z), -wi.z);
	let sinTheta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
	let x = sinTheta * cos(phi);
	let y = sinTheta * sin(phi);
	let c = vec3(x, y, z);
	// compute halfway direction;
	let h = c + wi;
	// return without normalization as this is done later (see line 25)
	return h;
}

// Sample the GGX VNDF
fn sample_vndf_GGX(urand: vec2<f32>, wi: vec3<f32>, alpha: vec2<f32>) -> vec3<f32> {
    let u = urand.yx;
    // warp to the hemisphere configuration
    let wiStd = normalize(vec3(wi.xy * alpha, wi.z));
    // sample the hemisphere
    let wmStd = sample_vndf_Hemisphere(u, wiStd);
    // warp back to the ellipsoid configuration
    let wm = normalize(vec3(wmStd.xy * alpha, wmStd.z));
    // return final normal
    return wm;
}

// ------------------------
// BRDF stuff from kajiya
// ------------------------

fn ggx_ndf(a2: f32, cos_theta: f32) -> f32 {
    let denom_sqrt = cos_theta * cos_theta * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom_sqrt * denom_sqrt);
}

fn g_smith_ggx1_(ndotv: f32, a2: f32) -> f32 {
    let tan2_v = (1.0 - ndotv * ndotv) / (ndotv * ndotv);
    return 2.0 / (1.0 + sqrt(1.0 + a2 * tan2_v));
}

fn pdf_ggx_vn(a2: f32, wo: vec3<f32>, h: vec3<f32>) -> f32 {
    let g1 = g_smith_ggx1_(wo.z, a2);
    let d = ggx_ndf(a2, h.z);
    return g1 * d * max(0.f, dot(wo, h)) / wo.z;
}

struct NdfSample {
    m: vec3<f32>,
    pdf: f32,
};

// From https://github.com/h3r2tic/kajiya/blob/d3b6ac22c5306cc9d3ea5e2d62fd872bea58d8d6/assets/shaders/inc/brdf.hlsl#LL182C1-L214C6
// https://github.com/NVIDIAGameWorks/Falcor/blob/c0729e806045731d71cfaae9d31a992ac62070e7/Source/Falcor/Experimental/Scene/Material/Microfacet.slang
// https://jcgt.org/published/0007/04/01/paper.pdf
fn sample_vndf(alpha: f32, wo: vec3<f32>, urand: vec2<f32>) -> NdfSample {
    let alpha_x = alpha;
    let alpha_y = alpha;
    let a2 = alpha_x * alpha_y;

    let h = sample_vndf_GGX(urand, wo, vec2(alpha_x, alpha_y));

    let pdf = pdf_ggx_vn(a2, wo, h);

    var res: NdfSample;
    res.m = h;
    res.pdf = pdf;
    return res;
}

fn eval_fresnel_schlick(f0: vec3<f32>, f90: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    return mix(f0, f90, pow(max(0.0, 1.0 - cos_theta), 5.0));
}

// Defined wrt the projected solid angle metric
struct BrdfSample {
    value_over_pdf: vec3<f32>,
    value: vec3<f32>,
    pdf: f32,

    transmission_fraction: vec3<f32>,

    wi: vec3<f32>,

    // For filtering / firefly suppression
    approx_roughness: f32,
}

fn BrdfSample_invalid() -> BrdfSample {
    var res: BrdfSample;
    res.value_over_pdf = vec3(0.0);
    res.pdf = 0.0;
    res.wi = vec3(0.0, 0.0, -1.0);
    res.transmission_fraction = vec3(0.0);
    res.approx_roughness = 0.0;
    return res;
}

struct SmithShadowingMasking {
    g: f32,
    g_over_g1_wo: f32,
}

fn SmithShadowingMasking_eval(ndotv: f32, ndotl: f32, a2: f32) -> SmithShadowingMasking {
    var res: SmithShadowingMasking;
    res.g = g_smith_ggx1_(ndotl, a2) * g_smith_ggx1_(ndotv, a2);
    res.g_over_g1_wo = g_smith_ggx1_(ndotl, a2);
    return res;
}

const BRDF_SAMPLING_MIN_COS = 1e-5;

fn brdf_sample(roughness: f32, F0: vec3<f32>, wo: vec3<f32>, urand: vec2<f32>) -> BrdfSample {
    let ndf_sample = sample_vndf(roughness, wo, urand);

    let wi = reflect(-wo, ndf_sample.m);

    if (ndf_sample.m.z <= BRDF_SAMPLING_MIN_COS || wi.z <= BRDF_SAMPLING_MIN_COS || wo.z <= BRDF_SAMPLING_MIN_COS) {
        return BrdfSample_invalid();
    }

    // Change of variables from the half-direction space to regular lighting geometry.
    let jacobian = 1.0 / (4.0 * dot(wi, ndf_sample.m));

    let fresnel = eval_fresnel_schlick(F0, vec3(1.0), dot(ndf_sample.m, wi));
    let a2 = roughness * roughness;
    let cos_theta = ndf_sample.m.z;

    let shadowing_masking = SmithShadowingMasking_eval(wo.z, wi.z, a2);

    var res: BrdfSample;
    res.pdf = ndf_sample.pdf * jacobian / wi.z;
    res.wi = wi;
    res.transmission_fraction = vec3(1.0) - fresnel;
    res.approx_roughness = roughness;

    res.value_over_pdf = fresnel * shadowing_masking.g_over_g1_wo;

    res.value =
            fresnel
            * shadowing_masking.g
            * ggx_ndf(a2, cos_theta)
            / (4.0 * wo.z * wi.z);

    return res;
}

// https://jcgt.org/published/0003/02/01/paper.pdf

// For encoding normals or unit direction vectors as octahedral coordinates.
fn octa_encode(v: vec3<f32>) -> vec2<f32> {
    var n = v / (abs(v.x) + abs(v.y) + abs(v.z));
    let octahedral_wrap = (1.0 - abs(n.yx)) * select(vec2(-1.0), vec2(1.0), n.xy > 0.0);
    let n_xy = select(octahedral_wrap, n.xy, n.z >= 0.0);
    return n_xy * 0.5 + 0.5;
}

// For decoding normals or unit direction vectors from octahedral coordinates.
fn octa_decode(v: vec2<f32>) -> vec3<f32> {
    let f = v * 2.0 - 1.0;
    var n = vec3(f.xy, 1.0 - abs(f.x) - abs(f.y));
    let t = saturate(-n.z);
    let w = select(vec2(t), vec2(-t), n.xy >= vec2(0.0));
    n = vec3(n.xy + w, n.z);
    return normalize(n);
}

const U8MAXF = 255.0;

fn pack_16bit_nor(oct_nor: vec2<f32>) -> u32 {
    let unorm1 = u32(saturate(oct_nor.x) * U8MAXF + 0.5);
    let unorm2 = u32(saturate(oct_nor.y) * U8MAXF + 0.5);
    return (unorm1 & 0xFFu) | ((unorm2 & 0xFFu) << 8u);
}

fn unpack_16bit_nor(packed: u32) -> vec2<f32> {
    let unorm1 = packed & 0xFFu;
    let unorm2 = (packed >> 8u) & 0xFFu;
    return vec2(f32(unorm1) / U8MAXF, f32(unorm2) / U8MAXF);
}

#import bevy_pbr::lighting::specular
fn bevy_light(roughness: f32, NdotV: f32, normal: vec3<f32>, V: vec3<f32>, R: vec3<f32>, F0: vec3<f32>, f_ab: vec2<f32>, lightColor: vec3<f32>, direction_to_light: vec3<f32>, neighbor_pdf: f32) -> vec3<f32> {
    let incident_light = direction_to_light;

    let half_vector = normalize(incident_light + V);
    let NoL = saturate(dot(normal, incident_light));
    let NoH = saturate(dot(normal, half_vector));
    let LoH = saturate(dot(incident_light, half_vector));

    let specular_light = specular(F0, roughness, half_vector, NdotV, NoL, NoH, LoH, 1.0, f_ab);

    return (specular_light / neighbor_pdf) * lightColor * NoL;
}