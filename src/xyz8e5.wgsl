#define_import_path ssgi::xyz8e5
// https://github.com/DGriffin91/shared_exponent_formats/blob/main/src/wgsl/xyz8e5.wgsl

const XYZ8E5_EXPONENT_BITS        = 5u;
const XYZ8E5_MANTISSA_BITS        = 8;
const XYZ8E5_MANTISSA_BITSU       = 8u;
const XYZ8E5_EXP_BIAS             = 15;
const XYZ8E5_MAX_VALID_BIASED_EXP = 31u;

const MAX_XYZ8E5_EXP              = 16u;
const XYZ8E5_MANTISSA_VALUES      = 256;
const MAX_XYZ8E5_MANTISSA         = 255;
const MAX_XYZ8E5_MANTISSAU        = 255u;
const MAX_XYZ8E5_                 = 65280.0;
const EPSILON_XYZ8E5_             = 0.00000011920929;

fn floor_log2_(x: f32) -> i32 {
    let f = bitcast<u32>(x);
    let biasedexponent = (f & 0x7F800000u) >> 23u;
    return i32(biasedexponent) - 127;
}

fn is_sign_negative(v: f32) -> u32 {
    return (bitcast<u32>(v) >> 31u) & 1u;
}

// Similar to https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_shared_exponent.txt
fn vec3_to_xyz8e5_(xyz_in: vec3<f32>) -> u32 {
    let xsign = is_sign_negative(xyz_in.x) << 8u;
    let ysign = is_sign_negative(xyz_in.y) << 8u;
    let zsign = is_sign_negative(xyz_in.z) << 8u;

    var xyz = min(abs(xyz_in), vec3(MAX_XYZ8E5_));

    let maxxyz = max(xyz.x, max(xyz.y, xyz.z));
    var exp_shared = max(-XYZ8E5_EXP_BIAS - 1, floor_log2_(maxxyz)) + 1 + XYZ8E5_EXP_BIAS;
    var denom = exp2(f32(exp_shared - XYZ8E5_EXP_BIAS - XYZ8E5_MANTISSA_BITS));

    let maxm = i32(floor(maxxyz / denom + 0.5));
    if (maxm == XYZ8E5_MANTISSA_VALUES) {
        denom *= 2.0;
        exp_shared += 1;
    }

    let s = vec3<u32>(floor(xyz / denom + 0.5));

    return (u32(exp_shared) << 27u) | ((s.z | zsign) << 18u) | ((s.y | ysign) << 9u) | ((s.x | xsign) << 0u);
}

// Builtin extractBits() is not working on WEBGL or DX12
// DX12: HLSL: Unimplemented("write_expr_math ExtractBits")
fn extract_bits(value: u32, offset: u32, bits: u32) -> u32 {
    let mask = (1u << bits) - 1u;
    return (value >> offset) & mask;
}

fn xyz8e5_to_vec3_(v: u32) -> vec3<f32> {
    let exponent = i32(extract_bits(v, 27u, XYZ8E5_EXPONENT_BITS)) - XYZ8E5_EXP_BIAS - XYZ8E5_MANTISSA_BITS;
    let scale = exp2(f32(exponent));

    // Extract both the mantissa and sign at the same time.
    let xb = extract_bits(v,  0u, XYZ8E5_MANTISSA_BITSU + 1u);
    let yb = extract_bits(v,  9u, XYZ8E5_MANTISSA_BITSU + 1u);
    let zb = extract_bits(v, 18u, XYZ8E5_MANTISSA_BITSU + 1u);

    // xb & 0xFFu masks out for just the mantissa
    // xb & 0x100u << 23u masks out just the sign bit and shifts it over 
    // to the corresponding IEEE 754 sign location 
    return vec3(
        bitcast<f32>(bitcast<u32>(f32(xb & 0xFFu)) | (xb & 0x100u) << 23u),
        bitcast<f32>(bitcast<u32>(f32(yb & 0xFFu)) | (yb & 0x100u) << 23u),
        bitcast<f32>(bitcast<u32>(f32(zb & 0xFFu)) | (zb & 0x100u) << 23u),
    ) * scale;
}