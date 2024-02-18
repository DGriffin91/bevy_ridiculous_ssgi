pub mod bind_group_utils;
pub mod copy_frame;
pub mod lighting_pass;
pub mod prepass_downsample;
pub mod ssgi;
pub mod ssgi_generate_sh;
pub mod ssgi_resolve;

use bevy::{
    asset::load_internal_asset,
    core_pipeline::prepass::{DeferredPrepass, DepthPrepass, MotionVectorPrepass},
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssetUsages,
        texture::{CompressedImageFormats, ImageType},
    },
};
use bevy_mod_taa::disocclusion::DisocclusionSettings;
use copy_frame::{CopyFrame, CopyFramePlugin};
use lighting_pass::CustomDeferredPbrLightingPlugin;
use prepass_downsample::{PrepassDownsample, PrepassDownsamplePlugin};
use ssgi::{SSGIPass, SSGISamplePlugin};
use ssgi_generate_sh::{SSGIGenerateSH, SSGIGenerateSHPlugin};
use ssgi_resolve::{SSGIResolve, SSGIResolvePlugin};

pub const RGB9E5_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(285370495827304598);
pub const XYZ8E5_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(572304958723049851);
pub const SAMPLING_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(20394857203948570);
pub const SSGI_COMMON_SHADER_HANDLE: Handle<Shader> = Handle::weak_from_u128(10429385740952873);

pub struct SSGIPlugin;
impl Plugin for SSGIPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, RGB9E5_SHADER_HANDLE, "rgb9e5.wgsl", Shader::from_wgsl);
        load_internal_asset!(app, XYZ8E5_SHADER_HANDLE, "xyz8e5.wgsl", Shader::from_wgsl);
        load_internal_asset!(
            app,
            SAMPLING_SHADER_HANDLE,
            "sampling.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SSGI_COMMON_SHADER_HANDLE,
            "ssgi_common.wgsl",
            Shader::from_wgsl
        );

        app.add_systems(Startup, load_blue_noise)
            .add_systems(Update, add_disocclusion_settings)
            .add_plugins((
                ExtractResourcePlugin::<BlueNoise>::default(),
                CustomDeferredPbrLightingPlugin,
                CopyFramePlugin,
                PrepassDownsamplePlugin,
                SSGISamplePlugin,
                SSGIGenerateSHPlugin,
                SSGIResolvePlugin,
            ));
        // todo webgl
        //if !app.is_plugin_added::<DisocclusionPlugin>() {
        //    app.add_plugins(DisocclusionPlugin);
        //}
    }
}

fn add_disocclusion_settings(
    mut commands: Commands,
    query: Query<Entity, (With<SSGIPass>, Without<DisocclusionSettings>)>,
) {
    for entity in &query {
        commands
            .entity(entity)
            .insert(DisocclusionSettings::default());
    }
}

/// Bundle to apply SSGI
#[derive(Bundle, Default)]
pub struct SSGIBundle {
    pub copy_frame: CopyFrame,
    pub prepass_downsample: PrepassDownsample,
    pub ssgi_pass: SSGIPass,
    pub ssgi_generate_sh: SSGIGenerateSH,
    pub ssgi_resolve: SSGIResolve,
    pub deferred_prepass: DeferredPrepass,
    pub depth_prepass: DepthPrepass,
    pub motion_vector_prepass: MotionVectorPrepass,
}

#[derive(Resource, ExtractResource, Clone)]
pub struct BlueNoise(pub Handle<Image>);

pub const BLUE_NOISE_GROUP_N: u32 = 1;
pub const BLUE_NOISE_ENTRY_N: u32 = 31;
pub const BLUE_NOISE_DIMS: u32 = 64;

pub fn load_blue_noise(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    commands.insert_resource(BlueNoise(images.add(setup_blue_noise_image(
        include_bytes!("blue_noise_64x64_l64.dds"),
        ImageType::Extension("dds"),
    ))));
}

fn setup_blue_noise_image(bytes: &[u8], image_type: ImageType) -> Image {
    let image_sampler = bevy::render::texture::ImageSampler::Descriptor(
        bevy::render::texture::ImageSamplerDescriptor {
            label: Some("blue_noise_64x64_l64".to_string()),
            address_mode_u: bevy::render::texture::ImageAddressMode::Repeat,
            address_mode_v: bevy::render::texture::ImageAddressMode::Repeat,
            address_mode_w: bevy::render::texture::ImageAddressMode::Repeat,
            mag_filter: bevy::render::texture::ImageFilterMode::Nearest,
            min_filter: bevy::render::texture::ImageFilterMode::Nearest,
            mipmap_filter: bevy::render::texture::ImageFilterMode::Nearest,
            ..default()
        },
    );
    Image::from_buffer(
        String::from("Blue Noise"),
        bytes,
        image_type,
        CompressedImageFormats::NONE,
        false,
        image_sampler,
        RenderAssetUsages::default(),
    )
    .unwrap()
}
