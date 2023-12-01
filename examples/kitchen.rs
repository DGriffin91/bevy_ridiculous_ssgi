use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    input::common_conditions::input_toggle_active,
    pbr::{CascadeShadowConfigBuilder, DefaultOpaqueRendererMethod, PbrPlugin},
    prelude::*,
    window::PresentMode,
};
use bevy_basic_camera::{CameraController, CameraControllerPlugin};
use bevy_inspector_egui::quick::FilterQueryInspectorPlugin;
use bevy_mod_taa::{TAABundle, TAAPlugin};
use bevy_ridiculous_ssgi::{ssgi::SSGIPass, SSGIBundle, SSGIPlugin};

fn main() {
    App::new()
        .insert_resource(Msaa::Off)
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(Color::rgb(0.875, 0.95, 0.995) * 2.0))
        .insert_resource(AmbientLight {
            color: Color::rgb(1.0, 1.0, 1.0),
            brightness: 0.0,
        })
        .add_plugins((
            DefaultPlugins
                .set(PbrPlugin {
                    add_default_deferred_lighting_plugin: false,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        present_mode: PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                }),
            CameraControllerPlugin,
            TAAPlugin,
            SSGIPlugin,
            LogDiagnosticsPlugin::default(),
            FrameTimeDiagnosticsPlugin::default(),
            FilterQueryInspectorPlugin::<With<SSGIPass>>::default()
                .run_if(input_toggle_active(false, KeyCode::Tab)),
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(SceneBundle {
        scene: asset_server.load("models/kitchen.gltf#Scene0"),
        ..default()
    });

    // Sun
    commands.spawn(DirectionalLightBundle {
        transform: Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            (13.3f32).to_radians(),
            (180.0 - 14.2f32).to_radians(),
            0.0,
        )),
        directional_light: DirectionalLight {
            color: Color::rgb(0.95, 0.93, 0.85),
            illuminance: 120000.0,
            shadows_enabled: true,
            shadow_depth_bias: 0.02,
            shadow_normal_bias: 1.8,
        },
        cascade_shadow_config: CascadeShadowConfigBuilder {
            num_cascades: 3,
            maximum_distance: 10.0,
            ..default()
        }
        .into(),
        ..default()
    });

    // camera
    commands
        .spawn((
            Camera3dBundle {
                camera: Camera {
                    hdr: true,
                    ..default()
                },
                transform: Transform::from_xyz(3.4, 1.7, 1.0)
                    .looking_at(Vec3::new(-0.1, 1.2, 0.2), Vec3::Y),
                projection: Projection::Perspective(PerspectiveProjection {
                    fov: std::f32::consts::PI / 3.0,
                    ..default()
                }),
                ..default()
            },
            CameraController {
                walk_speed: 2.0,
                mouse_key_enable_mouse: MouseButton::Right,
                ..default()
            },
            SSGIBundle {
                ssgi_pass: SSGIPass {
                    mip_min: 4.0,
                    mip_max: 4.0,
                    ..default()
                },
                ..default()
            },
        ))
        // TAABundle has to be inserted because both SSGIBundle and TAABundle try to add some of the same components
        .insert(TAABundle::sample8());

    commands.spawn(SceneBundle {
        scene: asset_server.load("models/FlightHelmet/FlightHelmet.gltf#Scene0"),
        ..default()
    });
}
