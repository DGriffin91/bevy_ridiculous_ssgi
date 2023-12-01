use bevy::ecs::prelude::*;
use bevy::input::common_conditions::input_toggle_active;
use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    math::vec3,
    pbr::{DefaultOpaqueRendererMethod, PbrPlugin},
    prelude::*,
    window::PresentMode,
};
use bevy_basic_camera::{CameraController, CameraControllerPlugin};
use bevy_inspector_egui::quick::FilterQueryInspectorPlugin;
use bevy_mod_taa::{TAABundle, TAAPlugin};
use bevy_ridiculous_ssgi::ssgi::SSGIPass;
use bevy_ridiculous_ssgi::{SSGIBundle, SSGIPlugin};

fn main() {
    App::new()
        .insert_resource(Msaa::Off)
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(Color::BLACK))
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
        ))
        .add_plugins(
            FilterQueryInspectorPlugin::<With<SSGIPass>>::default()
                .run_if(input_toggle_active(false, KeyCode::Tab)),
        )
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(SceneBundle {
        scene: asset_server.load("models/cornell_box.glb#Scene0"),
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
                transform: Transform::from_xyz(0.0, 1.0, 4.6)
                    .looking_at(vec3(0.0, 1.0, 0.0), Vec3::Y),
                projection: Projection::Perspective(PerspectiveProjection {
                    fov: std::f32::consts::PI / 6.0,
                    near: 0.1,
                    far: 1000.0,
                    aspect_ratio: 1.0,
                }),
                ..default()
            },
            CameraController {
                walk_speed: 2.0,
                mouse_key_enable_mouse: MouseButton::Right,
                ..default()
            },
            SSGIBundle::default(),
        ))
        .insert(TAABundle::sample8());
}
