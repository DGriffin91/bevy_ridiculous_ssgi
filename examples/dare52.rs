use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    input::common_conditions::input_toggle_active,
    math::vec3,
    pbr::{DefaultOpaqueRendererMethod, PbrPlugin},
    prelude::*,
    render::view::ColorGrading,
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
            FilterQueryInspectorPlugin::<With<SSGIPass>>::default()
                .run_if(input_toggle_active(false, KeyCode::Tab)),
        ))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(SceneBundle {
        scene: asset_server.load("models/dare52.glb#Scene0"),
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
                transform: Transform::from_xyz(4.0, 4.0, 4.0)
                    .looking_at(vec3(1.0, 0.0, 1.0), Vec3::Y),
                color_grading: ColorGrading {
                    exposure: 0.0,
                    ..default()
                },
                ..default()
            },
            CameraController {
                walk_speed: 2.0,
                mouse_key_enable_mouse: MouseButton::Right,
                ..default()
            },
            SSGIBundle {
                ssgi_pass: SSGIPass {
                    brightness: 1.5,
                    square_falloff: true,
                    horizon_occlusion: 100.0,
                    render_scale: 6,
                    cascade_0_directions: 8,
                    interval_overlap: 1.0,
                    mip_min: 3.0,
                    mip_max: 4.0,
                    divide_steps_by_square_of_cascade_exp: false,
                    ..default()
                },
                ..default()
            },
        ))
        .insert(TAABundle::sample8());
}
