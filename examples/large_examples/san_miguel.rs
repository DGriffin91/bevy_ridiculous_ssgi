use std::f32::consts::PI;

use bevy::input::common_conditions::input_toggle_active;
use bevy::{
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    input::mouse::MouseMotion,
    math::vec3,
    pbr::{
        CascadeShadowConfigBuilder, DefaultOpaqueRendererMethod, PbrPlugin,
        TransmittedShadowReceiver,
    },
    prelude::*,
    render::view::ColorGrading,
    window::PresentMode,
};
use bevy_basic_camera::{CameraController, CameraControllerPlugin};
use bevy_inspector_egui::quick::FilterQueryInspectorPlugin;
use bevy_mod_mipmap_generator::{MipmapGeneratorPlugin, MipmapGeneratorSettings};
use bevy_mod_taa::{TAABundle, TAAPlugin};
use bevy_ridiculous_ssgi::{ssgi::SSGIPass, SSGIBundle, SSGIPlugin};

fn main() {
    App::new()
        .insert_resource(Msaa::Off)
        .insert_resource(DefaultOpaqueRendererMethod::deferred())
        .insert_resource(ClearColor(Color::rgb(1.75, 1.8, 2.1)))
        .insert_resource(AmbientLight {
            color: Color::rgb(1.0, 1.0, 1.0),
            brightness: 0.0,
        })
        .insert_resource(MipmapGeneratorSettings {
            anisotropic_filtering: 16,
            ..default()
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
            MipmapGeneratorPlugin,
            CameraControllerPlugin,
            TAAPlugin,
            SSGIPlugin,
            LogDiagnosticsPlugin::default(),
            FrameTimeDiagnosticsPlugin::default(),
            FilterQueryInspectorPlugin::<With<SSGIPass>>::default()
                .run_if(input_toggle_active(false, KeyCode::Tab)),
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (proc_scene, move_directional_light))
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        SceneBundle {
            scene: asset_server.load("large_models/san-miguel/san-miguel.gltf#Scene0"),
            transform: Transform::from_xyz(-18.0, 0.0, 0.0),
            ..default()
        },
        PostProcScene,
    ));

    commands.spawn(SceneBundle {
        scene: asset_server.load("models/FlightHelmet/FlightHelmet.gltf#Scene0"),
        ..default()
    });

    // Sun
    commands.spawn((
        DirectionalLightBundle {
            transform: Transform::from_rotation(Quat::from_euler(
                EulerRot::XYZ,
                PI * -0.43,
                PI * -0.08,
                0.0,
            )),
            directional_light: DirectionalLight {
                color: Color::rgb_linear(1.0, 0.66, 0.51),
                illuminance: 2300000.0 * 0.2,
                shadows_enabled: true,
                shadow_depth_bias: 0.04,
                shadow_normal_bias: 1.8,
            },
            cascade_shadow_config: CascadeShadowConfigBuilder {
                num_cascades: 4,
                maximum_distance: 30.0,
                ..default()
            }
            .into(),
            ..default()
        },
        GrifLight,
    ));
    // camera
    commands
        .spawn((
            Camera3dBundle {
                camera: Camera {
                    hdr: true,
                    ..default()
                },
                transform: Transform::from_xyz(-10.5, 1.7, -1.0)
                    .looking_at(Vec3::new(0.0, 3.5, 0.0), Vec3::Y),
                projection: Projection::Perspective(PerspectiveProjection {
                    fov: std::f32::consts::PI / 3.0,
                    ..default()
                }),
                color_grading: ColorGrading {
                    exposure: -3.0,
                    ..default()
                },
                ..default()
            },
            CameraController {
                walk_speed: 2.0,
                mouse_key_enable_mouse: MouseButton::Right,
                ..default()
            },
            EnvironmentMapLight {
                diffuse_map: asset_server.load("environment_maps/pisa_diffuse_rgb9e5_zstd.ktx2"),
                specular_map: asset_server.load("environment_maps/pisa_specular_rgb9e5_zstd.ktx2"),
                intensity: 1000.0,
            },
            SSGIBundle {
                ssgi_pass: SSGIPass {
                    brightness: 3.0,
                    ..default()
                },
                ..default()
            },
        ))
        .insert(TAABundle::sample8());
}

pub fn all_children<F: FnMut(Entity)>(
    children: &Children,
    children_query: &Query<&Children>,
    closure: &mut F,
) {
    for child in children {
        if let Ok(children) = children_query.get(*child) {
            all_children(children, children_query, closure);
        }
        closure(*child);
    }
}

#[derive(Component)]
pub struct PostProcScene;

#[derive(Component)]
pub struct GrifLight;

#[allow(clippy::type_complexity)]
pub fn proc_scene(
    mut commands: Commands,
    materials_query: Query<Entity, With<PostProcScene>>,
    children_query: Query<&Children>,
    has_std_mat: Query<&Handle<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    lights: Query<
        Entity,
        (
            Or<(With<PointLight>, With<DirectionalLight>, With<SpotLight>)>,
            Without<GrifLight>,
        ),
    >,
    cameras: Query<Entity, With<Camera>>,
) {
    for entity in materials_query.iter() {
        if let Ok(children) = children_query.get(entity) {
            all_children(children, &children_query, &mut |entity| {
                if let Ok(mat_h) = has_std_mat.get(entity) {
                    if let Some(mat) = materials.get_mut(mat_h) {
                        //mat.flip_normal_map_y = true;
                        //dbg!("!!!");
                        match mat.alpha_mode {
                            AlphaMode::Mask(_) => {
                                mat.diffuse_transmission = 0.6;
                                mat.double_sided = true;
                                mat.cull_mode = None;
                                mat.thickness = 0.2;
                                commands.entity(entity).insert(TransmittedShadowReceiver);
                            }
                            _ => (),
                        }
                    }
                }

                // Remove Default Lights
                if lights.get(entity).is_ok() {
                    commands.entity(entity).despawn_recursive();
                }

                // Remove Default Cameras
                if cameras.get(entity).is_ok() {
                    commands.entity(entity).despawn_recursive();
                }
            });
            commands.entity(entity).remove::<PostProcScene>();
        }
    }
}

fn move_directional_light(
    mut query: Query<&mut Transform, With<DirectionalLight>>,
    mut motion_evr: EventReader<MouseMotion>,
    keys: Res<ButtonInput<KeyCode>>,
    mut e_rot: Local<Vec3>,
) {
    if !keys.pressed(KeyCode::KeyL) {
        return;
    }
    for mut trans in &mut query {
        let euler = trans.rotation.to_euler(EulerRot::XYZ);
        let euler = vec3(euler.0, euler.1, euler.2);

        for ev in motion_evr.read() {
            *e_rot = vec3(
                (euler.x.to_degrees() + ev.delta.y * 2.0).to_radians(),
                (euler.y.to_degrees() + ev.delta.x * 2.0).to_radians(),
                euler.z,
            );
        }
        let store = euler.lerp(*e_rot, 0.2);
        trans.rotation = Quat::from_euler(EulerRot::XYZ, store.x, store.y, store.z);
    }
}
