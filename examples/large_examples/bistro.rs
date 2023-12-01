use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin},
    input::{common_conditions::input_toggle_active, mouse::MouseMotion},
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
            scene: asset_server.load("large_models/bistro/BistroExterior_metal_fix.gltf#Scene0"),
            transform: Transform::from_xyz(-18.0, 0.0, 0.0),
            ..default()
        },
        PostProcScene,
    ));

    // Sun
    commands.spawn((
        DirectionalLightBundle {
            transform: Transform::from_rotation(Quat::from_euler(
                EulerRot::XYZ,
                -0.9159545,
                -0.026048705,
                0.0,
            )),
            directional_light: DirectionalLight {
                color: Color::rgb_linear(1.0, 0.66, 0.51),
                illuminance: 200000.0,
                shadows_enabled: true,
                shadow_depth_bias: 0.04,
                shadow_normal_bias: 1.8,
            },
            cascade_shadow_config: CascadeShadowConfigBuilder {
                num_cascades: 4,
                maximum_distance: 100.0,
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
                transform: Transform::from_xyz(-2.164, 5.034, -41.973)
                    .with_rotation(Quat::from_euler(EulerRot::XYZ, 3.014, 0.630, -3.066)),
                projection: Projection::Perspective(PerspectiveProjection {
                    fov: std::f32::consts::PI / 4.0,
                    ..default()
                }),
                tonemapping: Tonemapping::AgX,
                color_grading: ColorGrading {
                    exposure: -1.0,
                    ..default()
                },
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
    names: Query<&Name>,
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
                        mat.flip_normal_map_y = true;
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
                        if let Ok(name) = names.get(entity) {
                            match name.as_str() {
                                // Some of the red awnings
                                "Mesh.1190" | "Mesh.1191" | "Mesh.1192" | "Mesh.1193"
                                | "Mesh.1194" | "Mesh.1195" | "Mesh.1196" => {
                                    mat.diffuse_transmission = 0.15;
                                    mat.double_sided = true;
                                    mat.cull_mode = None;
                                    mat.thickness = 0.2;
                                    commands.entity(entity).insert(TransmittedShadowReceiver);
                                }
                                _ => (),
                            }
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
    keys: Res<Input<KeyCode>>,
    mut e_rot: Local<Vec3>,
) {
    if !keys.pressed(KeyCode::L) {
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
        dbg!(store.x, store.y, store.z);
        trans.rotation = Quat::from_euler(EulerRot::XYZ, store.x, store.y, store.z);
    }
}
