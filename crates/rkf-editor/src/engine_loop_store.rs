//! Engine loop → UI Store push helpers.
//!
//! Pushes environment settings, lights, and editor state to the UI Store's
//! push buffer so that bound widgets can read them reactively.

use crate::light_editor::SceneLight;
use crate::store::signals::PushBuffer;
use crate::store::types::UiValue;
use rkf_runtime::environment::EnvironmentSettings;

/// Camera values extracted from EditorState for store push.
pub(crate) struct CameraStoreValues {
    pub fov_degrees: f32,
    pub fly_speed: f32,
    pub near: f32,
    pub far: f32,
}

/// Push camera settings to the store push buffer.
///
/// Called from the engine loop every frame so that bound widgets stay in sync.
pub(crate) fn push_camera_to_store(buffer: &PushBuffer, cam: &CameraStoreValues) {
    let mut buf = buffer.lock().expect("store push buffer poisoned");
    buf.push(("camera/fov".into(), UiValue::Float(cam.fov_degrees as f64)));
    buf.push(("camera/fly_speed".into(), UiValue::Float(cam.fly_speed as f64)));
    buf.push(("camera/near".into(), UiValue::Float(cam.near as f64)));
    buf.push(("camera/far".into(), UiValue::Float(cam.far as f64)));
}

/// Push all EnvironmentSettings fields to the store push buffer.
///
/// Called from the engine loop after `apply_environment_settings()` when
/// dirty.scene is true.
pub(crate) fn push_environment_to_store(
    buffer: &PushBuffer,
    env: &EnvironmentSettings,
    active_camera_uuid: Option<uuid::Uuid>,
) {
    // Set active camera on main thread for env/ path resolution.
    let cam_uuid = active_camera_uuid;
    rinch::shell::rinch_runtime::run_on_main_thread(move || {
        if let Some(store) = rinch::core::context::try_use_context::<crate::store::UiStore>() {
            store.set_active_camera(cam_uuid);
        }
    });

    let mut buf = buffer.lock().expect("store push buffer poisoned");

    // Atmosphere
    let a = &env.atmosphere;
    buf.push(("env/atmosphere.enabled".into(), UiValue::Bool(a.enabled)));
    buf.push(("env/atmosphere.sun_intensity".into(), UiValue::Float(a.sun_intensity as f64)));
    buf.push(("env/atmosphere.rayleigh_scale".into(), UiValue::Float(a.rayleigh_scale as f64)));
    buf.push(("env/atmosphere.mie_scale".into(), UiValue::Float(a.mie_scale as f64)));
    buf.push(("env/atmosphere.sun_color".into(), UiValue::String(
        crate::store::types::rgb_to_hex([a.sun_color[0] as f64, a.sun_color[1] as f64, a.sun_color[2] as f64])
    )));
    buf.push(("env/atmosphere.sun_direction".into(), UiValue::Vec3([
        a.sun_direction[0] as f64, a.sun_direction[1] as f64, a.sun_direction[2] as f64,
    ])));

    // Fog
    let f = &env.fog;
    buf.push(("env/fog.enabled".into(), UiValue::Bool(f.enabled)));
    buf.push(("env/fog.density".into(), UiValue::Float(f.density as f64)));
    buf.push(("env/fog.height_falloff".into(), UiValue::Float(f.height_falloff as f64)));
    buf.push(("env/fog.ambient_dust_density".into(), UiValue::Float(f.ambient_dust_density as f64)));
    buf.push(("env/fog.dust_asymmetry".into(), UiValue::Float(f.dust_asymmetry as f64)));
    buf.push(("env/fog.color".into(), UiValue::String(
        crate::store::types::rgb_to_hex([f.color[0] as f64, f.color[1] as f64, f.color[2] as f64])
    )));
    buf.push(("env/fog.vol_ambient_color".into(), UiValue::String(
        crate::store::types::rgb_to_hex([f.vol_ambient_color[0] as f64, f.vol_ambient_color[1] as f64, f.vol_ambient_color[2] as f64])
    )));
    buf.push(("env/fog.vol_ambient_intensity".into(), UiValue::Float(f.vol_ambient_intensity as f64)));

    // Clouds
    let c = &env.clouds;
    buf.push(("env/clouds.enabled".into(), UiValue::Bool(c.enabled)));
    buf.push(("env/clouds.coverage".into(), UiValue::Float(c.coverage as f64)));
    buf.push(("env/clouds.density".into(), UiValue::Float(c.density as f64)));
    buf.push(("env/clouds.altitude".into(), UiValue::Float(c.altitude as f64)));
    buf.push(("env/clouds.thickness".into(), UiValue::Float(c.thickness as f64)));
    buf.push(("env/clouds.wind_speed".into(), UiValue::Float(c.wind_speed as f64)));

    // Post-process
    let p = &env.post_process;
    buf.push(("env/post_process.gi_intensity".into(), UiValue::Float(p.gi_intensity as f64)));
    buf.push(("env/post_process.bloom_enabled".into(), UiValue::Bool(p.bloom_enabled)));
    buf.push(("env/post_process.bloom_intensity".into(), UiValue::Float(p.bloom_intensity as f64)));
    buf.push(("env/post_process.bloom_threshold".into(), UiValue::Float(p.bloom_threshold as f64)));
    buf.push(("env/post_process.exposure".into(), UiValue::Float(p.exposure as f64)));
    buf.push(("env/post_process.tone_map_mode".into(), UiValue::Int(p.tone_map_mode as i64)));
    buf.push(("env/post_process.sharpen_strength".into(), UiValue::Float(p.sharpen_strength as f64)));
    buf.push(("env/post_process.dof_enabled".into(), UiValue::Bool(p.dof_enabled)));
    buf.push(("env/post_process.dof_focus_distance".into(), UiValue::Float(p.dof_focus_distance as f64)));
    buf.push(("env/post_process.dof_focus_range".into(), UiValue::Float(p.dof_focus_range as f64)));
    buf.push(("env/post_process.dof_max_coc".into(), UiValue::Float(p.dof_max_coc as f64)));
    buf.push(("env/post_process.motion_blur_intensity".into(), UiValue::Float(p.motion_blur_intensity as f64)));
    buf.push(("env/post_process.god_rays_intensity".into(), UiValue::Float(p.god_rays_intensity as f64)));
    buf.push(("env/post_process.vignette_intensity".into(), UiValue::Float(p.vignette_intensity as f64)));
    buf.push(("env/post_process.grain_intensity".into(), UiValue::Float(p.grain_intensity as f64)));
    buf.push(("env/post_process.chromatic_aberration".into(), UiValue::Float(p.chromatic_aberration as f64)));
}

/// Push editor mode and gizmo mode to the store push buffer.
///
/// Called from the engine loop every frame so that action `checked` callbacks
/// can read the current mode.
pub(crate) fn push_modes_to_store(
    buffer: &PushBuffer,
    editor_mode: crate::editor_state::EditorMode,
    gizmo_mode: crate::gizmo::GizmoMode,
) {
    let mode_str = match editor_mode {
        crate::editor_state::EditorMode::Default => "default",
        crate::editor_state::EditorMode::Sculpt => "sculpt",
        crate::editor_state::EditorMode::Paint => "paint",
    };
    let gizmo_str = match gizmo_mode {
        crate::gizmo::GizmoMode::Translate => "translate",
        crate::gizmo::GizmoMode::Rotate => "rotate",
        crate::gizmo::GizmoMode::Scale => "scale",
    };
    let mut buf = buffer.lock().expect("store push buffer poisoned");
    buf.push(("editor/mode".into(), UiValue::String(mode_str.into())));
    buf.push(("gizmo/mode".into(), UiValue::String(gizmo_str.into())));
}

/// Push debug mode and grid visibility to the store push buffer.
///
/// Called from the engine loop every frame alongside `push_modes_to_store`.
pub(crate) fn push_debug_and_grid_to_store(
    buffer: &PushBuffer,
    debug_mode: u32,
    show_grid: bool,
) {
    let mut buf = buffer.lock().expect("store push buffer poisoned");
    buf.push(("editor/debug_mode".into(), UiValue::Int(debug_mode as i64)));
    buf.push(("editor/show_grid".into(), UiValue::Bool(show_grid)));
}

/// Push scene collection counts and selection to the store push buffer.
///
/// Called from the engine loop when `dirty.scene` or `dirty.lights` is true.
/// Pushes scalar/string summaries that store-bound widgets can consume.
///
/// NOTE: The full collection data (Vec<ObjectSummary>, Vec<MaterialSummary>,
/// Vec<LightSummary>) remains on UiSignals because UiValue doesn't support
/// complex typed lists. A future `UiValue::List` or typed-signal extension
/// would be needed to migrate them fully. Selection also stays on UiSignals
/// as the primary source of truth because it's tightly coupled to tree sync,
/// inspector data push, and material usage computation. The store path
/// `editor/selected` is a read-only mirror for widgets that only need to
/// know *what* is selected (e.g. status bar, action enabled-checks).
pub(crate) fn push_collection_counts_to_store(
    buffer: &PushBuffer,
    object_count: usize,
    light_count: usize,
    material_count: usize,
    selected: Option<&crate::editor_state::SelectedEntity>,
) {
    let mut buf = buffer.lock().expect("store push buffer poisoned");
    buf.push(("editor/object_count".into(), UiValue::Int(object_count as i64)));
    buf.push(("editor/light_count".into(), UiValue::Int(light_count as i64)));
    buf.push(("editor/material_count".into(), UiValue::Int(material_count as i64)));

    let selected_str = match selected {
        Some(crate::editor_state::SelectedEntity::Object(id)) => format!("object:{id}"),
        Some(crate::editor_state::SelectedEntity::Light(id)) => format!("light:{id}"),
        Some(crate::editor_state::SelectedEntity::Scene) => "scene".to_string(),
        Some(crate::editor_state::SelectedEntity::Project) => "project".to_string(),
        None => String::new(),
    };
    buf.push(("editor/selected".into(), UiValue::String(selected_str)));
}

/// Push viewport camera UUID to the store push buffer.
///
/// Called from the engine loop every frame so that the camera selector stays in sync.
pub(crate) fn push_viewport_camera_to_store(buffer: &PushBuffer, camera: Option<uuid::Uuid>) {
    let mut buf = buffer.lock().expect("store push buffer poisoned");
    let value = match camera {
        Some(id) => UiValue::String(id.to_string()),
        None => UiValue::String(String::new()),
    };
    buf.push(("viewport/camera".into(), value));
}

/// Push FPS (frame time in ms) to the store push buffer.
///
/// Called from the engine loop every ~500ms.
pub(crate) fn push_fps_to_store(buffer: &PushBuffer, fps_ms: f64) {
    let mut buf = buffer.lock().expect("store push buffer poisoned");
    buf.push(("editor/fps".into(), UiValue::Float(fps_ms)));
}

/// Push camera display position to the store push buffer.
///
/// Called from the engine loop every ~250ms.
pub(crate) fn push_camera_position_to_store(buffer: &PushBuffer, pos: glam::Vec3) {
    let mut buf = buffer.lock().expect("store push buffer poisoned");
    buf.push(("camera/position".into(), UiValue::Vec3([
        pos.x as f64, pos.y as f64, pos.z as f64,
    ])));
}

/// Push loading status to the store push buffer.
///
/// Called when loading status changes. `None` means idle, `Some(msg)` shows a
/// loading indicator.
pub(crate) fn push_loading_status_to_store(buffer: &PushBuffer, msg: Option<String>) {
    let mut buf = buffer.lock().expect("store push buffer poisoned");
    let value = match msg {
        Some(s) => UiValue::String(s),
        None => UiValue::None,
    };
    buf.push(("editor/loading_status".into(), value));
}

/// Push brush (sculpt/paint shared) settings to the store push buffer.
///
/// Called from the engine loop every frame so that bound widgets stay in sync.
pub(crate) fn push_brush_to_store(
    buffer: &PushBuffer,
    radius: f32,
    strength: f32,
    falloff: f32,
) {
    let mut buf = buffer.lock().expect("store push buffer poisoned");
    buf.push(("sculpt/radius".into(), UiValue::Float(radius as f64)));
    buf.push(("sculpt/strength".into(), UiValue::Float(strength as f64)));
    buf.push(("sculpt/falloff".into(), UiValue::Float(falloff as f64)));
}

/// Push all light fields to the store push buffer.
///
/// Called from the engine loop when `dirty.lights` is true.
pub(crate) fn push_lights_to_store(buffer: &PushBuffer, lights: &[SceneLight]) {
    let mut buf = buffer.lock().expect("store push buffer poisoned");
    for light in lights {
        let id = light.id;
        buf.push((format!("light:{id}/position.x"), UiValue::Float(light.position.x as f64)));
        buf.push((format!("light:{id}/position.y"), UiValue::Float(light.position.y as f64)));
        buf.push((format!("light:{id}/position.z"), UiValue::Float(light.position.z as f64)));
        buf.push((format!("light:{id}/intensity"), UiValue::Float(light.intensity as f64)));
        buf.push((format!("light:{id}/range"), UiValue::Float(light.range as f64)));
    }
}
