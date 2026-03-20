# UI Store — Implementation Plan

## Principles

1. **Done = user sees it working.** Compilation and tests passing is necessary
   but not sufficient. Every task ends with a visual verification step.
2. **Incremental migration.** Old and new systems coexist during migration.
   Nothing breaks mid-way. Each commit leaves the editor fully functional.
3. **Code cleanliness.** Small modules, clear responsibilities, self-documenting
   types. No 1000-line files. If a module is getting big, split it before
   proceeding.
4. **One task, one commit.** Each numbered task is a single atomic commit with
   its own verification step.

## File Organization

```
crates/rkf-editor/src/
  store/
    mod.rs              — UiStore struct, public API
    path.rs             — path parsing, interning, routing metadata
    types.rs            — UiValue, type conversion, field metadata
    actions.rs          — Action registration, dispatch, shortcuts
    signals.rs          — lazy signal cache, push batching
    routing.rs          — path → EditorCommand routing logic
  ui/
    widgets/
      mod.rs            — re-exports
      float_slider.rs   — FloatSlider (pure widget, no store knowledge)
      bool_toggle.rs    — BoolToggle
      color_swatch.rs   — ColorSwatch (with onchange-during-render guard)
      vec3_input.rs     — Vec3Input (3 float fields)
      enum_select.rs    — EnumSelect (dropdown)
      action_button.rs  — ActionButton (icon/label, fires action)
    bound/
      mod.rs            — re-exports
      bound_slider.rs   — BoundSlider (connects FloatSlider to store path)
      bound_toggle.rs   — BoundToggle
      bound_color.rs    — BoundColor
      bound_vec3.rs     — BoundVec3
      bound_select.rs   — BoundSelect
    menus/
      mod.rs            — Menu, MenuItem, ContextMenu components
      menu_bar.rs       — top-level menu bar built from action IDs
```

---

## Phase 0: Store Foundation

### 0.1 — UiValue type and conversion

Create `store/types.rs`:

```rust
pub enum UiValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
    Vec3([f64; 3]),
    None,
}
```

Implement conversions:
- `UiValue → GameValue` (for writing to ECS)
- `GameValue → UiValue` (for reading from ECS)
- `UiValue::from_hex(s) → UiValue::Vec3` (color hex → RGB)
- `UiValue::to_hex(&self) → String` (RGB → color hex)

Write unit tests for every conversion path. Test edge cases (NaN, negative
values, invalid hex, out-of-range casts).

**Done:** Tests pass. No UI changes.

### 0.2 — Path parsing and interning

Create `store/path.rs`:

```rust
pub enum PathRoute {
    /// entity:{uuid}/{component}/{field}
    EcsField { entity_id: Uuid, component: String, field: String },
    /// light:{id}/{field}
    LightField { light_id: u64, field: String },
    /// material:{slot}/{field}
    MaterialField { slot: u16, field: String },
    /// camera/{field}
    CameraField { field: String },
    /// editor/{field}
    EditorState { field: String },
    /// sculpt/{field}, paint/{field}
    ToolState { tool: String, field: String },
    /// console/*, drag/*
    SystemPath { path: String },
}
```

Implement `parse_path(s: &str) -> PathRoute`. Intern paths to integer IDs
for fast lookup (HashMap<String, u32> + Vec<PathEntry>).

Write unit tests for path parsing: all route types, malformed paths, UUID
parsing, edge cases.

**Done:** Tests pass. No UI changes.

### 0.3 — Signal cache and push batching

Create `store/signals.rs`:

Lazy signal cache: `HashMap<u32, Signal<UiValue>>`. Signals created on first
`read()`. Engine→UI push buffer: `Arc<Mutex<Vec<(String, UiValue)>>>`.

Drain method called via `run_on_main_thread` — updates corresponding signals.

Write unit tests for:
- Signal creation on first read
- Push updates signal value
- Multiple pushes in one batch
- Reading a path that was never pushed returns UiValue::None

**Done:** Tests pass. No UI changes.

### 0.4 — Write routing

Create `store/routing.rs`:

`fn route_write(path: &PathRoute, value: GameValue) -> Option<EditorCommand>`

Maps parsed paths to EditorCommands:
- EcsField → SetComponentField
- LightField → SetLightPosition / SetLightIntensity / etc.
- CameraField → SetCameraFov / SetCameraSpeed / etc.
- EditorState → SetDebugMode / SetEditorMode / etc.
- ToolState → SetBrushRadius / etc.

Write unit tests for every routing path.

**Done:** Tests pass. No UI changes.

### 0.5 — UiStore struct and public API

Create `store/mod.rs`:

```rust
pub struct UiStore {
    paths: PathRegistry,
    signals: SignalCache,
    push_buffer: Arc<Mutex<Vec<(String, UiValue)>>>,
    cmd_tx: crossbeam::channel::Sender<EditorCommand>,
    actions: ActionRegistry,  // empty for now
}

impl UiStore {
    pub fn read(&self, path: &str) -> Signal<UiValue>;
    pub fn read_float(&self, path: &str) -> f64;
    pub fn read_bool(&self, path: &str) -> bool;
    pub fn read_string(&self, path: &str) -> String;
    pub fn set(&self, path: &str, value: UiValue);
    pub fn push_buffer(&self) -> Arc<Mutex<Vec<(String, UiValue)>>>;
}
```

Create `UiStore` as a rinch context alongside existing `UiSignals`. Both
coexist during migration.

Wire into `main.rs`: create UiStore, `create_context(store)`.
Wire into engine loop: get push buffer handle, drain each frame via
`run_on_main_thread`.

Write integration test: create store, push a value from "engine thread",
verify signal updates on "UI thread".

**Verify:** Editor starts normally. Store exists but nothing uses it yet.
All existing functionality unchanged. Launch editor, open a project, sculpt
an object, save — everything works exactly as before.

---

## Phase 1: Layer 1 Widgets

### 1.1 — FloatSlider widget

Create `ui/widgets/float_slider.rs`:

A pure slider component. Props: `value: f64`, `min`, `max`, `step`,
`decimals`, `label`, `suffix`, `on_change: impl Fn(f64)`.

Must handle the onchange-during-render guard internally (skip callback if
value matches initial).

Styled to match existing `SliderRow` appearance exactly.

**Verify:** Add a test `FloatSlider` to the bottom of the Editor Camera panel
(temporary, hardcoded value). Confirm it renders correctly and matches the
visual style of existing sliders. Remove the test slider.

### 1.2 — BoolToggle widget

Create `ui/widgets/bool_toggle.rs`:

Props: `value: bool`, `label`, `on_change: impl Fn(bool)`.

Styled to match existing `ToggleRow`.

**Verify:** Same approach — temporary test toggle, verify visual match.

### 1.3 — ColorSwatch widget

Create `ui/widgets/color_swatch.rs`:

Props: `value: String` (hex), `label`, `on_change: impl Fn(String)`.

Wraps `ColorPicker` with the onchange-during-render guard built in. Uses
the manual DOM construction pattern (not rsx!) to avoid the closure depth
issues we hit.

**Verify:** Temporary test swatch, verify it renders and doesn't segfault.
Click it, change the color, verify callback fires.

### 1.4 — Vec3Input widget

Create `ui/widgets/vec3_input.rs`:

Props: `value: [f64; 3]`, `labels: [&str; 3]`, `on_change: impl Fn([f64; 3])`.

Three inline FloatSlider or number inputs for X/Y/Z.

**Verify:** Temporary test, verify renders and callbacks work.

### 1.5 — EnumSelect widget

Create `ui/widgets/enum_select.rs`:

Props: `value: String`, `options: &[(String, String)]` (value, label),
`label`, `on_change: impl Fn(String)`.

Wraps rinch `Select`. Handles keyed rebuild for dynamic option lists.

**Verify:** Temporary test, verify renders and selection works.

---

## Phase 2: Layer 2 Bound Widgets

### 2.1 — BoundSlider

Create `ui/bound/bound_slider.rs`:

```rust
#[component]
pub fn BoundSlider(
    path: &'static str,
    label: &str,
    min: f64, max: f64, step: f64, decimals: u32,
    suffix: &str,
) -> NodeHandle
```

Reads `store.read(path)`, renders `FloatSlider`, on change calls
`store.set(path, UiValue::Float(v))`.

**Verify:** Add a `BoundSlider` for `"camera/fov"` alongside the existing
FOV slider. Both should show the same value and both should work when
dragged. Remove the test slider after verification.

### 2.2 — BoundToggle

Same pattern for booleans.

**Verify:** Test alongside existing atmosphere toggle.

### 2.3 — BoundColor

Same pattern for color pickers.

**Verify:** Test alongside existing sun color picker.

### 2.4 — BoundSelect

Same pattern for dropdowns.

**Verify:** Test alongside existing tone map mode selector.

---

## Phase 3: Action System

### 3.1 — Action registry

Create `store/actions.rs`:

```rust
pub struct Action {
    pub id: &'static str,
    pub label: &'static str,
    pub shortcut: Option<&'static str>,
    pub enabled: Option<fn(&UiStore) -> bool>,
    pub checked: Option<fn(&UiStore) -> bool>,
    pub execute: fn(&UiStore),
}
```

`ActionRegistry`: register, lookup by ID, match shortcut string to action.

Write unit tests: register, lookup, duplicate ID detection, shortcut
matching.

**Done:** Tests pass. No UI changes.

### 3.2 — ActionButton widget

Create `ui/widgets/action_button.rs`:

Reads action metadata from store. Renders a button with label, optional
shortcut hint, enabled/disabled state, checked/unchecked state. Calls
`store.execute_action(id)` on click.

**Verify:** Add a temporary ActionButton for "view.show_grid" action next
to the existing grid toggle. Both should work identically.

### 3.3 — Register core editor actions

Register all current toolbar/menu actions:
- Mode switching: `mode.select`, `mode.sculpt`, `mode.paint`
- Gizmo: `gizmo.translate`, `gizmo.rotate`, `gizmo.scale`
- View: `view.show_grid`, `view.debug.*`
- Edit: `edit.undo`, `edit.redo`, `edit.delete`, `edit.duplicate`
- File: `file.save`, `file.save_as`, `file.open`, `file.new_project`

Don't migrate the UI yet — just register them so they're available.

**Done:** Actions registered. Existing UI unchanged. All editor functions
still work.

---

## Phase 4: Migrate Environment Panel

This is the proof-of-concept migration. Replace the current environment
sections with store-bound widgets.

### 4.1 — Register environment field paths

In the store initialization, register all EnvironmentSettings field paths
with their target types:

```
entity:{cam}/EnvironmentSettings/atmosphere.sun_intensity → Float
entity:{cam}/EnvironmentSettings/atmosphere.sun_color → Color
entity:{cam}/EnvironmentSettings/fog.density → Float
entity:{cam}/EnvironmentSettings/fog.enabled → Bool
... (all fields)
```

The `{cam}` is resolved dynamically from `editor/active_camera`.

Wire the engine loop to push all EnvironmentSettings fields to the store
after `apply_environment_settings`.

**Verify:** Add `eprintln!` in the push path. Confirm all fields are being
pushed each frame when dirty. Remove the debug print.

### 4.2 — Replace AtmosphereSection with bound widgets

Replace the manual SliderRow + signal wiring in AtmosphereSection with:

```rust
BoundSlider { path: "env/atmosphere.sun_intensity", label: "Sun Intensity", ... }
BoundColor  { path: "env/atmosphere.sun_color", label: "Sun Color" }
BoundToggle { path: "env/atmosphere.enabled", label: "Atmosphere" }
BoundSlider { path: "env/atmosphere.rayleigh_scale", label: "Rayleigh Scale", ... }
BoundSlider { path: "env/atmosphere.mie_scale", label: "Mie Scale", ... }
```

Note: sun azimuth/elevation are derived (they compute sun_direction). This
is a special case — the bound widget for azimuth/elevation would need a
custom transform. Handle this with a dedicated `SunDirectionControl`
component that internally manages the azimuth/elevation→direction conversion
and writes `env/atmosphere.sun_direction` as a Vec3.

**Verify:** Launch editor. Open project. Atmosphere section should look and
behave identically to before. Drag each slider, confirm the rendering
updates in real-time. Switch to a scene camera and back — values should
persist.

### 4.3 — Replace FogSection with bound widgets

Same approach. Includes the color pickers (BoundColor for fog.color and
fog.vol_ambient_color) and the ambient intensity slider.

**Verify:** Fog sliders work. Color pickers work without segfault. Enable
fog, adjust density, confirm volumetrics update.

### 4.4 — Replace CloudsSection with bound widgets

**Verify:** Cloud sliders work. Enable clouds, adjust coverage, confirm
clouds appear/change.

### 4.5 — Replace PostProcessSection with bound widgets

Includes GI intensity, bloom, exposure, tone map mode (BoundSelect), DoF,
motion blur, god rays, vignette, grain, chromatic aberration.

**Verify:** Every slider works. Toggle bloom on/off. Switch tone map mode.
Enable DoF. Confirm all visual changes match expected behavior.

### 4.6 — Remove old environment signal infrastructure

Now that all environment fields go through the store:
- Remove environment-related entries from `SliderSignals`
- Remove `sync_env_sliders_from_settings()` entries for migrated fields
- Remove `send_atmosphere_commands`, `send_fog_commands`,
  `send_cloud_commands`, `send_post_process_commands`
- Remove `send_env_color`, `send_env_field` and related helpers
- Remove `sun_color`, `fog_color`, `vol_ambient_color` from `UiSignals`
- Remove `environment_profile_name` from `UiSignals` (read from store)

**Verify:** Full editor regression. Open project, adjust every environment
slider, adjust colors, toggle features, save scene, reload — everything
works. No old-system remnants. The environment panel is now fully
store-driven.

---

## Phase 5: Migrate Camera Settings

### 5.1 — Camera fields through store

Register: `camera/fov`, `camera/fly_speed`, `camera/near`, `camera/far`.

Replace CameraSettingsSection SliderRows with BoundSliders.

Remove camera entries from SliderSignals and send_camera_commands.

**Verify:** FOV slider works, fly speed works, near/far work. Fly around
the scene to confirm camera parameters are correct.

---

## Phase 6: Migrate Viewport Toolbar

### 6.1 — Mode buttons as ActionButtons

Replace the manual mode button rendering with:
```rust
ActionButton { action: "mode.select" }
ActionButton { action: "mode.sculpt" }
ActionButton { action: "mode.paint" }
```

**Verify:** Click each mode. Confirm the button highlights correctly and the
editor enters the right mode. Sculpt something, paint something.

### 6.2 — Gizmo buttons as ActionButtons

Replace gizmo mode buttons with ActionButtons.

**Verify:** G/R/L shortcuts work. Gizmo mode buttons highlight correctly.
Translate, rotate, scale an object.

### 6.3 — Camera selector as BoundSelect

Replace the viewport camera selector with a BoundSelect on
`viewport/camera`.

**Verify:** Dropdown shows scene cameras. Selecting one changes the viewport.
Selecting "Editor Camera" returns to free camera.

---

## Phase 7: Migrate System Menu

### 7.1 — Menu bar from action IDs

Replace the hardcoded titlebar menu with a data-driven menu built from
action IDs:

```rust
Menu {
    items: &[
        MenuItem::SubMenu("File", &["file.new_project", "file.open", "---",
            "file.save", "file.save_as", "---", "file.quit"]),
        MenuItem::SubMenu("Edit", &["edit.undo", "edit.redo", "---",
            "edit.delete", "edit.duplicate"]),
        MenuItem::SubMenu("View", &["view.show_grid", "---",
            "view.debug.normals", "view.debug.positions", ...]),
    ]
}
```

**Verify:** Every menu item works. Shortcuts displayed correctly. Disabled
items are grayed out (e.g. Undo when undo stack is empty).

### 7.2 — Keyboard shortcuts through store

Replace the manual shortcut matching in the input handler with
`store.match_shortcut()`. Remove `pending_*` flags that are now handled
by actions.

**Verify:** Ctrl+S saves. Ctrl+Z undoes. Delete deletes. G/R/L switch
gizmo. Every shortcut works.

---

## Phase 8: Migrate Remaining Property Panels

### 8.1 — Light properties

Register light field paths. Replace light property sliders with
BoundSliders. Remove light-specific send commands.

**Verify:** Select a light. Adjust intensity, range, position. Confirm
lighting updates in real-time.

### 8.2 — Material properties

Register material field paths. Replace material property sliders with
BoundSliders and BoundColors.

**Verify:** Select a material. Adjust albedo, roughness, metallic. Confirm
material updates on objects in viewport.

### 8.3 — Component inspector (generic)

Enhance the component inspector to auto-generate bound widgets from
FieldMeta:
- `FieldType::Float` → BoundSlider (with range from FieldMeta.range)
- `FieldType::Bool` → BoundToggle
- `FieldType::Color` → BoundColor
- `FieldType::Vec3` / `WorldPosition` → BoundVec3
- `FieldType::EntityRef` → entity picker
- `FieldType::AssetRef` → asset path picker
- `FieldType::Int` / `Enum` → BoundSelect or number input

**Verify:** Select an entity with a gameplay component. All fields render
with appropriate widgets. Edit a field, confirm it updates in the engine.
Test with multiple component types.

---

## Phase 9: Migrate Collection Views

### 9.1 — Scene tree from store list

Push scene objects as a store list. Scene tree reads from store instead
of `ui.objects` signal.

Selection writes to `editor/selected` store path. Other panels read
selection from store.

**Verify:** Scene tree shows all objects. Click to select. Selection
highlights in tree and viewport. Rename works. Expand/collapse works.

### 9.2 — Materials panel from store list

Push material slots as a store list. Selection through store.

**Verify:** Material grid shows all slots. Click to select. Properties
panel updates.

### 9.3 — Drag-and-drop through store

Replace DragContext signals with store drag/* paths.

**Verify:** Drag material onto object — material applies. Drag shader onto
material — shader assigns. Visual feedback during drag works.

---

## Phase 10: Migrate Console

### 10.1 — Console entries through store

Push console entries via `store.append("console/entries", ...)`.
Filter toggles as bound widgets on `console/filter/*`.
Clear as a registered action.

Replace `ui.console_entries` and `ui.console_filter` signals.

**Verify:** Console shows engine messages. Filter toggles work. Clear
button works. Auto-scroll works. Build errors appear.

---

## Phase 11: Migrate Status and Remaining State

### 11.1 — Status bar from store

Status bar reads: `editor/fps`, `editor/object_count`, `editor/selected`,
`editor/mode`, `camera/position`.

Replace direct UiSignals reads.

**Verify:** Status bar shows correct FPS, position, selection info, mode.

### 11.2 — Loading modal from store

Read `editor/loading_status` from store.

**Verify:** Open a project. Loading modal appears during build. Disappears
when done.

---

## Phase 12: Final Cleanup

### 12.1 — Remove SliderSignals

All slider signals have been migrated to store. Remove the struct, its
construction, its context registration.

**Verify:** Editor compiles and runs. No regressions.

### 12.2 — Remove old UiSignals fields

Remove all fields that moved to store. UiSignals should only contain
truly UI-local state (if any remains — it might be empty enough to
remove entirely).

**Verify:** Editor compiles and runs. No regressions.

### 12.3 — Remove old command helpers

Remove `send_env_*`, `apply_editor_command` routing for anything now
handled by store routing. Remove `pending_*` flags replaced by actions.

**Verify:** Editor compiles and runs. Every feature works.

### 12.4 — Remove old engine→UI push code

Remove `push_dirty_ui_signals`, `sync_env_sliders_from_settings`, and
the per-field signal sets in `engine_loop_ui.rs`. All pushes now go
through `store.push_batch()`.

**Verify:** Full regression test. Open project, manipulate scene, sculpt,
paint, save, load, undo, redo, adjust all environment settings, switch
modes, use menus, use shortcuts. Everything works.

### 12.5 — Documentation update

Update CLAUDE.md to reflect the new UI Store architecture:
- How to add a new editable field (3 steps)
- How to add a new action (1 registration)
- How to add a new panel (use bound widgets)
- Remove references to SliderSignals, UiSignals field lists, send_* functions

**Done:** A new developer (or AI agent) can add a field or action by
reading CLAUDE.md without studying the old boilerplate patterns.

---

## Verification Checklist (Phase 12.4)

After final cleanup, every item must be manually verified:

- [ ] Editor starts without crash
- [ ] Open existing project — scene loads, objects visible
- [ ] Atmosphere: sun azimuth, elevation, intensity, color all work
- [ ] Atmosphere: toggle on/off
- [ ] Fog: density, height falloff, dust, color, ambient, toggle
- [ ] Clouds: coverage, density, altitude, thickness, wind speed, toggle
- [ ] Post-process: GI intensity, bloom, exposure, tone map, sharpen, DoF,
      motion blur, god rays, vignette, grain, chromatic aberration
- [ ] Camera: FOV, fly speed, near, far, position readout
- [ ] Environment source selector (linked camera)
- [ ] Profile name display
- [ ] Select object — properties panel shows component fields
- [ ] Edit Transform via property sliders
- [ ] Edit gameplay component fields via inspector
- [ ] Scene tree — shows objects, click selects, expand/collapse
- [ ] Gizmo — translate, rotate, scale objects
- [ ] G/R/L shortcuts switch gizmo mode
- [ ] Sculpt mode — brush radius, strength, falloff work
- [ ] Sculpt — add/subtract geometry
- [ ] Paint mode — material paint, color paint, erase
- [ ] Material panel — select material, drag onto object
- [ ] Shader panel — select shader, assign to material
- [ ] Light editor — select light, adjust properties
- [ ] File > Save, File > Open, File > New Project
- [ ] Ctrl+Z undo, Ctrl+Shift+Z redo
- [ ] Delete selected object
- [ ] Console — messages appear, filters work, clear works
- [ ] Status bar — FPS, position, selection, mode display
- [ ] Loading modal during project build
- [ ] Viewport camera selector — switch to scene camera and back
- [ ] Environment settings persist in scene save/load
- [ ] Environment profile auto-save to .rkenv
- [ ] MCP screenshot — renders correctly
- [ ] No segfaults, no panics, no visual artifacts

---

## Estimated Scope

- Phase 0 (Store Foundation): 5 tasks — infrastructure only, no UI risk
- Phase 1 (Layer 1 Widgets): 5 tasks — new components, no migration risk
- Phase 2 (Layer 2 Bound Widgets): 4 tasks — new components, no migration risk
- Phase 3 (Action System): 3 tasks — new infrastructure + action registration
- Phase 4 (Environment Panel): 6 tasks — first real migration, highest risk
- Phase 5 (Camera Settings): 1 task — small migration
- Phase 6 (Viewport Toolbar): 3 tasks — mode/gizmo migration
- Phase 7 (System Menu): 2 tasks — menu migration
- Phase 8 (Property Panels): 3 tasks — remaining panel migrations
- Phase 9 (Collection Views): 3 tasks — scene tree, materials, drag-drop
- Phase 10 (Console): 1 task
- Phase 11 (Status/Loading): 2 tasks
- Phase 12 (Cleanup): 5 tasks — remove old code, verify everything

**Total: 43 tasks across 13 phases.**

The highest-risk phase is Phase 4 (environment migration) because it's the
first to actually replace working code. Everything before it is additive.
Everything after it follows the proven pattern.
