# UI Store Architecture

## Problem

Adding an editable property to the editor currently requires touching 6+ files:
struct field, component reflection, signal definition, signal constructor,
engine→UI push function, UI→engine send function, and the panel component.
Forgetting any step causes silent bugs or crashes.

The same boilerplate applies to every data source: ECS components, lights,
materials, camera properties, editor settings. User-defined gameplay
components will need the same wiring, and users can't be expected to do it.

## Design

### Three Layers

```
Layer 3: Property Sheets
    Auto-generated panels from field metadata.
    "Show me the editable fields for this component."

Layer 2: Bound Widgets
    Connect a widget to a store path.
    BoundSlider { path: "entity:{id}/Fog/density", label: "Density", ... }

Layer 1: Typed Widgets
    Pure UI components. Take a value, render it, call back on change.
    FloatSlider, ColorSwatch, BoolToggle, Vec3Input, EnumSelect, ...
    No knowledge of store, ECS, or commands.
```

### UI Store

A single reactive key-value store that bridges the UI thread and the engine
thread. All editor-visible state flows through it.

#### Paths

Every piece of state has a string path:

```
# ECS component fields
entity:{uuid}/{ComponentName}/{field.path}
entity:a1b2c3/EnvironmentSettings/fog.density
entity:a1b2c3/Transform/position

# Non-ECS editor state
light:{id}/{field}
light:3/intensity
light:3/color

material:{slot}/{field}
material:5/roughness
material:5/albedo

camera/{field}
camera/fov
camera/fly_speed

editor/{field}
editor/mode
editor/selected
editor/show_grid
```

#### Values

The store uses a small set of raw UI value types:

```rust
enum UiValue {
    Float(f64),      // sliders, number inputs
    Int(i64),        // enum selects, counters
    Bool(bool),      // toggles
    String(String),  // text inputs, color hex, asset paths
    Vec3([f64; 3]),  // position/rotation/scale inputs
}
```

These are the types that UI widgets naturally produce. They are NOT the
engine's typed values — the store handles conversion.

#### Type Conversion

Each registered path has a target type. The store converts:

| UI sends        | Target type  | Conversion              |
|-----------------|-------------|-------------------------|
| `Float(f64)`    | `f32`       | `v as f32`             |
| `Float(f64)`    | `u32`       | `v as u32`             |
| `String(hex)`   | `[f32; 3]`  | parse hex → RGB floats |
| `Vec3([f64;3])` | `Vec3`      | component-wise `as f32`|
| `Bool(b)`       | `bool`      | passthrough            |
| `Int(i64)`      | `u32`       | `v as u32`             |

Conversion happens in ONE place: the store's `set()` method. The UI never
constructs `GameValue`, never does `as f32`, never parses hex colors. Widgets
send raw values, the store converts and routes.

#### Reads (Engine → UI)

The engine thread pushes state into the store after each frame:

```rust
// Engine thread (inside editor_state lock):
store.push_batch(&[
    ("entity:{cam}/EnvironmentSettings/fog.density", UiValue::Float(0.02)),
    ("entity:{cam}/EnvironmentSettings/atmosphere.sun_intensity", UiValue::Float(3.0)),
    ("light:0/intensity", UiValue::Float(5.0)),
    ("camera/fov", UiValue::Float(70.0)),
    ...
]);
```

Each path maps to a `Signal<UiValue>` on the UI thread. The push updates the
signal, triggering reactive re-renders of any bound widgets. Signals are
created lazily on first read.

The engine only pushes paths that changed (dirty tracking). This replaces
`sync_env_sliders_from_settings()`, `push_dirty_ui_signals()`, and all the
manual signal sets in `engine_loop_ui.rs`.

#### Writes (UI → Engine)

```rust
// UI thread (from widget callback):
store.set("entity:{cam}/EnvironmentSettings/fog.density", UiValue::Float(0.05));
```

The store:
1. Converts `UiValue::Float(0.05)` → `GameValue::Float(0.05)` (target is f32)
2. Parses the path to determine routing
3. Sends the appropriate `EditorCommand`:
   - `entity:*/Component/field` → `SetComponentField { entity_id, component, field, value }`
   - `light:*/field` → `SetLightField { light_id, field, value }`
   - `material:*/field` → `SetMaterial { slot, ... }`
   - `camera/*` → camera commands
4. Updates the local signal immediately for responsive UI

This replaces all `send_*_commands()` functions, `send_env_field()`,
`send_env_color()`, etc.

### Widget Layer (Layer 1)

Pure UI components with no store knowledge:

```rust
#[component]
fn FloatSlider(
    value: f64,
    min: f64, max: f64, step: f64, decimals: u32,
    label: &str, suffix: &str,
    on_change: impl Fn(f64),
) -> NodeHandle { ... }

#[component]
fn ColorSwatch(
    value: String,  // hex
    label: &str,
    on_change: impl Fn(String),
) -> NodeHandle { ... }

#[component]
fn BoolToggle(
    value: bool,
    label: &str,
    on_change: impl Fn(bool),
) -> NodeHandle { ... }
```

These replace `SliderRow`, `ToggleRow`, and the manual `ColorPicker` wiring.
They handle the onchange-during-render guard internally.

### Bound Widget Layer (Layer 2)

Connect a widget to a store path:

```rust
#[component]
fn BoundSlider(
    path: &'static str,
    label: &str,
    min: f64, max: f64, step: f64, decimals: u32,
    suffix: &str,
) -> NodeHandle {
    let store = use_context::<UiStore>();
    let value = store.read_float(path);  // returns Signal<f64>

    FloatSlider {
        value: value.get(),
        min, max, step, decimals, label, suffix,
        on_change: move |v| { store.set(path, UiValue::Float(v)); },
    }
}

#[component]
fn BoundColor(path: &'static str, label: &str) -> NodeHandle {
    let store = use_context::<UiStore>();
    let value = store.read_string(path);  // returns hex string signal

    ColorSwatch {
        value: value.get(),
        label,
        on_change: move |hex| { store.set(path, UiValue::String(hex)); },
    }
}
```

### Property Sheet Layer (Layer 3)

Auto-generate a panel from field metadata:

```rust
#[component]
fn ComponentInspector(entity_id: Uuid, component_name: &str) -> NodeHandle {
    let registry = use_context::<GameplayRegistry>();
    let entry = registry.component_entry(component_name);

    // For each field in entry.meta, render the appropriate bound widget
    // based on field_type:
    //   Float → BoundSlider (using field.range for min/max)
    //   Bool → BoundToggle
    //   Color → BoundColor
    //   Vec3 → BoundVec3
    //   EntityRef → BoundEntityRef
    //   AssetRef → BoundAssetRef

    let path_prefix = format!("entity:{entity_id}/{component_name}");
    // ... render widgets
}
```

This replaces the current text-only component inspector AND the dedicated
environment panel. EnvironmentSettings would render as a nice typed inspector
automatically because its FieldMeta entries specify types and ranges.

### Actions (Menus, Context Menus, Shortcuts)

Not everything is a data binding. Menu items ("Save Scene", "Undo", "Delete
Selected"), context menu actions ("Duplicate", "Focus Camera"), and keyboard
shortcuts are stateless commands. The store handles these too:

```rust
// Register an action with metadata
store.register_action(Action {
    id: "scene.save",
    label: "Save Scene",
    shortcut: Some("Ctrl+S"),
    enabled: |store| store.read_bool("editor/has_unsaved_changes"),
    execute: |store| store.dispatch(EditorCommand::SaveScene),
});

store.register_action(Action {
    id: "edit.undo",
    label: "Undo",
    shortcut: Some("Ctrl+Z"),
    enabled: |store| store.read_bool("editor/can_undo"),
    execute: |store| store.dispatch(EditorCommand::Undo),
});

store.register_action(Action {
    id: "object.delete",
    label: "Delete",
    shortcut: Some("Delete"),
    enabled: |store| store.read("editor/selected").is_some(),
    execute: |store| store.dispatch(EditorCommand::DeleteSelected),
});
```

#### Menu construction

Menus are built from action IDs. The store provides the label, enabled state,
and shortcut text. The menu component doesn't know what the actions do:

```rust
Menu {
    items: &[
        MenuItem::Action("scene.save"),
        MenuItem::Action("scene.save_as"),
        MenuItem::Separator,
        MenuItem::Action("scene.open"),
        MenuItem::Action("scene.new"),
        MenuItem::Separator,
        MenuItem::SubMenu("Recent", &recent_items),
    ]
}
```

#### Context menus

Same mechanism. Right-click on a scene tree node:

```rust
ContextMenu {
    items: &[
        MenuItem::Action("object.duplicate"),
        MenuItem::Action("object.delete"),
        MenuItem::Separator,
        MenuItem::Action("object.focus_camera"),
        MenuItem::Action("object.rename"),
    ]
}
```

The `enabled` callback on each action determines whether the menu item is
grayed out. The store provides the reactive state needed to evaluate it.

#### Keyboard shortcuts

The store owns the shortcut→action mapping. Input handling queries the store:

```rust
// In input handling:
if let Some(action_id) = store.match_shortcut(key, modifiers) {
    if store.is_action_enabled(action_id) {
        store.execute_action(action_id);
    }
}
```

This replaces the current scattered `pending_*` flags on EditorState
(pending_save, pending_undo, pending_delete, etc.) and the manual shortcut
matching in the input handler. Adding a new action = register it once with
its shortcut, enabled condition, and command.

#### Stateful menu items

Some menu items reflect state (checkmarks, radio buttons):

```rust
store.register_action(Action {
    id: "view.show_grid",
    label: "Show Grid",
    shortcut: Some("G"),
    checked: Some(|store| store.read_bool("editor/show_grid")),
    execute: |store| store.set("editor/show_grid", UiValue::Bool(!store.read_bool_val("editor/show_grid"))),
});

store.register_action(Action {
    id: "debug.mode.normals",
    label: "Normals",
    checked: Some(|store| store.read_int_val("editor/debug_mode") == 1),
    execute: |store| store.set("editor/debug_mode", UiValue::Int(1)),
});
```

### What This Replaces

| Current                          | New                              |
|----------------------------------|----------------------------------|
| `SliderSignals` (30+ signals)   | Store paths (lazy signals)       |
| `UiSignals` env fields          | Store paths                      |
| `sync_env_sliders_from_settings`| `store.push_batch()`             |
| `send_*_commands()` functions   | `store.set()` routing            |
| `send_env_color()`              | `store.set()` with hex→RGB conv  |
| Per-field `SliderRow` boilerplate| `BoundSlider` with path          |
| Manual `ColorPicker` wiring     | `BoundColor` with path           |
| Dedicated EnvironmentPanel      | `ComponentInspector` auto-layout |
| `apply_component_command` routing| Store routing (parse path)       |
| `pending_*` flags on EditorState | `store.dispatch()` actions      |
| Manual shortcut matching         | `store.match_shortcut()`        |
| Hardcoded menu item lists        | `store.register_action()` + IDs |
| Menu enabled/disabled logic      | Action `enabled` callbacks      |

### What Stays

- `EnvironmentSettings` struct and its fields (data model)
- `engine_components.rs` reflection (get_field/set_field) — still needed for
  the store to read/write ECS data. Could eventually be derived.
- `apply_environment_settings()` — reads ECS, writes GPU uniforms. Unchanged.
- `EditorCommand` enum — the store sends commands, but the enum still exists
  as the transport mechanism.
- `EditorState` — still owns the world, light editor, etc. The store doesn't
  replace the engine-side state, only the UI-side wiring.

### Adding a New Field (After This Refactor)

1. Add field to the struct (e.g. `FogSettings.vol_ambient_intensity: f32`)
2. Add get_field/set_field in `engine_components.rs`
3. Add to `apply_environment_settings()` if it affects GPU state
4. Add `BoundSlider` to the panel (one line)

Step 4 could be automatic if using `ComponentInspector` with good FieldMeta.

### Migration Path

1. Build `UiStore` core: path registry, type conversion, get/set, dispatch
2. Build action system: register_action, match_shortcut, execute
3. Build Layer 1 widgets (FloatSlider, ColorSwatch, BoolToggle)
4. Build Layer 2 bound widgets (BoundSlider, BoundColor, BoundToggle)
5. Migrate environment panel to bound widgets (proof of concept)
6. Migrate system menu to action-based (File, Edit, View menus)
7. Migrate keyboard shortcuts to store.match_shortcut
8. Migrate light editor, material editor, camera settings
9. Build Layer 3 ComponentInspector (auto-layout from FieldMeta)
10. Remove old SliderSignals, sync functions, send functions, pending_* flags
11. Extend FieldMeta with range hints for auto-generated sliders
12. Context menus via action IDs

### Thread Safety

The store lives on the UI thread (all signals are thread-local). The engine
thread communicates via:

- **Push**: `Arc<Mutex<Vec<(String, UiValue)>>>` staging buffer. Engine writes,
  UI drains via `run_on_main_thread`. Same pattern as current signal pushes.
- **Write**: `store.set()` sends an `EditorCommand` through the existing
  `crossbeam::channel`. No new threading mechanism needed.

### Editor Modes and Tools

The editor has interaction modes (Select, Sculpt, Paint) with mode-specific
tools and settings. These all map naturally to store paths and actions.

#### Mode switching

The current mode is store state. Switching is an action:

```
editor/mode                     → "select" | "sculpt" | "paint"
```

```rust
store.register_action(Action {
    id: "mode.select",
    label: "Select",
    shortcut: Some("1"),
    checked: Some(|s| s.read_string_val("editor/mode") == "select"),
    execute: |s| s.set("editor/mode", UiValue::String("select".into())),
});
```

The viewport toolbar renders mode buttons from action IDs. Each button
reads `editor/mode` for its active state. No special mode-switching logic
in the toolbar component.

#### Gizmo

Gizmo mode and interaction are store state + actions:

```
gizmo/mode                      → "translate" | "rotate" | "scale"
gizmo/space                     → "local" | "world"
gizmo/snap_enabled              → bool
gizmo/snap_value                → f64
```

Shortcuts: G → `gizmo.translate`, R → `gizmo.rotate`, L → `gizmo.scale`.
These are actions that set `gizmo/mode`. The gizmo renderer reads the mode
from the store.

Transform writes from gizmo drag go through:
```
store.set("entity:{id}/Transform/position", UiValue::Vec3(...))
```

#### Sculpt mode

Sculpt has tool settings that map to store paths:

```
sculpt/brush_type               → "add" | "subtract" | "smooth" | "flatten"
sculpt/radius                   → f64
sculpt/strength                 → f64
sculpt/falloff                  → f64
```

The brush palette becomes bound widgets:
```rust
BoundSlider  { path: "sculpt/radius",   label: "Radius",   min: 0.1, max: 50.0, ... }
BoundSlider  { path: "sculpt/strength", label: "Strength", min: 0.0, max: 1.0,  ... }
BoundSlider  { path: "sculpt/falloff",  label: "Falloff",  min: 0.0, max: 1.0,  ... }
```

Brush type switching:
```rust
store.register_action(Action {
    id: "sculpt.brush.add",
    label: "Add",
    checked: Some(|s| s.read_string_val("sculpt/brush_type") == "add"),
    execute: |s| s.set("sculpt/brush_type", UiValue::String("add".into())),
});
```

The actual sculpt operation (mouse drag → geometry edit) stays in the engine
loop. The store manages only the tool settings UI, not the stroke lifecycle.

#### Paint mode

Same pattern as sculpt:

```
paint/mode                      → "material" | "color" | "erase"
paint/radius                    → f64
paint/falloff                   → f64
paint/material_slot             → i64
paint/color                     → String (hex)
```

The paint color picker becomes:
```rust
BoundColor { path: "paint/color", label: "Paint Color" }
```

#### Viewport toolbar layout

The viewport toolbar is built entirely from action IDs and store paths:

```rust
#[component]
fn ViewportToolbar() -> NodeHandle {
    // Mode buttons — each is an action with a checked state
    ActionButton { action: "mode.select" }
    ActionButton { action: "mode.sculpt" }
    ActionButton { action: "mode.paint" }
    Separator {}
    // Gizmo buttons — shown when mode is "select"
    ActionButton { action: "gizmo.translate" }
    ActionButton { action: "gizmo.rotate" }
    ActionButton { action: "gizmo.scale" }
    Separator {}
    // Camera selector — a bound widget
    BoundSelect { path: "viewport/camera", ... }
}
```

No mode-specific logic in the toolbar. It just renders actions. The `checked`
callbacks handle active-state highlighting. Conditional visibility (e.g. hide
gizmo buttons in sculpt mode) can use the store:

```rust
ActionButton { action: "gizmo.translate", visible: store.read_string("editor/mode") == "select" }
```

#### Conditional panels

Mode-specific panels (brush palette, paint settings) show/hide based on
`editor/mode`. The layout system queries the store:

```rust
if store.read_string_val("editor/mode") == "sculpt" {
    BrushPalette {}
}
```

Or the panel itself handles it:
```rust
#[component]
fn BrushPalette() -> NodeHandle {
    let store = use_context::<UiStore>();
    let mode = store.read_string("editor/mode");
    if mode.get() != "sculpt" && mode.get() != "paint" {
        return empty_node();
    }
    // ... render brush settings
}
```

### Open Questions

1. **Path format**: Should paths be strings or typed keys? Strings are flexible
   and debuggable but have runtime parsing cost. Typed keys are faster but less
   flexible. Recommendation: strings for now, optimize later if profiling shows
   it matters.

2. **Dirty tracking granularity**: Push all changed fields per frame, or push
   entire component snapshots? Per-field is more efficient but requires diffing.
   Component snapshots are simpler. Recommendation: start with component
   snapshots (serialize + compare), optimize to per-field if needed.

3. **Custom panels**: Some panels (scene tree, asset browser) aren't property
   editors. They still benefit from the store for state (selection, filter text)
   but don't use bound widgets. The store should support arbitrary paths, not
   just component fields.

4. **Undo integration**: Currently undo captures transform snapshots. With the
   store routing all writes, undo could be implemented at the store level —
   every `set()` call automatically creates an undo entry. This is a future
   enhancement, not required for v1.
