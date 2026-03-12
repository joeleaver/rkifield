# Behavior System Architecture

## Motivation

The engine currently has no user-facing system for adding gameplay logic to entities. All behavior is hardcoded in Rust within the engine binaries. A user who wants an enemy to patrol, a door to open, or a pickup to spin must modify engine source code directly.

Games need a way to define behaviors — patrol routes, AI decisions, physics responses, timed sequences — and attach them to entities in the editor. The system must support rapid iteration: edit code, save, see the result without restarting the editor.

## Core Principles

1. **Components are data. Systems are behavior.** An entity's behavior is determined by which components it has. Systems are standalone functions that process all entities matching a component query. Behavior is never attached to an individual entity — it emerges from data + systems.

2. **Hot-reloadable game logic.** Gameplay code compiles as a dynamic library (`.so`/`.dll`). The editor watches for source changes, recompiles in the background, and hot-swaps the library without restarting. Component data survives the reload via serialization. Hot-reload only occurs in edit mode — play mode must be stopped first.

3. **Rust-first.** Users write components and systems in Rust. No scripting language in v1. The architecture supports adding an embedded JS runtime (via `deno_core`) later without redesign — the ECS is the shared substrate, and a JS system is just another system implementation.

4. **Editor-integrated.** Components appear in the inspector with editable fields. An "Add Component" menu lists all registered gameplay components. The editor compiles game code and reports errors inline.

5. **Drop a file, it exists.** No manual registration, no editing a central file. Write a `.rs` file in the right directory, save, and it auto-registers via `automod` + `inventory`. The closest Rust can get to Unity's "just write a script" workflow.

6. **hecs is the world.** All entities — SDF objects, lights, cameras, gameplay entities — live in hecs. The renderer reads hecs (querying `Transform + SdfTree`, `Transform + Light`, etc.). There is no separate `Scene` data structure. The edit world (hecs) IS the authored state — scene save serializes it, scene load deserializes into it, and Play clones it into a separate play world.

## Architecture Overview

```
ENGINE (static binary, always running)
├── rkf-runtime
│     ├── hecs World (ALL entities: SDF objects, lights, cameras, gameplay)
│     ├── SystemContext, CommandQueue
│     ├── SystemScheduler (phases, ordering, per-phase flush)
│     ├── GameplayRegistry (component metadata, system list, blueprints)
│     ├── ComponentMeta trait (field introspection for inspector, MCP, future JS)
│     ├── DylibLoader (load, unload, reload game logic)
│     └── Sequence (built-in timed action runner)
├── rkf-editor
│     ├── Edit World (hecs — IS the authored state, renderer displays it)
│     ├── Play World (hecs clone — systems tick here, discarded on Stop)
│     ├── Edit Pipeline (unified: writes to edit world + pokes rinch signals)
│     ├── Inspector (renders component fields from ComponentMeta)
│     ├── Add Component menu (populated from GameplayRegistry)
│     ├── Play/Stop mode (clone edit world → play world, discard on stop)
│     └── File watcher + compiler (cargo build, status bar, error console)
├── rkf-render
│     └── Reads hecs (queries Transform + SdfTree, Transform + Light, etc.)
└── rkf-core, ... (unchanged)

GAME DYLIB (user's code, hot-reloaded)
├── components/    ← auto-discovered via automod
│     ├── health.rs      #[component]
│     ├── patrol.rs      #[component]
│     └── ...
├── systems/       ← auto-discovered via automod
│     ├── patrol_system.rs    #[system(phase = Update)]
│     ├── death_system.rs     #[system(phase = Update)]
│     └── ...
├── blueprints/    ← .rkblueprint RON files
└── lib.rs         ← register() calls registry.collect_all()
```

### Data Flow

```
1. Editor starts → compiles game dylib → loads dylib → calls register()
2. register() calls registry.collect_all() → finds all #[component] and #[system()] registrations via inventory
3. Editor loads .rkscene → deserializes entities + components into edit world (hecs)
4. Inspector reads/writes edit world via edit pipeline → rinch signals update
5. Viewport interactions (gizmo, sculpt, paint) → edit pipeline modifies edit world → rinch signals update
6. User hits Play → edit world cloned into play world → systems tick on play world → edit world untouched
7. User edits .rs file → file watcher triggers cargo build
8. Build succeeds → queued for reload (reload only happens in edit mode)
9. User hits Stop → play world discarded → renderer switches back to edit world
10. Queued reload applies: edit world gameplay components serialized → old dylib unloaded → new dylib loaded → register() → components deserialized
```

## Game State Store

### Motivation

Games need cross-entity, cross-scene state: player health, quest progress, inventory, world flags. In Unity, this is a perennial pain point — developers invent singleton GameManagers, static classes, or ScriptableObject hacks to hold global state. There's no standard answer.

This engine provides a built-in **global state store** as first-class infrastructure. It is not tied to any entity or scene — it persists across scene loads and is the foundation of the save system.

### Design

The state store is a namespaced key-value map with typed access, accessed through `GameManager`:

```rust
// Write (from any system)
ctx.game.set("player/health", 85.0_f32);
ctx.game.set("player/has_blue_key", true);
ctx.game.set("quest/main/stage", 3_i32);
ctx.game.set("world/time_of_day", 0.75_f32);

// Read (from any system)
let health = ctx.game.get::<f32>("player/health");       // Option<f32>
let stage = ctx.game.get::<i32>("quest/main/stage");      // Option<i32>
let has_key = ctx.game.get::<bool>("player/has_blue_key"); // Option<bool>

// Remove (cleanup when entity despawned, quest completed, etc.)
ctx.game.remove("player/has_blue_key");                   // single key
ctx.game.remove_prefix("player/a7f3b2c1/");              // all keys under prefix
```

The `/` separator is a convention for organization — the store is a flat map of string keys, not a hierarchy. But the convention enables namespace enumeration:

```rust
// List everything under "quest/"
for (key, value) in ctx.game.list("quest/") {
    // "quest/main/stage" => 3
    // "quest/side_01/complete" => true
}
```

### Supported Value Types

The store holds `GameValue` variants internally, with typed convenience methods on top:

| Type | Rust | Example |
|------|------|---------|
| Bool | `bool` | `ctx.game.set("unlocked", true)` |
| Int | `i64` | `ctx.game.set("score", 42_i64)` |
| Float | `f64` | `ctx.game.set("health", 85.0)` |
| String | `String` | `ctx.game.set("player/name", "Ada")` |
| Vec3 | `Vec3` | `ctx.game.set("player/direction", dir)` |
| WorldPosition | `WorldPosition` | `ctx.game.set("player/position", transform.position)` |
| Quat | `Quat` | `ctx.game.set("player/rotation", rot)` |
| Color | `[f32; 4]` | `ctx.game.set("sky/tint", color)` |
| List | `Vec<GameValue>` | For ordered collections |
| Ron | `String` (RON) | Arbitrary serde types via `set_ron`/`get_ron` |

For arbitrary structs that don't map to a built-in variant, the `Ron` type serializes via serde:

```rust
ctx.game.set_ron("player/inventory", &my_inventory);           // serializes to RON string
let inv: Inventory = ctx.game.get_ron("player/inventory")?;    // deserializes from RON
```

### Write Semantics

State writes are **immediate** — a call to `ctx.game.set()` takes effect right away. A system that writes a value can read it back immediately, and systems that run later in the same phase will see the new value.

This is consistent with how component value mutations work in the ECS (see Ordering below). The game state store does not use deferred writes.

### State vs Events

The store holds **persistent state** — values that exist until overwritten or removed. These survive scene loads, hot-reloads, and frame boundaries.

**Events** are separate: fire-and-forget notifications consumed within a single frame. Events carry a source entity and an optional data payload. They are immediate — visible to any system that runs after the emitter in the current frame (across all phases), then drained at frame end:

```rust
// Persistent state — stays until overwritten
ctx.game.set("door/kitchen/open", true);

// Fire-and-forget event — optional source entity + optional payload
ctx.game.emit("door_opened", Some(door_entity), None);
ctx.game.emit("damage_dealt", Some(attacker_entity), Some(GameValue::Float(25.0)));
ctx.game.emit("menu/resume", None, None);   // UI-originated, no source entity

// Receiving side — filter by event name
for event in ctx.game.events("door_opened") {
    // event.source: Option<Entity> — who emitted it (None for UI events)
    // event.data: Option<GameValue> — optional payload
    if let Some(door) = event.source { /* ... */ }
}
```

**Entity liveness:** An event's `source` entity may have been despawned between emission and consumption (e.g., emitted in Update, despawned at the Update→LateUpdate CommandQueue flush, consumed in LateUpdate). Consumers should check `ctx.world.contains(entity)` before using a source handle. This is consistent with hecs in general — entity handles can go stale, and code that holds them must account for that.

### State vs Components

| | Game State Store | Components |
|---|---|---|
| **Scope** | Global, game-level | Per-entity |
| **Examples** | Player progress, quest flags, settings, world state | This guard's patrol route, this door's open progress |
| **Accessed by** | Any system, by string key | Systems that query for specific component types |
| **Persists across** | Scene loads, play sessions (save file) | Only the entity's lifetime |
| **Save game** | Serialize the whole store | Not included (runtime simulation state) |

The distinction is clear: **components** describe what an entity IS and how it behaves right now. **Game state** describes what has HAPPENED in the game world — progress, decisions, unlocks, flags.

### Save/Load

The state store is the foundation of the save system. Saving a game is serializing the store:

```rust
// Save
let save_data = ctx.game.save_to_ron();
std::fs::write("saves/slot_1.rksave", save_data)?;

// Load
let save_data = std::fs::read_to_string("saves/slot_1.rksave")?;
ctx.game.load_from_ron(&save_data)?;
```

The `.rksave` file contains the entire state store as RON. Loading overwrites all state and emits a single `"state_loaded"` event (no source, no payload). Since `load_from_ron` is called from within a system (via `ctx.game`), the event follows normal event timing — it is visible to all systems that run later in the current frame and is drained at frame end. Save/load should be triggered from a gameplay system (e.g., a `save_system` in `Phase::Update`), not from engine code outside the system loop.

**Engine-automated restore:** Scene state (which scenes are loaded, camera position) is written to the store by engine systems each frame under reserved `engine/` keys. On `load_from_ron`, the engine reads these keys back and automatically restores the scene state — unloading scenes that shouldn't be loaded, loading ones that should be, and restoring the camera. This happens as part of the `"state_loaded"` processing, before gameplay systems see the event. The user does not need to handle scene transitions manually on load.

**Gameplay component restore:** Entity components in the play world are NOT automatically updated by `load_from_ron` — the store changed, but components still hold their old values. Gameplay systems should listen for the `"state_loaded"` event and call `sync_from_store` on entities with `#[persist]` fields to pull the loaded values into their components. Systems that derive state from store keys (e.g., HUD reading `"player/health"`) update naturally since they read the store each frame.

This keeps save/load as a single operation on a single data structure — the store is the save file, the engine handles its own state automatically, and gameplay systems handle theirs via the event.

### Persistence Helpers

Manually writing component fields to the store for save/load is tedious and error-prone. The `#[persist]` field annotation automates the mapping:

```rust
#[component]
pub struct Player {
    #[persist("player/{stable_id}/health")]
    pub health: f32,
    #[persist("player/{stable_id}/score")]
    pub score: i32,
    pub combo_timer: f32,        // not persisted, not in store
}
```

`{stable_id}` is substituted with the entity's `StableId` at sync time. The user controls the key structure — any combination of literal strings and `{stable_id}` is valid:

```rust
#[persist("health")]                         // simple singleton key
#[persist("player/{stable_id}/health")]      // per-entity, organized by role
#[persist("{stable_id}/health")]             // per-entity, organized by entity
#[persist("teams/red/{stable_id}/score")]    // deeply nested
```

The `#[component]` attribute macro generates sync methods for annotated fields:

```rust
// Component → store (call before saving)
ctx.sync_to_store::<Player>(entity);
// writes ctx.game.set("player/a7f3b2c1-4d5e-6f78-9a0b-c1d2e3f4a5b6/health", self.health), etc.

// Store → component (call after loading)
ctx.sync_from_store::<Player>(entity);
// reads ctx.game.get("player/a7f3b2c1-4d5e-6f78-9a0b-c1d2e3f4a5b6/health") into self.health, etc.
```

These are generic methods on `SystemContext`. The `#[component]` macro generates a `Persistable` trait implementation for each component with `#[persist]` fields — `SystemContext` delegates to it via `T: Persistable`. The user never interacts with the trait directly.

`{stable_id}` expands to the entity's full UUID string. Keys are verbose but unique — the store is a flat HashMap, so key length has no performance impact.

Since namespaces are nestable via `/` prefix matching, enumeration works at any depth:

```rust
ctx.game.list("player/")                                          // all players' persisted data
ctx.game.list("player/a7f3b2c1-4d5e-6f78-9a0b-c1d2e3f4a5b6/")   // one specific player
```

Missing keys are handled gracefully — the field keeps its current value if the store key doesn't exist. This means new fields added after a save file was created simply use their defaults.

The `#[persist]` annotation works with any `GameValue`-compatible type, including `Vec3`, `WorldPosition`, `Quat`, and arbitrary structs (via the `Ron` variant). The sync helpers handle type conversion transparently.

Key uniqueness is the user's responsibility. Using `{stable_id}` in the key path guarantees per-entity uniqueness. Omitting it is valid for singleton components but will collide if multiple entities share the same key.

This is not a full save system — it is a lightweight bridge between component data and the game state store. The user controls when sync happens and which entities participate. The store remains the single thing that gets serialized to `.rksave`. There is no automatic "sync all persisted components" step — the user explicitly decides what to save and when. This keeps the save system predictable: adding a `#[persist]` field is a declaration of intent, not a behavior change. The user writes the save logic that calls `sync_to_store` for the components they care about.

## System Context

Every system receives a `SystemContext` — the complete interface between gameplay code and the engine:

```
SystemContext:
    world: &mut hecs::World           // all entities and components
    game: &mut GameManager            // game state store, events, elapsed time
    dt: f32                           // frame delta time in seconds
    commands: CommandQueue             // deferred spawn/despawn/insert/remove
    registry: &GameplayRegistry       // component metadata, blueprint catalog
```

`SystemContext` is intentionally minimal. Input state, collision events, and other engine data are accessed by querying components or reading `GameManager` events — not through dedicated fields. This keeps the interface stable and avoids coupling systems to specific engine subsystems. Additional fields can be added later without breaking existing systems.

**Note:** An input system (keyboard, mouse, gamepad, remapping) is a prerequisite for gameplay but is specified in a separate document. It will feed into `SystemContext` — likely via an `Input` field or a queryable component. The behavior system architecture does not depend on the specific input model.

### Position Ergonomics

The engine uses `WorldPosition { chunk: IVec3, local: Vec3 }` internally to avoid float precision loss at large distances. But gameplay code should not need to think about chunks. All gameplay-facing APIs that accept positions take `impl Into<WorldPosition>`, and `WorldPosition` implements `From<Vec3>` with automatic chunk normalization:

```rust
// Users write Vec3 — precision is handled automatically:
ctx.commands.spawn_blueprint("guard", Vec3::new(10.0, 0.0, 5.0));

// Even at large distances, From<Vec3> normalizes to the correct chunk:
let far_away = Vec3::new(100_000.0, 0.0, 0.0);
// → WorldPosition { chunk: (computed), local: (small value) }
```

Arithmetic on `WorldPosition` handles chunk boundaries transparently:

```rust
let offset = Vec3::new(5.0, 0.0, 0.0);
let new_pos = transform.position + offset;   // WorldPosition + Vec3 → WorldPosition
let dir = pos_a - pos_b;                     // WorldPosition - WorldPosition → Vec3 (via f64)
```

`Transform.position` is always `WorldPosition` under the hood — full precision is maintained. Users who never import `WorldPosition` get correct behavior at any distance. For advanced use cases (explicit chunk math, cross-chunk teleportation), `WorldPosition` is available directly.

### Convenience Methods

`SystemContext` provides shortcuts for common operations:

```
ctx.commands.spawn_blueprint("guard", position)                          // look up blueprint in registry, queue spawn
ctx.game.emit("door_opened", Some(door_entity), None)                   // event with source entity
ctx.game.emit("hit", Some(attacker), Some(GameValue::Float(25.0)))      // event with source + payload
ctx.game.set("quest/main/stage", 3)                                     // persistent state
```

### CommandQueue

Rust's ownership rules prevent modifying the `hecs::World` while iterating over it. The `CommandQueue` collects deferred **structural** mutations — spawn, despawn, add component, remove component — that are applied after the current phase completes:

```
CommandQueue:
    spawn(builder: EntityBuilder) -> TempEntity                              // current scene
    spawn_persistent(builder: EntityBuilder) -> TempEntity                   // no scene, survives transitions
    spawn_in_scene(builder: EntityBuilder, scene: &str) -> TempEntity        // explicit scene
    spawn_blueprint(name: &str, position: impl Into<WorldPosition>) -> TempEntity              // current scene
    spawn_blueprint_persistent(name: &str, position: impl Into<WorldPosition>) -> TempEntity   // no scene
    spawn_blueprint_in_scene(name: &str, position: impl Into<WorldPosition>, scene: &str) -> TempEntity
    despawn(entity: Entity)
    insert<C: Component>(entity: Entity, component: C)
    insert<C: Component>(temp: TempEntity, component: C)   // batch onto pending spawn
    remove<C: Component>(entity: Entity)
```

**Scene ownership defaults:** `spawn()` and `spawn_blueprint()` tag the new entity with the current scene. If exactly one scene is loaded, that scene is used automatically. If multiple scenes are loaded, `spawn()` is a **runtime error** — the user must call `spawn_in_scene()` to specify which scene explicitly. This avoids silent "spawned into the wrong scene" bugs. `spawn_persistent()` always works regardless of how many scenes are loaded. Use the `_persistent` variants for entities that must survive scene transitions (player, HUD, music controller).

`TempEntity` is a handle to a not-yet-materialized entity. It works with the same `CommandQueue` — you can call `commands.insert(temp, SomeComponent {...})` to batch additional components onto the pending spawn. All commands for a `TempEntity` are applied atomically when the queue flushes. You cannot query a `TempEntity` from hecs — it doesn't exist there until flush.

```rust
let guard = ctx.commands.spawn_blueprint("guard", position);
ctx.commands.insert(guard, Alerted { target: player_entity });
ctx.commands.insert(guard, CustomPatrolRoute { waypoints: vec![...] });
// All three materialize together on flush — no half-built entity visible mid-phase
// guard belongs to the current scene by default
```

Commands are flushed **per-phase**. An entity spawned in `Phase::Update` does not exist until `Phase::LateUpdate`. This is separate from component value mutations (like `health.current -= 10`), which are immediate — see Ordering below.

**Always use `ctx.commands` for structural changes.** While `ctx.world` is a raw `&mut World` that technically allows direct `spawn()`/`despawn()` calls, doing so bypasses auto-component injection (StableId, Transform, EditorMetadata), name deduplication, and scene ownership tagging. Direct world mutation is unsupported — use the CommandQueue.

## Phases and Ordering

### Phases

Systems are organized into phases that define broad execution order within a frame:

```
Phase::Update        — general gameplay: AI, movement, interactions, game logic
Phase::LateUpdate    — runs after Update: camera follow, UI sync, cleanup
```

Phases are hard boundaries: the `CommandQueue` (structural ECS changes — spawn, despawn, add/remove component) is flushed between phases. This means entities spawned in Update exist by the time LateUpdate runs, but not during Update.

Additional phases (`PrePhysics`, `PostPhysics`) will be added when Rapier is integrated into the system scheduler. Currently physics runs as a separate engine step.

### Ordering Within a Phase

By default, systems within a phase run in **arbitrary order**. The engine makes no guarantees. For most systems this is fine — they operate on different components and don't interact.

When ordering matters, systems declare explicit dependencies using the **full function name**:

```rust
#[system(phase = Update)]
fn combat_system(ctx: &mut SystemContext) {
    // modifies Health components
}

#[system(phase = Update, after = "combat_system")]
fn death_system(ctx: &mut SystemContext) {
    // reads Health — guaranteed to see combat's changes
}
```

The scheduler builds a dependency graph within each phase. Systems with no dependencies run in arbitrary order. Systems with `after` or `before` are topologically sorted. Cycles are detected and reported as errors in the editor console.

**Name resolution:** Systems are referenced by their function name. If the name is unique across the entire game crate, the short name works. If two modules define a system with the same name (e.g., `ai::combat_system` and `player::combat_system`), both are flagged as ambiguous — use the module-qualified form (`after = "ai::combat_system"`) to disambiguate.

| Mechanism | Use case |
|---|---|
| No annotation (default) | "I don't care when I run" — the common case |
| `after = "combat_system"` | "I need to see what combat wrote" |
| `before = "hud_system"` | "I need to write before HUD reads" |
| `after = "ai::combat_system"` | Disambiguate when names collide across modules |
| Separate phases | "All Update logic settles before LateUpdate reacts" |

### What's Deferred, What's Immediate

Within a phase, different kinds of mutations have different timing:

| Mutation | Timing | Why |
|---|---|---|
| **Structural ECS changes** (spawn, despawn, add/remove component) | **Deferred** — applied between phases via CommandQueue | Can't modify entity structure while iterating queries |
| **Component value writes** (`health.current -= 10` via `query_mut`) | **Immediate** — visible to systems that run later in the same phase | Inherent to hecs — `query_mut` gives `&mut` to live data |
| **Game state writes** (`ctx.game.set(...)`) | **Immediate** — visible to systems that run later in the same phase | Consistent with component value writes |
| **Events** (`ctx.game.emit(...)`) | **Immediate** — visible to all systems that run later in the frame (across phases) | Append-only, drained once at frame end |

The practical consequence: **system ordering within a phase is arbitrary by default, but matters when two systems read/write the same data.** In those cases, use `after`/`before` to make the dependency explicit. The Systems panel in the editor shows the resolved execution order so ordering is always visible and debuggable.

### Frame Execution Order

```
1. Engine: input polling, window events
2. Engine: pre-tick bridge
     - Update dt
     - Read UI rinch signals → inject into game state store and event queue
       (so systems see last frame's UI interactions as normal store values and events)
     - Process input state (when input system exists)
3. Gameplay: run all Phase::Update systems (respecting dependency order)
4. Engine: flush CommandQueue (structural ECS changes materialize)
5. Gameplay: run all Phase::LateUpdate systems (respecting dependency order)
6. Engine: flush CommandQueue, drain events (events cleared once per frame, not per phase)
7. Engine: post-tick bridge
     - Push game state store values → rinch signals (so UI reflects this frame's changes)
8. Engine: transform bake → BVH refit → render
```

Step 2 (pre-tick bridge) is where the engine translates external inputs into the formats systems consume — UI interactions become store values and events, raw input becomes queryable input state. Step 7 (post-tick bridge) is the reverse: systems' store writes become rinch signals so the UI updates reactively. This keeps the boundary clean: systems only interact with `ctx.game` and input state, never with rinch directly.

Events persist across phases within a frame — an event emitted during Update is visible to LateUpdate systems. This supports the common pattern of "gameplay emits events in Update, UI/camera systems react in LateUpdate." Events are drained once at the end of the frame (step 6), not between phases.

The CommandQueue is flushed per-phase (steps 4 and 6) — entities spawned in Update exist by the time LateUpdate runs.

Systems run between input and rendering. They see this frame's input and their changes are visible in this frame's render.

## Component Design

### Components Are Pure Data

A component is a Rust struct annotated with the `#[component]` attribute macro:
- The macro automatically derives `Serialize`, `Deserialize`, generates `ComponentMeta`, generates type-erased `ComponentEntry` function pointers (including field-level `get_field`/`set_field`), and registers the type via `inventory`
- Components contain only serializable data — no function pointers, closures, or trait objects

```rust
#[component]
pub struct Patrol {
    pub waypoints: Vec<Vec3>,
    pub speed: f32,
    #[serde(skip, default)]
    pub current_index: usize,     // transient — reset on reload/scene load
    #[serde(skip, default)]
    pub wait_timer: f32,          // transient — reset on reload/scene load
}
```

Components may use any type that implements `Serialize + Deserialize`: primitives, `Vec`, `HashMap`, `Option`, `String`, enums with data, nested structs. They may reference other entities via `hecs::Entity` (see Entity References below).

**Why no function pointers or trait objects?** Components must survive hot-reload via serialization. Function pointers and vtable pointers become invalid when the dylib is unloaded. Use enums instead:

```rust
// BAD: trait object — can't serialize, dangling after reload
pub struct AiBrain {
    pub strategy: Box<dyn CombatStrategy>,
}

// GOOD: enum — serializable, survives reload
pub struct AiBrain {
    pub strategy: CombatStrategy,
}

enum CombatStrategy {
    Melee { range: f32 },
    Ranged { preferred_distance: f32 },
    Flee,
}
```

### Enum Components

The `#[component]` macro also works on enums. An enum component represents a discrete state rather than a bag of fields:

```rust
#[component]
pub enum GuardState {
    Patrolling,
    Alerted,
    Chasing,
    Attacking,
    Returning,
}
```

Enum components have no named fields in `ComponentMeta`. The inspector renders them as a **variant selector** (dropdown for unit variants, variant selector + fields for data variants). `get_field`/`set_field` do not apply — the whole component is read/written as a unit via `serialize`/`deserialize_insert`. Serialization uses standard serde enum encoding (unit variants as strings, data variants as tagged structs).

### Transient vs Persistent Fields

Fields marked `#[serde(skip)]` are transient — they reset to their default value on scene load and hot-reload. These hold runtime-only state like timers, cached indices, and in-progress interpolation values.

The `#[component]` macro detects `#[serde(skip)]` fields and marks them as `transient: true` in the generated `ComponentMeta`, which the editor inspector uses to hide or gray them out in edit mode.

### The ComponentMeta Trait

Provides field-level introspection for the editor inspector, MCP tools, and (future) JS bindings:

```
trait ComponentMeta:
    fn type_name() -> &'static str
    fn fields() -> &'static [FieldMeta]
```

```
FieldMeta:
    name: &'static str
    field_type: FieldType          // Float, Int, Bool, Vec3, String, Entity, Enum, List
    transient: bool                // true for #[serde(skip)] fields
    range: Option<(f64, f64)>      // for numeric slider display
    default: Option<GameValue>     // inspector shows this when adding component
```

The `#[component]` attribute macro generates this from the struct definition. The macro also generates type-erased function pointers for serialization, deserialization, field-level access, insertion, removal, and presence-checking — the internal plumbing that makes hot-reload, scene save/load, inspector, and MCP tools work across the dylib boundary (see Hot-Reload System below).

Three consumers share this one reflection layer:
- **Editor inspector** — renders fields as sliders, text inputs, dropdowns
- **MCP tools** — `component_get` and `component_set` use it for field discovery and validation
- **Future JS bindings** — property access on JS component proxy objects

## Entity References

### StableId

Every entity automatically receives a `StableId` component at creation — a unique, persistent identifier that never changes across the entity's lifetime:

```
StableId(Uuid)   // auto-assigned, globally unique (UUID v4)
```

StableId is an engine-managed component — the engine assigns it automatically via `Uuid::new_v4()`, and users never add, remove, or modify it. It lives in hecs like any other component (queryable via `world.query::<&StableId>()`), but is not user-facing. UUIDs require no central counter or coordination — any code path (editor, CommandQueue flush, blueprint spawn, scene load) can generate one independently with no collision risk.

StableIds are the serialization and reference layer. When a component field holds an `hecs::Entity` reference, it is serialized as the target's `StableId` and resolved back to a live `hecs::Entity` on load. This handles all edge cases: entities with duplicate names, renamed entities, runtime-spawned entities with no hierarchy position.

`hecs::Entity` does not natively implement `Serialize`/`Deserialize`. The `#[component]` macro detects `Entity` fields in direct struct fields and common wrapper types (`Option<Entity>`, `Vec<Entity>`) and generates serialization code that remaps them through the bidirectional index maps. The `ComponentEntry.serialize` function pointer receives `&World`, which gives it access to the StableId components for lookup. Users write `hecs::Entity` in their component fields and the macro handles the rest — no manual conversion or wrapper types needed. Engine components like `Sequence` that contain `Entity` references in complex nested structures (enum variants inside Vecs) implement their own Entity remapping — the macro is not involved.

The engine maintains bidirectional index maps alongside the component for fast lookup:

```
stable_to_entity: HashMap<Uuid, Entity>
entity_to_stable: HashMap<Entity, Uuid>
```

Users rarely interact with StableIds directly — they work with `hecs::Entity` in code and see entity names in the inspector.

### Entity Lookup API

Finding entities is a common need. The engine provides multiple tools for different use cases, all accessed through `SystemContext`:

**Marker components** — find unique entities by type (the ECS-idiomatic pattern):
```rust
#[component]
pub struct Player;   // empty marker, no fields

let (player_entity, (_player, transform)) = ctx.find_one::<(&Player, &Transform)>()?;
// Returns Result<(Entity, Q), QueryError>
// QueryError::NotFound  — no entity matches the query
// QueryError::Multiple  — more than one entity matches (ambiguous)
```

**Path-based lookup** — find entities by their position in the hierarchy:
```rust
// Absolute path (from root-level entities — those without a Parent component)
let door = ctx.find_path("World/Doors/KitchenDoor")?;

// Relative path (from a specific entity)
let weapon = ctx.find_relative(me, "Arm/Hand/Sword")?;    // down into children
let sibling = ctx.find_relative(me, "../OtherGuard")?;      // up then across
```

Paths use entity names from `EditorMetadata` and the parent/child hierarchy. Relative paths are robust to moving subtrees — `"../Sibling"` describes the relationship, not the absolute location.

**Tags** — find groups of entities:
```rust
let enemies = ctx.find_tagged("enemy");   // iterator of Entity
// Uses EditorMetadata.tags, indexed for fast lookup
```

**Component queries** — find entities by their data (standard hecs):
```rust
for (entity, (transform, health)) in ctx.world.query::<(&Transform, &Health)>().iter() {
    if health.current < 50.0 { /* ... */ }
}
```

### Entity Naming

Entity names (from `EditorMetadata.name`) must be unique among siblings in the hierarchy. When an entity is created or renamed, the engine checks siblings and auto-appends a suffix if there's a collision: "Guard", "Guard_2", "Guard_3". This is enforced at both the editor level (on rename, on entity creation) and the runtime level (CommandQueue auto-deduplicates names on flush). This guarantees that path-based lookups are always unambiguous, regardless of whether the entity was created by the editor or by a gameplay system. (Same approach as Godot's scene tree.)

### Inspector Entity References

In the inspector, entity reference fields display as a tree picker showing entity names:

```
┌─ Chaser ────────────────────────┐
│   Target: [Guard_01 ▾]         │  ← click to pick from hierarchy
│   Speed:  3.0                  │
└─────────────────────────────────┘
```

Three representations of the same reference, converted automatically:

| Context | Representation |
|---|---|
| Runtime (code) | `hecs::Entity(42)` — fast, direct |
| Serialized (scene file) | `StableId(a7f3...)` — stable across sessions |
| Inspector (UI) | `"Guard_01"` — human-readable |

## System Design

### Systems Are Free Functions

A system is a function with the signature `fn(&mut SystemContext)`, annotated with the `#[system]` attribute macro:

```rust
#[system(phase = Update)]
fn patrol_system(ctx: &mut SystemContext) {
    for (entity, (transform, patrol)) in
        ctx.world.query_mut::<(&mut Transform, &mut Patrol)>()
    {
        if patrol.waypoints.is_empty() { continue; }
        // ... movement logic
    }
}
```

The `#[system]` attribute registers the function with `inventory` so it is auto-discovered at startup. No manual registration needed — write the function in a file under `systems/`, and it participates in the game loop. Optional parameters:

- `phase = Update` (required) — which phase to run in
- `after = "system_name"` — run after another system within the same phase (repeatable for multiple dependencies)
- `before = "system_name"` — run before another system within the same phase (repeatable for multiple dependencies)

Multiple dependencies use repeated attributes:

```rust
#[system(phase = Update, after = "combat_system", after = "movement_system")]
fn death_system(ctx: &mut SystemContext) { /* ... */ }
```

Systems carry no per-instance state — all state lives in components. This keeps them simple, testable, and compatible with hot-reload (function pointers are re-acquired from the new dylib on each reload).

### Auto-Registration

The game dylib uses `automod` to discover all modules and `inventory` to collect all annotated components and systems:

```rust
// my-game-logic/src/lib.rs
automod::dir!("src/components");
automod::dir!("src/systems");

#[no_mangle]
pub fn register(registry: &mut GameplayRegistry) {
    registry.collect_all();  // gathers all #[component] and #[system()] items
}
```

The user's workflow:
1. Create `systems/patrol_system.rs`
2. Write the system function with `#[system(phase = Update)]`
3. Save
4. Editor recompiles → system is active

No editing `lib.rs`, no editing a registration function, no `mod` declarations.

### Panic Safety

User system code may panic (array bounds, unwrap, etc.). A panic unwinding across the dylib boundary into the engine is undefined behavior. The scheduler wraps every system call in `std::panic::catch_unwind()`:

- If a system panics: catch the panic, log the error to the editor console, mark the system as **faulted**
- Faulted systems are skipped on subsequent frames — no error spam at 60fps
- On hot-reload: all fault flags are cleared, giving the new code a fresh start
- The engine and editor remain fully stable regardless of game code bugs

## Sequences

State machine components handle ongoing, event-reactive behaviors like AI states, physics responses, and mode switching. But timed multi-step sequences — flash three times then die, slide a door open, run a cutscene — are tedious to express as state machines.

The `Sequence` component provides entity-centric commands for frame-spanning actions:

```
Sequence:
    steps: Vec<SequenceStep>
    current: usize
    timer: f32
```

Built with a linear, imperative-feeling API:

```rust
// Door opens, waits, closes
Sequence::new()
    .move_to(open_position, 1.0)
    .wait(3.0)
    .move_to(closed_position, 1.0)
    .emit("door_closed")     // source = this entity, no payload

// Rise, spin, shrink, vanish
Sequence::new()
    .move_by(Vec3::Y * 5.0, 1.0)
    .rotate_by(Quat::from_rotation_y(PI), 0.5)
    .scale_to(Vec3::ZERO, 0.3)
    .despawn()

// Pulse emission three times, then die
Sequence::new()
    .repeat(3, |s| s
        .set_state("fx/flash", true)
        .wait(0.15)
        .set_state("fx/flash", false)
        .wait(0.15)
    )
    .despawn()
```

### Available Steps

| Step | Description |
|------|-------------|
| `.wait(seconds)` | Pause for N seconds of game time |
| `.move_to(position, duration)` | Lerp position to target — accepts `Vec3` or `WorldPosition` (via `impl Into<WorldPosition>`) |
| `.move_by(offset, duration)` | Lerp position by relative `Vec3` offset |
| `.rotate_to(quat, duration)` | Lerp rotation to target |
| `.rotate_by(quat, duration)` | Lerp rotation by relative amount |
| `.scale_to(scale, duration)` | Lerp scale to target |
| `.emit(event_name)` | Fire-and-forget event (source = this entity, no payload) |
| `.emit_with(event_name, payload)` | Event with data (source = this entity, with `GameValue` payload) |
| `.emit_from(event_name, source, payload)` | Event with custom source entity and optional payload |
| `.set_state(key, value)` | Write persistent value to game state store |
| `.spawn_blueprint(name)` | Instantiate a blueprint at this entity's position (inherits this entity's scene ownership) |
| `.despawn()` | Remove this entity |
| `.repeat(n, \|s\| ...)` | Repeat a sub-sequence N times |

### Easing

All lerp steps accept an optional easing function:

```rust
.move_to(target, 1.0).ease(Ease::OutBounce)
.scale_to(Vec3::ZERO, 0.5).ease(Ease::InCubic)
```

Default easing is linear.

### Implementation

`Sequence` is a regular component — it derives `Serialize + Deserialize` and survives hot-reload. A single engine-provided system (`sequence_system`) ticks all `Sequence` components each frame in `Phase::Update`, advancing through steps. It is a normal system — visible in the Systems panel, and user systems can declare `after = "sequence_system"` or `before = "sequence_system"` for explicit ordering. No async, no coroutines, no lifetimes — just data.

**Completion:** When a Sequence finishes its last step, `sequence_system` **removes the Sequence component** from the entity. User systems detect completion by checking for the absence of a Sequence component (e.g., the guard AI system transitions state when `!world.has::<Sequence>(entity)`). This is clean and ECS-idiomatic — component presence is the signal, not a flag.

Steps that refer to the entity's current position (like `move_to`) capture the starting value when the step begins executing, not when the Sequence is built. This means a sequence built before play mode starts will use the entity's position at the time each step activates.

Steps that reference other entities (like `.emit_from(name, source, payload)`) store `hecs::Entity` handles. If the referenced entity is despawned before the step executes, the handle is stale — this is the general entity reference liveness problem, not specific to sequences. The sequence system checks entity liveness at execution time and substitutes `None` for dead source entities.

`SequenceStep` is an enum — new step variants can be added as the engine evolves. Visual effect steps (per-object tint, emission override, material swap) will be added when the renderer supports per-object visual overrides. The current step set covers transform, timing, events, state, spawning, and lifecycle.

### State Machines + Sequences Together

Complex behaviors combine both patterns. The state machine handles decisions (what to do), sequences handle execution (how to do it):

```rust
#[component]
pub enum GuardState {
    Patrolling,
    Alerted,        // sequence: turn toward noise, wait, look around
    Chasing,
    Attacking,      // sequence: wind up, strike, cooldown
    Returning,
}
```

The guard AI system matches on the enum for transitions. When entering `Alerted` or `Attacking`, it inserts a `Sequence` component. When the sequence completes, the system transitions to the next state. The enum stays small because the tedious frame-by-frame timing is offloaded to the sequence runner.

## Hot-Reload System

### Dylib Architecture

Gameplay code compiles as a Rust `cdylib` crate:

```toml
# my-game-logic/Cargo.toml
[lib]
crate-type = ["cdylib"]

[dependencies]
rkf-runtime = { path = "../crates/rkf-runtime" }
```

The dylib exports a single function:

```rust
#[no_mangle]
pub fn register(registry: &mut GameplayRegistry) {
    registry.collect_all();
}
```

### ABI Compatibility

Passing `&mut GameplayRegistry` across the dylib boundary relies on Rust's ABI being consistent between the engine binary and the game dylib. Rust does not guarantee a stable ABI, but in practice it is consistent when both are compiled with the same `rustc` version and the same dependency versions.

**Hard constraint: engine and game dylib must be compiled with the same Rust toolchain.** The engine embeds its `rustc` version hash at compile time. On dylib load, the version hash is compared. On mismatch, the dylib is rejected with a clear error:

```
Error: Game code was compiled with rustc 1.86.0 but the editor requires 1.85.0.
Run 'rustup override set 1.85.0' in your project directory.
```

This is analogous to Unity shipping a specific Mono/IL2CPP runtime — the game code must target the engine's compiler version.

### Type-Erased Component Operations

The engine (static binary) does not know about game-defined types like `Patrol` or `Health`. But it needs to serialize, deserialize, inspect, and modify them during hot-reload, in the inspector, and via MCP tools. The `#[component]` attribute macro generates type-erased function pointers for each component type:

```
ComponentEntry (stored in GameplayRegistry per component type):
    name: &'static str                                    // stable string ID (from ComponentMeta)
    serialize:          fn(&World, Entity) -> Option<String>   // serialize to RON if entity has it
    deserialize_insert: fn(&mut World, Entity, &str) -> Result // deserialize and attach to entity
    remove:             fn(&mut World, Entity)                  // remove from entity (calls drop)
    has:                fn(&World, Entity) -> bool              // check presence
    get_field:          fn(&World, Entity, &str) -> Option<GameValue>   // read one field by name
    set_field:          fn(&mut World, Entity, &str, GameValue) -> Result // write one field by name
    meta:               &'static [FieldMeta]                   // field introspection
```

The `get_field`/`set_field` function pointers provide **granular field-level access** to type-erased components. Four consumers depend on this:
- **Inspector**: reads field values for display, writes on slider drag or text edit
- **MCP tools**: `component_get` and `component_set` read/write specific fields
- **Push to edit**: copies individual field values from the play world to the edit world
- **Undo/redo**: captures and restores individual field values as `GameValue`, not whole-component RON blobs

These function pointers live in the dylib. When the dylib is unloaded, they become invalid — which is why the reload cycle carefully serializes and removes components *before* unloading, and deserializes using the *new* dylib's function pointers after loading.

### The Reload Cycle

```
1. File watcher detects .rs change in game crate
2. Editor status bar: "Compiling..."
3. Editor spawns: cargo build -p my-game-logic --lib
4. On compile error: status bar shows error, old dylib stays loaded, nothing changes
5. On success:
   a. For each registered component type (using OLD dylib's function pointers):
      - Visit all entities that have this component
      - Serialize the component data to RON string
      - Remove the component from the entity (drop runs in old dylib)
      - Store: (entity_id, component_name, ron_string)
   b. Unload old dylib (dlclose)
   c. Load new dylib (dlopen via libloading)
   d. Verify rustc version hash matches
   e. Call register() → populate registry with new function pointers and metadata
   f. For each stored (entity_id, component_name, ron_string):
      - Look up component_name in new registry
      - Call deserialize_insert with the new dylib's function pointer
      - On failure: log warning, skip this component (entity keeps other components)
   g. Status bar: "Reloaded"
```

Entities are never despawned or respawned during reload. Only gameplay component data is round-tripped through serialization. Engine components (`Transform`, `Parent`, etc.) are defined in the static binary and are completely unaffected.

### What Survives a Reload

| Data | Survives? | How |
|------|-----------|-----|
| Component field values | Yes | Serialized to RON, deserialized with new types |
| Transient fields (`#[serde(skip)]`) | Reset to default | By design — runtime state should re-derive |
| Entity IDs | Yes | Entities stay in the World, never despawned |
| Engine components (Transform, etc.) | Untouched | Defined in static binary, not in dylib |
| System function pointers | Replaced | Re-acquired from new dylib via `register()` |
| Game state store | Untouched | Lives in GameManager, not in dylib |
| Faulted system flags | Cleared | New code gets a fresh start |

### Handling Schema Changes

The `#[component]` macro configures serde for schema resilience: it adds `#[serde(default)]` at the struct level (so missing fields use `Default::default()`) and uses lenient deserialization that ignores unknown fields. This means hot-reload and scene load tolerate schema changes without user intervention. The component type must implement `Default` — the macro derives it automatically.

When a user adds, removes, or renames a field on a component:

- **Added field**: Missing in the serialized data — gets `Default::default()` via struct-level `#[serde(default)]`.
- **Removed field**: Present in the serialized data but unknown to the new type — silently dropped by lenient deserialization.
- **Renamed field**: Old field is dropped (treated as removed), new field gets default. Users can use `#[serde(alias = "old_name")]` to migrate gracefully.
- **Changed type**: Deserialization fails for that component. The component is removed from the entity and a warning is logged. The entity keeps all other components.
- **Removed component type entirely**: Serialized data for that type has no matching entry in the new registry. Data is discarded, warning logged.

The editor logs all reload issues to the console panel so users can see what was reset or dropped.

## Play/Stop Mode

### Two Worlds

The behavior system uses two separate hecs Worlds:

- **Edit World**: The editor's live working data. All entities with all components (engine AND gameplay). The inspector reads/writes this. The renderer displays this. This is never touched by gameplay systems. It is the in-memory representation of the scene file.
- **Play World**: A copy of the edit world, created when the user enters Play mode. Systems tick on this copy. It is disposable — discarded on Stop.

These are separate `hecs::World` instances. They are never the same object.

### Edit Mode

During edit mode, the edit world is the source of truth. The dylib must be loaded — gameplay components are live in hecs, and the inspector renders their fields via `ComponentMeta`.

All edit operations — inspector fields, gizmo transforms, sculpt, paint, MCP tools, undo/redo — go through the **edit pipeline**, which:
1. Writes to the edit world (persists to save, survives Play/Stop)
2. Pokes the relevant rinch signal (UI updates reactively)

The renderer reads the edit world directly. There is no separate authored state data structure — the edit world IS the authored state.

This is bidirectional. A gizmo drag originates in viewport space, but the edit pipeline ensures the edit world and UI signals stay in sync. Scene save serializes the edit world to `.rkscene`. Scene load deserializes into the edit world.

### Play Mode

When the user enters Play mode:

1. The editor creates a **play world** by cloning the edit world (see below)
2. The editor snapshots the game state store
3. The edit world and store snapshot are untouched for the duration of play
4. Systems begin ticking each frame on the play world, modifying components and the store
5. The renderer switches to displaying the play world
6. Play mode has its own **separate UI controls** — not a tinted version of the authoring inspector
7. Entities spawned during play exist only in the play world
8. Entities despawned during play are only removed from the play world

### Play World Clone

Creating the play world is not a simple memcpy — gameplay components are type-erased from the engine's perspective:

1. Create a new empty `hecs::World` (the play world)
2. Iterate all edit world entities, create corresponding entities in the play world
3. Build a mapping from edit world `hecs::Entity` handles to play world handles (via `StableId`)
4. **Engine components** (`Transform`, `Parent`, `Light`, `SdfTree`, etc.): the engine knows these types and clones them directly, remapping entity references using the mapping from step 3
5. **Gameplay components**: serialized from the edit world via `ComponentEntry.serialize` (entity references become `StableId`), then deserialized into the play world via `ComponentEntry.deserialize_insert` (resolving `StableId` to play world entity handles)

This reuses the same serialize/deserialize infrastructure that hot-reload and scene save/load already require. It is a one-shot operation (happens once when the user hits Play), so the serialization cost is negligible. The key benefit: entity reference remapping (in `Parent`, gameplay component `Entity` fields, etc.) is handled automatically by the StableId round-trip — no custom clone or remap logic needed.

### Stop Mode

When the user exits Play mode:

1. Systems stop ticking
2. The play world is discarded entirely
3. The game state store is restored from the snapshot
4. The renderer switches back to displaying the edit world (which was never modified)

Stop is instant and unconditional — all play state is discarded.

### Push to Edit

During play mode, the play UI can show a **"push to edit"** button next to individual property values. Clicking it writes that specific value to the corresponding entity and component in the edit world. This lets the user preserve a play-mode tweak (e.g., a patrol speed that feels right) without leaving play mode.

This is a targeted, user-initiated action — not an automatic diff or divergence report. The user pushes exactly the values they want to keep, one at a time, while they're still seeing the result in the viewport.

### Play Mode UI

Play mode uses **completely separate controls** from the authoring inspector. The authoring controls read/write the edit world. The play mode controls read/write the play world. They are different UI panels with different data sources — not a tinted version of the same inspector. The specific play mode UI design is specified elsewhere.

### Edit vs Runtime Geometry Modification

Edit-mode tools (sculpt, paint, gizmo) are **disabled during Play**. They are authoring tools that go through the edit pipeline and modify the edit world.

Runtime geometry modification — destruction, procedural terrain deformation, paint guns — is a separate capability available to gameplay systems. The specific API (likely a method on `SystemContext` or a deferred command) is TBD — it will expose the same brush/occupancy operations that the edit-mode tools use, but targeting the play world. These changes are gameplay state: they exist only in the play world (via copy-on-write brick data) and are discarded on Stop.

The underlying voxel modification code (occupancy bits, surface voxels, SDF recompute) is shared. The difference is the data flow: edit-mode tools use the edit pipeline, runtime operations use the ECS.

### Brick Pool Sharing (Copy-on-Write)

When the edit world is cloned into a play world, `SdfTree` components reference brick pool data (voxelized geometry). Both worlds share this data by default — no duplication, no extra memory.

If a gameplay system modifies geometry during play (runtime sculpting, destruction), only the affected bricks are copied into play-world-owned storage. The edit world's brick data is never touched. On Stop, the play-world copies are freed.

This keeps memory usage minimal for non-destructive play and only allocates extra for the specific bricks that runtime operations modify.

### Scene Transitions During Play

Gameplay systems can load and unload scenes during play mode. Scene loads target the **play world** — the edit world is never affected:

- **Unload**: all play world entities with `SceneOwnership { scene: Some("level_01") }` are despawned
- **Load**: the new `.rkscene` is deserialized into the play world, entities tagged with the new scene name
- **Persistent entities** (`scene: None`) survive scene transitions — the player, HUD, music controller, etc.

On Stop, everything in the play world is discarded — both the original cloned entities and any entities loaded during play. The edit world still holds the scene that was open when Play was pressed.

The scene load/unload API is available through `SystemContext` (specific methods TBD), using the same deserialization infrastructure as editor scene loading but targeting the play world instead of the edit world.

### Hot-Reload and Play Mode

Hot-reload only occurs in **edit mode**. If the user edits code during play, the build completes and the reload is queued — it applies automatically when play stops and the editor returns to edit mode. This avoids the complexity of round-tripping two worlds' gameplay components simultaneously.

The iteration loop during play is: Stop → queued reload applies automatically → Play. Since Stop is instant (discard play world, restore store), this adds only seconds to the cycle.

## Scene Ownership

Entities in the ECS are tagged with scene ownership to control their lifecycle across scene loads:

```rust
// Engine component — tracks which scene spawned this entity
pub struct SceneOwnership {
    pub scene: Option<String>,   // None = persistent, survives scene unloads
}
```

When a scene is unloaded, all entities with `scene == Some("level_01")` are removed. Entities with `scene == None` persist indefinitely — the player, HUD, music controller, etc.

```rust
ctx.commands.spawn(builder);                               // current scene (cleaned up on scene unload)
ctx.commands.spawn_persistent(builder);                    // persistent (survives scene transitions)
ctx.commands.spawn_in_scene(builder, "level_02");          // explicit scene
```

All return `TempEntity` — the entity materializes on phase flush (see CommandQueue above).

The default is current-scene ownership. If exactly one scene is loaded, `spawn()` uses it automatically. If multiple scenes are loaded, `spawn()` is a runtime error — the code must specify which scene via `spawn_in_scene()`. This keeps scene ownership explicit and avoids silent bugs. Persistence is the opt-in exception (like Unity's `DontDestroyOnLoad`), reserved for entities that must survive scene transitions: the player, HUD, music controller, global managers. Scene-loaded entities from `.rkscene` files are automatically tagged with their source scene.

## Blueprints

A blueprint is a reusable entity template — a saved configuration of components with default values. Stored as `.rkblueprint` RON files.

```ron
// blueprints/guard.rkblueprint
Blueprint(
    name: "Guard",
    components: {
        "Health": (current: 100.0, max: 100.0),
        "Patrol": (waypoints: [], speed: 2.5),
        "GuardState": "Patrolling",
        "EditorMetadata": (name: "Guard", tags: ["enemy", "npc"], locked: false),
    },
)
```

### Creation

In the editor: select an entity → right-click → "Save as Blueprint". The editor serializes all components except per-instance ones (`Transform`, `WorldTransform`, `Parent`, `StableId`, `SceneOwnership`) to a `.rkblueprint` file. This includes engine components like `SdfTree` (the entity's shape) and `EditorMetadata`, as well as all gameplay components.

### Instantiation

In the editor: "Create from Blueprint" menu lists all `.rkblueprint` files found in the project. Selecting one spawns a new entity with the blueprint's components, placing it at the camera's look-at point.

In code:
```rust
ctx.commands.spawn_blueprint("guard", position);
```

The `GameplayRegistry` holds a blueprint catalog loaded from the project's blueprint directory. Blueprints are plain data files — they reload when the file changes, no dylib involved.

### Asset Resolution

Blueprints that include an `SdfTree` reference asset files (e.g., `"guard.rkf"`). The serialized `SdfTree` stores the asset path, not a runtime `BrickMapHandle`. When the blueprint entity materializes at CommandQueue flush, the asset path is handed to the streaming system for loading — the same path that scene-loaded entities use. The entity exists in hecs immediately but is not renderable until the asset finishes loading and the `BrickMapHandle` is resolved. This is identical to how scene load works: entities appear in the world, and their geometry streams in asynchronously.

### Blueprint vs Instance

A blueprint is a template, not a linked prefab. Once spawned, the instance is independent — changing the blueprint file does not retroactively update existing instances. This avoids the complexity of Unity's prefab override system. A "re-apply blueprint" editor action can be added later if needed.

## Scene Graph and hecs World Model

### Unified Entity Model

All entities live in hecs — SDF objects, lights, cameras, fog volumes, and gameplay entities. There is no separate `Scene` or `SceneObject` data structure for rendering. The renderer queries hecs directly for the components it needs.

An entity's type is determined by its components:

| Entity type | Components |
|---|---|
| SDF object | `Transform` + `SdfTree` |
| Light | `Transform` + `Light` |
| Camera | `Transform` + `CameraComponent` |
| Fog volume | `Transform` + `FogVolumeComponent` |
| Empty (grouping) | `Transform` only |
| Gameplay-only | `Transform` + gameplay components |

Every entity automatically receives three engine-managed components at creation: `Transform` (using `WorldPosition`), `EditorMetadata` (name, tags, locked), and `StableId`. These are always present — the engine adds them at CommandQueue flush time if the user didn't include them. Defaults: `Transform` at origin with identity rotation and unit scale, `EditorMetadata` with an auto-generated name (e.g., "Entity_1") and empty tags, `StableId` with a fresh UUID. This applies uniformly — whether the entity was created by the editor, by `commands.spawn()`, or by `commands.spawn_blueprint()`. Beyond the three auto-components, components are mixed freely — an SDF object can also have `Health` and `Patrol`.

### Hierarchy

Parent-child relationships use the `Parent` component. One hierarchy system for everything — no separate object hierarchy and node hierarchy. The engine computes a `WorldTransform` component each frame by walking the hierarchy and composing local transforms — this is derived data, never serialized or authored directly.

**Cascading despawn:** Despawning an entity also despawns all of its descendants (children, grandchildren, etc.). This matches user expectations — deleting a "Guard" entity removes the guard and its child light, weapon, and particle emitter entities. Orphan prevention: the CommandQueue collects all descendants at flush time before despawning. Scene unload despawns by `SceneOwnership` tag, which catches all entities regardless of hierarchy, but cascading despawn ensures that reparented children of a despawned entity are also cleaned up.

### SdfTree Component

The SDF blend tree (previously `SceneNode`) lives inside a single `SdfTree` component:

```
SdfTree:
    root: SceneNode     // tree of SDF operations (union, smooth-min, subtract)
```

The tree is internal to the entity — it defines how the object's shape is composed from SDF primitives. Gameplay systems don't need to query or modify it. The renderer walks the tree during ray marching, same as before.

This replaces the old `SceneObject.root_node` field. The tree structure is unchanged — it's just stored as a component instead of a field on a separate data structure.

### Identity

- **Runtime:** `hecs::Entity` — fast, used in code and component references
- **Serialization:** `StableId(Uuid)` — persistent, used in scene files and save data
- **Display:** `EditorMetadata.name` — human-readable, unique among siblings

The old `u32` object IDs are eliminated.

### Scene File Format (`.rkscene` v3)

The scene file is the serialized form of the authored state. With the unified hecs model, the format is uniform — every entity is a StableId and a bag of named components:

```ron
Scene(
    version: 3,
    entities: [
        (
            stable_id: "a7f3b2c1-4d5e-6f78-9a0b-c1d2e3f4a5b6",
            parent: None,
            components: {
                "Transform": (position: (chunk: (0, 0, 0), local: (10.0, 0.0, 5.0)), rotation: (0.0, 0.383, 0.0, 0.924), scale: (1.0, 1.0, 1.0)),
                "SdfTree": (root: (name: "Body", sdf_source: Voxelized(asset: "guard.rkf", voxel_size: 0.1), children: [])),
                "EditorMetadata": (name: "Guard_01", tags: ["enemy", "npc"], locked: false),
                "Health": (current: 100.0, max: 100.0),
                "Patrol": (waypoints: [...], speed: 2.5),
            },
        ),
        (
            stable_id: "b8e4c3d2-5f6a-7b89-0c1d-e2f3a4b5c6d7",
            parent: Some("a7f3b2c1-4d5e-6f78-9a0b-c1d2e3f4a5b6"),
            components: {
                "Transform": (...),
                "Light": (kind: Point, color: (1.0, 0.8, 0.6), intensity: 5.0, range: 20.0),
                "EditorMetadata": (name: "TorchLight", tags: [], locked: false),
            },
        ),
    ],
)
```

No special cases for lights, cameras, SDF objects, or fog volumes — an entity's type is determined by which components it has. This replaces the current `.rkscene` v2 format which has separate sections for objects, lights, and cameras.

**Engine components** (`Transform`, `SdfTree`, `Light`, `CameraComponent`, `FogVolumeComponent`, `EditorMetadata`, `Parent`) are serialized/deserialized by the engine directly — it knows these types.

**Gameplay components** (`Health`, `Patrol`, user-defined types) are serialized/deserialized via the dylib's type-erased `ComponentEntry` function pointers. If a component type is no longer registered (e.g., the source file was deleted between sessions), its data is preserved as raw RON strings so it isn't lost on save. Re-adding the component type later restores the data.

**Parent references** use `StableId`, resolved to `hecs::Entity` on load via the stable-to-entity mapping.

### Edit World as Authored State

The edit world (the editor's `hecs::World`) IS the authored state. There is no separate data structure. The edit world is:

- **Loaded** from `.rkscene` on scene open (deserialize entities + components into hecs)
- **Modified** by the edit pipeline during edit mode (inspector, gizmo, sculpt, paint, MCP, undo/redo)
- **Saved** to `.rkscene` on save (serialize all entities + components from hecs)
- **Cloned** into a play world on Play (systems tick on the clone)
- **Never touched** by gameplay systems — it is always safe

The dylib must be loaded for the editor to function. Gameplay component types need to be known so they can be deserialized from the scene file and rendered in the inspector. On startup, the editor compiles the game crate, waits for the dylib to load, and only then proceeds to load the scene and present the UI. If no game crate exists (first-time project), the editor scaffolds a minimal one (empty `components/` and `systems/` directories, a stub `lib.rs` with `register()`) before compiling.

### Edit-Time Scene Model

**v1: single scene.** The editor opens one `.rkscene` at a time. All edit-world entities belong to that scene. "New Scene" clears the edit world. "Open Scene" replaces it. "Save" serializes the entire edit world to the open `.rkscene` file. `SceneOwnership` is present on all edit-world entities (tagged with the open scene's name), but in single-scene mode this is uniform.

**Future: multi-scene editing.** The data model supports loading multiple `.rkscene` files into the edit world simultaneously. Each entity is tagged with its source scene via `SceneOwnership`. The architecture is designed so that multi-scene editing is a UI/workflow enhancement, not an architectural change:

- **Active scene**: one scene is designated "active" in the editor — new entities are created in the active scene. A visible indicator (highlight in scene list, color in hierarchy panel) shows which scene is active. This parallels the runtime model (single scene → implicit, multiple → must be explicit).
- **Per-scene save**: the editor tracks dirty state per scene. "Save" saves the active scene. "Save All" saves all modified scenes. Each `.rkscene` file contains only its own entities.
- **Undo**: a single global undo stack (not per-scene). Undo operations cross scene boundaries — "undo the last thing I did" regardless of which scene it affected. This matches user expectations.
- **Scene tree UI**: the hierarchy panel groups or color-codes entities by source scene. Drag-and-drop between scenes re-tags `SceneOwnership`.
- **Cross-scene entity references**: **allowed but warned**. An entity in scene A can hold an `hecs::Entity` reference to an entity in scene B. The inspector shows a warning icon on cross-scene references, because they break if the target scene is not loaded. At runtime, the same warning applies — a cross-scene reference is only valid while both scenes are loaded. This is a deliberate choice: forbidding cross-scene references is too restrictive (a player entity in the persistent scene needs to reference level objects), but silent breakage is unacceptable.

### Prerequisites

The unified hecs world model requires foundational work before the behavior system can be built:

1. **Migrate renderer to hecs** — replace `Scene`/`SceneObject` queries with hecs component queries (`Transform` + `SdfTree`, `Transform` + `Light`, etc.)
2. **New `.rkscene` v3 format** — implement the entity + component bag format, with backward-compatible loading of v2 scenes
3. **Editor save/load rework** — read/write the new format, serialize/deserialize the edit world directly
4. **Edit pipeline** — unified entry point for all edit operations (inspector, gizmo, sculpt, paint, MCP, undo/redo) that writes to the edit world and pokes rinch signals
5. **StableId system** — assign StableIds on entity creation, maintain bidirectional mappings, serialize entity references as StableIds
6. **Dylib infrastructure** — the editor requires a loaded game dylib to function. Gameplay component types must be known for scene loading and inspector rendering. On startup: compile game crate → load dylib → then load scene.

These are not optional — the behavior system (hot-reload, Play/Stop, component inspection, scene save/load) depends on all of them.

### Edit Pipeline

All modifications to entity data during edit mode — regardless of source — go through a unified edit pipeline:

```
Edit source (inspector, gizmo, sculpt, paint, MCP, undo/redo)
    → write to edit world (hecs — renderer sees it immediately)
    → poke rinch signal (UI updates reactively)
```

This is bidirectional. Inspector edits originate from the UI panel. Gizmo, sculpt, and paint edits originate from the viewport. Both flow through the same pipeline, ensuring the edit world and signals stay in sync.

The edit pipeline is the single point of coordination. Individual tools (gizmo, sculpt, inspector) do not write to hecs directly — they submit edits to the pipeline. This is also the undo/redo capture point — the pipeline records changes for undo before applying them.

### Undo/Redo

The editor maintains a **single global undo stack** — a linear sequence of `UndoAction`s. Every operation that flows through the edit pipeline pushes an action before applying. Ctrl+Z pops and reverses the most recent action. Ctrl+Shift+Z re-applies. A new edit clears the redo stack.

**Action types:**

| Action | Undo captures | Undo restores |
|---|---|---|
| `PropertyChange` | entity, component name, field name, old `GameValue` | Writes old value via `set_field` |
| `ComponentAdd` | entity, component name, initial RON data | Removes the component |
| `ComponentRemove` | entity, component name, removed component's RON | Re-inserts from RON via `deserialize_insert` |
| `EntitySpawn` | entity, all components as RON + metadata | Despawns the entity |
| `EntityDespawn` | entity, all components as RON + metadata | Re-spawns with same StableId and components |
| `GeometryEdit` | entity, affected brick geometry snapshots | Restores brick data from snapshots |

**Coalescing:** Continuous interactions (gizmo drags, slider scrubs, paint strokes) produce many small edits per frame. These coalesce into a single undo action — begin on interaction start (mouse-down), finalize on interaction end (mouse-up). Undoing a gizmo drag restores the position from before the drag started, not an intermediate frame.

**Scope:**
- The undo stack is global, not per-entity or per-scene. In multi-scene editing, undo crosses scene boundaries.
- Undo is an **edit-mode operation**. Edit tools (inspector, gizmo, sculpt, paint) are disabled during play, so they generate no undo actions. The one exception is **push to edit** — it writes to the edit world via the edit pipeline and IS undoable. The stack is otherwise untouched by Play/Stop.
- The undo stack is **cleared on scene load** — opening a new scene starts a fresh undo context.
- Undo uses the same type-erased infrastructure as hot-reload and scene save (RON serialization, `get_field`/`set_field`, `ComponentEntry` function pointers). No separate reflection system.

### Threading Model

The system scheduler is **single-threaded** in v1. Systems execute sequentially within each phase — one `&mut SystemContext` at a time. This is simple, deterministic, and sufficient for the initial behavior system.

Parallel system execution (systems that don't share components running concurrently) is a future enhancement. The `after`/`before` dependency graph already encodes the information needed for a parallel scheduler — no system API changes would be required.

## Editor Integration

### Inspector

The right panel shows all components on the selected entity, rendered from `ComponentMeta`:

```
┌─ Guard_01 ──────────────────────────┐
│ Transform                           │
│   Position: [10.0, 0.0, 5.0]       │
│   Rotation: [0, 45, 0]             │
│   Scale:    [1, 1, 1]              │
│                                     │
│ Health                  [- Remove]  │
│   Current: ████████░░ 80.0 / 100.0 │
│   Max:     100.0                    │
│                                     │
│ Patrol                  [- Remove]  │
│   Speed:    2.5                     │
│   Waypoints: [3 points]            │
│                                     │
│ [+ Add Component ▾]                │
│   ┌────────────────┐               │
│   │ Collectible    │               │
│   │ DoorState      │               │
│   │ GuardState     │ ← from        │
│   └────────────────┘   registry    │
└─────────────────────────────────────┘
```

- **Add Component** dropdown lists all registered components minus those already on the entity
- **Remove** button detaches the component from the entity
- Numeric fields render as sliders (using `FieldMeta.range`) or text inputs
- Enum fields render as dropdowns
- Entity reference fields render as a picker (click to select target entity)
- Transient fields are hidden in edit mode, visible in play mode

### Compiler Integration

The editor owns the compilation process:

- Background thread watches the game logic crate directory using the `notify` crate
- On `.rs` file change: spawn `cargo build -p <game-crate> --lib` as a child process
- Status bar shows compilation state: idle → "Compiling..." → "Reloaded" or "Error: ..."
- Compile errors are shown in an editor console panel with file/line references
- The editor remains fully interactive during compilation — editing, camera movement, and play mode continue uninterrupted (reload is queued until edit mode)

### Systems Panel

A debug panel lists all registered systems, their phases, and per-system frame time:

```
┌─ Systems ───────────────────────────┐
│ Phase: Update                       │
│   patrol_system        0.02ms       │
│   guard_ai_system      0.05ms       │
│   door_system          0.01ms       │
│   death_system         0.01ms       │
│                                     │
│ Phase: LateUpdate                   │
│   camera_follow_system 0.01ms       │
│                                     │
│ Total gameplay:        0.10ms       │
└─────────────────────────────────────┘
```

### Engine Components

The existing ECS components in `rkf-runtime` (`Transform`, `CameraComponent`, `FogVolumeComponent`, `Parent`, `EditorMetadata`) predate this behavior system and were implemented as proof-of-concept scaffolding. They need to be reworked for the unified hecs world model:

- All components that participate in scene save/load and Play/Stop must derive `Serialize + Deserialize` (requires enabling the `serde` feature on `glam`)
- Derived/recomputed components like `WorldTransform` should not be serialized — they are regenerated each frame
- Engine components should implement `ComponentMeta` so the inspector can display them uniformly alongside gameplay components
- The `Scene`/`SceneObject` data structure is replaced by hecs entities with `Transform` + `SdfTree` components — the renderer queries hecs directly
- Dead code and unused placeholder implementations should be removed

### Engine Component Registration

Engine components (`Transform`, `Light`, `SdfTree`, `CameraComponent`, `FogVolumeComponent`, `EditorMetadata`, `Parent`, etc.) use the same `#[component]` attribute macro and `ComponentMeta` trait as gameplay components. The difference is registration timing and source:

- **Engine components**: registered by an `engine_register()` step that runs at editor startup, before any dylib is loaded. These are compiled into the static binary.
- **Gameplay components**: registered by the dylib's `register()` call after the dylib loads.

Both populate the same `GameplayRegistry`. The inspector, MCP tools, and scene serialization treat engine and gameplay components identically — they share the `ComponentMeta` interface.

New engine components follow the same pattern: annotate with `#[component]`, and they're automatically picked up by `engine_register()` via `inventory`. No special registration code needed.

## UI Integration

In-game UI (HUD, menus, dialogue, inventory) is rendered via **rinch**, composited over the engine viewport. Rinch is signal-based and reactive — UI elements bind to signals and update automatically when the underlying data changes. The UI is not an ECS system; it is a separate reactive layer.

The game state store is the primary bridge between gameplay systems and UI:

- **Gameplay → UI**: Systems write to the store (`ctx.game.set("player/health", 85.0)`). The engine pushes store values into rinch signals. Bound UI elements update reactively — no polling, no manual refresh.
- **UI → Gameplay**: UI interactions (button clicks, slider changes) write to rinch signals. The engine reads these signals before the next frame's system tick and injects the values into the game state store or event queue. From the gameplay system's perspective, UI-originated data appears as normal store values and events — systems read them via `ctx.game.get()` and `ctx.game.events()` like any other state. The UI layer never calls `ctx.game` directly — it communicates through rinch signals, and the engine bridges the gap.

Component data that needs UI display follows the same pattern — a LateUpdate system reads relevant components and writes summary values to the store, which rinch signals pick up.

The full UI system architecture (layout, widgets, styling, input routing) is specified in a separate document. This section defines only the data flow boundary between the behavior system and the UI layer.

## MCP Integration

The behavior system extends the existing MCP automation API:

### New MCP Tools

| Tool | Description |
|------|-------------|
| `component_list` | List all registered components (engine + gameplay) with field metadata |
| `component_get` | Read a specific component's values on an entity |
| `component_set` | Write fields on a component (partial update) |
| `component_add` | Add a component to an entity with specified values |
| `component_remove` | Remove a component from an entity |
| `system_list` | List registered systems, phases, and timing |
| `blueprint_list` | List available blueprints |
| `blueprint_spawn` | Instantiate a blueprint at a position |
| `state_get` | Read a value from the game state store by key |
| `state_set` | Write a value to the game state store |
| `state_list` | List all keys (optionally filtered by namespace prefix) |
| `play_start` | Enter play mode |
| `play_stop` | Exit play mode (revert to authored state) |

These tools let AI agents inspect and manipulate gameplay state at runtime — discovering what behaviors exist, tweaking component values, managing game state, and testing interactions without touching the editor UI.

## Future: JavaScript Scripting Layer

The architecture supports adding an embedded JavaScript runtime without redesign. The key insight: JS doesn't replace the ECS — it's another way to author systems.

A `JsSystem` wrapper (using `deno_core`, a Rust crate wrapping V8) loads a `.js` file, exposes component data as JS objects through the same `ComponentMeta` reflection layer, and calls the script's `update()` function each frame. The scheduler doesn't know or care whether a system is Rust or JS — both go through the same phase scheduling.

Components remain Rust structs in hecs. JS reads and writes their fields through a binding layer built on `ComponentMeta`. This means:
- No new reflection system — JS uses what the editor inspector and MCP tools already use
- Rust and JS systems interleave freely in the phase schedule
- Hot-reload of `.js` files would be near-instant (no compilation step)

Tradeoffs: V8 adds ~20MB to binary size and a `deno_core` dependency. JS systems would be ~10-100x slower than Rust for math-heavy code, but suitable for prototyping, level scripting, and mod support.

This is a future enhancement, not a v1 requirement. The v1 behavior system is Rust-only.

## User Workflow Summary

### Writing a New Behavior

1. Create `components/collectible.rs` in the game crate
2. Annotate the struct with `#[component]`
3. Create `systems/collectible_system.rs`
4. Write the system function with `#[system(phase = Update)]`
5. Save — editor auto-recompiles and hot-reloads
6. Select an entity → Add Component → "Collectible" appears in the menu
7. Configure fields in the inspector
8. Hit Play to test

### Testing a Behavior

1. Add component to entity in inspector
2. Set field values
3. Hit Play — systems tick, behavior runs
4. Tweak values in play UI — use "push to edit" to preserve values that feel right
5. Observe results in viewport
6. Edit code, save — build runs in background, reload queued
7. Hit Stop — play world discarded, queued reload applies automatically
8. Hit Play again to test with new code

### Creating a Blueprint

1. Set up an entity with the desired components and values
2. Right-click → Save as Blueprint → name it "Guard"
3. Later: Create from Blueprint → "Guard" → places a configured entity
