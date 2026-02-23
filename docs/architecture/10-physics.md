> **SUPERSEDED** by [v2 Architecture](../v2/ARCHITECTURE.md) — this document describes the v1 chunk-based engine.

# Physics

> **Status: DECIDED**

### Decision: Rapier + SDF Collision Adapter

**Chosen over:** Rapier with extracted mesh colliders (lags behind edits, loses SDF elegance), custom physics engine (huge scope — constraint solving, stacking, friction are hard problems).

Rapier handles dynamics: integration, broadphase, constraint solving, joints, raycasting. We write an adapter that makes our SDF world queryable as a Rapier collider for world-vs-object contacts.

**Architecture:**
```
┌──────────────┐     ┌──────────────────┐     ┌──────────────┐
│ Rapier World │◄───►│ SDF Collision     │◄───►│ Brick Pool + │
│ (dynamics)   │     │ Adapter           │     │ Spatial Index │
└──────────────┘     └──────────────────┘     └──────────────┘
       │
       ├── RigidBody (debris, props, ragdoll)
       ├── Collider (box, sphere, capsule — standard Rapier shapes)
       └── Joints (ragdoll constraints)
```

**SDF collision adapter — contact generation:**
```rust
/// Called by Rapier's collision pipeline for each body near SDF terrain
fn generate_sdf_contacts(body: &RigidBody, sdf: &SpatialIndex) -> Vec<ContactPoint> {
    let mut contacts = Vec::new();
    // Sample SDF at body's contact sample points
    for sample_pos in body.contact_samples() {
        let distance = sdf.evaluate(sample_pos);
        if distance < contact_threshold {
            let normal = sdf.gradient(sample_pos);  // central differences
            contacts.push(ContactPoint {
                position: sample_pos,
                normal,
                penetration: -distance,
            });
        }
    }
    contacts
}
```

**Object-vs-object:** Standard Rapier — primitive colliders (box, sphere, capsule) on rigid bodies. No SDF involvement.

**Object-vs-world:** SDF adapter. Each rigid body has sample points (vertices of its collision shape). SDF evaluated at each sample point for penetration.

### Decision: Custom SDF Character Controller

**Chosen over:** Rapier's built-in character controller (designed for mesh colliders, adapting to SDF adds complexity without benefit).

A dedicated controller that directly queries the SDF for movement resolution:

```rust
struct SdfCharacterController {
    capsule_radius: f32,
    capsule_height: f32,
    ground_snap_distance: f32,
    max_slope_angle: f32,
    step_height: f32,

    // State
    grounded: bool,
    ground_normal: Vec3,
    velocity: Vec3,
}

impl SdfCharacterController {
    fn move_and_slide(&mut self, desired: Vec3, sdf: &SpatialIndex, dt: f32) -> Vec3 {
        let mut position = self.position;
        let mut remaining = desired * dt;

        // Iterative slide (up to 4 iterations for corners)
        for _ in 0..4 {
            if remaining.length_squared() < 1e-6 { break; }

            // Sample SDF at capsule bottom, middle, top
            let samples = self.capsule_samples(position + remaining);
            let deepest = samples.iter()
                .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

            if let Some(contact) = deepest.filter(|c| c.distance < 0.0) {
                // Push out of surface
                position += contact.normal * (-contact.distance + SKIN_WIDTH);
                // Slide: remove velocity component into the surface
                remaining -= contact.normal * remaining.dot(contact.normal);
            } else {
                position += remaining;
                break;
            }
        }

        // Ground check
        let foot_distance = sdf.evaluate(position - Vec3::Y * self.capsule_height * 0.5);
        self.grounded = foot_distance < self.ground_snap_distance;
        if self.grounded {
            self.ground_normal = sdf.gradient(position - Vec3::Y * self.capsule_height * 0.5);
        }

        position
    }
}
```

**Features:**
- Capsule-vs-SDF collision with multi-sample coverage
- Iterative slide along surfaces (handles corners and wedges)
- Slope limiting (steep slopes become walls)
- Step climbing (small ledges auto-mounted)
- Ground snapping (stay grounded on slopes and stairs)

### Decision: Destruction ↔ Physics — Lazy SDF Re-Query

When CSG operations modify the world (see [Procedural Editing](./07-procedural-editing.md)), the physics system doesn't need explicit notification. Every contact check evaluates the SDF fresh — if terrain was removed, the SDF returns positive distance, and the contact disappears naturally.

This works because:
- SDF queries are cheap (one spatial index lookup + trilinear interpolation)
- Rapier re-checks contacts every physics step anyway
- No cached collision geometry to invalidate

**Debris spawning from destruction:**
When a CSG subtract occurs (explosion, mining), the edit system emits a `DestructionEvent { aabb, volume, material_id }`. A debris spawner creates SDF micro-object particles with the destroyed material, initial velocity outward from the impact point.

### Decision: Physics Timing — Fixed Timestep

```rust
const PHYSICS_HZ: f32 = 60.0;
const PHYSICS_DT: f32 = 1.0 / PHYSICS_HZ;

fn update_physics(elapsed: f32, rapier: &mut RapierContext, sdf: &SpatialIndex) {
    let mut accumulator = elapsed;
    while accumulator >= PHYSICS_DT {
        rapier.step(PHYSICS_DT);
        character_controller.update(PHYSICS_DT, sdf);
        accumulator -= PHYSICS_DT;
    }
    // Remaining fraction: interpolate render transforms
}
```

Fixed 60Hz physics step, decoupled from render frame rate. Interpolated transforms for smooth rendering at any frame rate.

### Session 10b Summary: Physics Decisions

| Decision | Choice | Notes |
|----------|--------|-------|
| Physics engine | Rapier (Rust-native) | Dynamics, broadphase, joints, raycasting |
| World collision | SDF collision adapter | Direct SDF queries for contact generation |
| Character controller | Custom SDF capsule controller | Slide, slope limit, step climb, ground snap |
| Object-vs-object | Standard Rapier primitives | Box, sphere, capsule colliders |
| Destruction sync | Lazy SDF re-query | No invalidation needed |
| Debris spawning | DestructionEvent → particle emitter | SDF micro-object particles |
| Timestep | Fixed 60Hz, interpolated rendering | Decoupled from frame rate |
