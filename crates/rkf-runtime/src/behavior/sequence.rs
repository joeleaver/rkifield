//! Sequence component — timed multi-step entity actions.
//!
//! Sequences handle frame-spanning actions like door animations, death effects,
//! and cutscene steps. They complement state machines: the state machine decides
//! WHAT to do, the sequence handles HOW (the timed execution).

use glam::{Quat, Vec3};
use rkf_core::WorldPosition;
use serde::{Deserialize, Serialize};

use super::game_value::GameValue;

// hecs::Entity doesn't implement Serialize/Deserialize. Entity references in
// sequences (EmitFrom.source) are serialized as u64 bits for now — proper
// StableId remapping is added in Phase 3 when the serialize infrastructure
// handles Entity fields across the board.
mod entity_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize_opt_entity<S: Serializer>(
        entity: &Option<hecs::Entity>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        match entity {
            Some(e) => e.to_bits().serialize(s),
            None => Option::<u64>::None.serialize(s),
        }
    }

    pub fn deserialize_opt_entity<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<hecs::Entity>, D::Error> {
        let bits: Option<u64> = Option::deserialize(d)?;
        Ok(bits.and_then(|b| hecs::Entity::from_bits(b)))
    }
}

// ─── Easing ───────────────────────────────────────────────────────────────

/// Easing function for interpolated sequence steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ease {
    /// No easing — constant rate.
    Linear,
    /// Quadratic ease-in.
    InQuad,
    /// Quadratic ease-out.
    OutQuad,
    /// Quadratic ease-in-out.
    InOutQuad,
    /// Cubic ease-in.
    InCubic,
    /// Cubic ease-out.
    OutCubic,
    /// Cubic ease-in-out.
    InOutCubic,
    /// Exponential ease-in.
    InExpo,
    /// Exponential ease-out.
    OutExpo,
    /// Bounce ease-out.
    OutBounce,
    /// Elastic ease-out.
    OutElastic,
    /// Overshoot ease-out.
    OutBack,
}

impl Default for Ease {
    fn default() -> Self {
        Ease::Linear
    }
}

impl Ease {
    /// Evaluate the easing function at `t` (0.0 to 1.0).
    pub fn eval(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Ease::Linear => t,
            Ease::InQuad => t * t,
            Ease::OutQuad => 1.0 - (1.0 - t) * (1.0 - t),
            Ease::InOutQuad => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
                }
            }
            Ease::InCubic => t * t * t,
            Ease::OutCubic => 1.0 - (1.0 - t).powi(3),
            Ease::InOutCubic => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
                }
            }
            Ease::InExpo => {
                if t == 0.0 {
                    0.0
                } else {
                    (2.0_f32).powf(10.0 * t - 10.0)
                }
            }
            Ease::OutExpo => {
                if t == 1.0 {
                    1.0
                } else {
                    1.0 - (2.0_f32).powf(-10.0 * t)
                }
            }
            Ease::OutBounce => {
                let n1 = 7.5625;
                let d1 = 2.75;
                if t < 1.0 / d1 {
                    n1 * t * t
                } else if t < 2.0 / d1 {
                    let t = t - 1.5 / d1;
                    n1 * t * t + 0.75
                } else if t < 2.5 / d1 {
                    let t = t - 2.25 / d1;
                    n1 * t * t + 0.9375
                } else {
                    let t = t - 2.625 / d1;
                    n1 * t * t + 0.984375
                }
            }
            Ease::OutElastic => {
                if t == 0.0 || t == 1.0 {
                    t
                } else {
                    let c4 = (2.0 * std::f32::consts::PI) / 3.0;
                    (2.0_f32).powf(-10.0 * t) * ((t * 10.0 - 0.75) * c4).sin() + 1.0
                }
            }
            Ease::OutBack => {
                let c1 = 1.70158;
                let c3 = c1 + 1.0;
                let t1 = t - 1.0;
                1.0 + c3 * t1 * t1 * t1 + c1 * t1 * t1
            }
        }
    }
}

// ─── Sequence steps ───────────────────────────────────────────────────────

/// A single step in a sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SequenceStep {
    /// Pause for a duration.
    Wait {
        /// Duration in seconds.
        duration: f32,
    },
    /// Lerp position to an absolute target.
    MoveTo {
        /// Target position.
        target: WorldPosition,
        /// Duration in seconds.
        duration: f32,
        /// Easing function.
        ease: Ease,
    },
    /// Lerp position by a relative offset.
    MoveBy {
        /// Offset vector.
        offset: Vec3,
        /// Duration in seconds.
        duration: f32,
        /// Easing function.
        ease: Ease,
    },
    /// Slerp rotation to an absolute target.
    RotateTo {
        /// Target rotation.
        target: Quat,
        /// Duration in seconds.
        duration: f32,
        /// Easing function.
        ease: Ease,
    },
    /// Slerp rotation by a relative amount.
    RotateBy {
        /// Relative rotation.
        rotation: Quat,
        /// Duration in seconds.
        duration: f32,
        /// Easing function.
        ease: Ease,
    },
    /// Lerp scale to an absolute target.
    ScaleTo {
        /// Target scale.
        target: Vec3,
        /// Duration in seconds.
        duration: f32,
        /// Easing function.
        ease: Ease,
    },
    /// Fire-and-forget event (source = this entity, no payload).
    Emit {
        /// Event name.
        name: String,
    },
    /// Event with data payload (source = this entity).
    EmitWith {
        /// Event name.
        name: String,
        /// Event payload.
        data: GameValue,
    },
    /// Event with custom source entity and optional payload.
    EmitFrom {
        /// Event name.
        name: String,
        /// Source entity (None for UI-originated events).
        #[serde(
            serialize_with = "entity_serde::serialize_opt_entity",
            deserialize_with = "entity_serde::deserialize_opt_entity"
        )]
        source: Option<hecs::Entity>,
        /// Optional payload.
        data: Option<GameValue>,
    },
    /// Write a persistent value to the game state store.
    SetState {
        /// Store key.
        key: String,
        /// Value to write.
        value: GameValue,
    },
    /// Instantiate a blueprint at this entity's position.
    SpawnBlueprint {
        /// Blueprint name (looked up in GameplayRegistry).
        name: String,
    },
    /// Remove this entity.
    Despawn,
    /// Repeat a sub-sequence N times.
    Repeat {
        /// Number of repetitions.
        count: usize,
        /// Steps to repeat.
        steps: Vec<SequenceStep>,
    },
}

impl SequenceStep {
    /// Duration of this step in seconds. Zero for instant steps.
    pub fn duration(&self) -> f32 {
        match self {
            SequenceStep::Wait { duration } => *duration,
            SequenceStep::MoveTo { duration, .. } => *duration,
            SequenceStep::MoveBy { duration, .. } => *duration,
            SequenceStep::RotateTo { duration, .. } => *duration,
            SequenceStep::RotateBy { duration, .. } => *duration,
            SequenceStep::ScaleTo { duration, .. } => *duration,
            // Instant steps:
            SequenceStep::Emit { .. }
            | SequenceStep::EmitWith { .. }
            | SequenceStep::EmitFrom { .. }
            | SequenceStep::SetState { .. }
            | SequenceStep::SpawnBlueprint { .. }
            | SequenceStep::Despawn => 0.0,
            // Repeat: sum of inner steps × count
            SequenceStep::Repeat { count, steps } => {
                let inner: f32 = steps.iter().map(|s| s.duration()).sum();
                inner * (*count as f32)
            }
        }
    }
}

// ─── Sequence component ───────────────────────────────────────────────────

/// Timed multi-step action runner. Attach to an entity to execute a sequence
/// of steps over time. Removed automatically when all steps complete.
///
/// ```ignore
/// Sequence::new()
///     .move_to(open_pos, 1.0)
///     .wait(3.0)
///     .move_to(closed_pos, 1.0)
///     .emit("door_closed")
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sequence {
    /// The steps to execute in order.
    pub steps: Vec<SequenceStep>,
    /// Index of the currently executing step.
    pub current: usize,
    /// Elapsed time within the current step.
    pub timer: f32,
}

impl Sequence {
    /// Create a new empty sequence builder.
    pub fn new() -> SequenceBuilder {
        SequenceBuilder { steps: Vec::new() }
    }

    /// Returns true if all steps have been executed.
    pub fn is_complete(&self) -> bool {
        self.current >= self.steps.len()
    }

    /// Total duration of all steps in seconds.
    pub fn total_duration(&self) -> f32 {
        self.steps.iter().map(|s| s.duration()).sum()
    }
}

impl Default for Sequence {
    fn default() -> Self {
        Self {
            steps: Vec::new(),
            current: 0,
            timer: 0.0,
        }
    }
}

// ─── Builder ──────────────────────────────────────────────────────────────

/// Fluent builder for constructing sequences.
pub struct SequenceBuilder {
    steps: Vec<SequenceStep>,
}

impl SequenceBuilder {
    /// Pause for `seconds`.
    pub fn wait(mut self, seconds: f32) -> Self {
        self.steps.push(SequenceStep::Wait {
            duration: seconds,
        });
        self
    }

    /// Lerp position to an absolute target over `duration` seconds.
    pub fn move_to(mut self, target: impl Into<WorldPosition>, duration: f32) -> Self {
        self.steps.push(SequenceStep::MoveTo {
            target: target.into(),
            duration,
            ease: Ease::Linear,
        });
        self
    }

    /// Lerp position by a relative offset over `duration` seconds.
    pub fn move_by(mut self, offset: Vec3, duration: f32) -> Self {
        self.steps.push(SequenceStep::MoveBy {
            offset,
            duration,
            ease: Ease::Linear,
        });
        self
    }

    /// Slerp rotation to an absolute target over `duration` seconds.
    pub fn rotate_to(mut self, target: Quat, duration: f32) -> Self {
        self.steps.push(SequenceStep::RotateTo {
            target,
            duration,
            ease: Ease::Linear,
        });
        self
    }

    /// Slerp rotation by a relative amount over `duration` seconds.
    pub fn rotate_by(mut self, rotation: Quat, duration: f32) -> Self {
        self.steps.push(SequenceStep::RotateBy {
            rotation,
            duration,
            ease: Ease::Linear,
        });
        self
    }

    /// Lerp scale to an absolute target over `duration` seconds.
    pub fn scale_to(mut self, target: Vec3, duration: f32) -> Self {
        self.steps.push(SequenceStep::ScaleTo {
            target,
            duration,
            ease: Ease::Linear,
        });
        self
    }

    /// Fire-and-forget event (source = this entity, no payload).
    pub fn emit(mut self, name: &str) -> Self {
        self.steps.push(SequenceStep::Emit {
            name: name.to_owned(),
        });
        self
    }

    /// Event with data payload (source = this entity).
    pub fn emit_with(mut self, name: &str, data: impl Into<GameValue>) -> Self {
        self.steps.push(SequenceStep::EmitWith {
            name: name.to_owned(),
            data: data.into(),
        });
        self
    }

    /// Event with custom source and optional payload.
    pub fn emit_from(
        mut self,
        name: &str,
        source: Option<hecs::Entity>,
        data: Option<GameValue>,
    ) -> Self {
        self.steps.push(SequenceStep::EmitFrom {
            name: name.to_owned(),
            source,
            data,
        });
        self
    }

    /// Write a persistent value to the game state store.
    pub fn set_state(mut self, key: &str, value: impl Into<GameValue>) -> Self {
        self.steps.push(SequenceStep::SetState {
            key: key.to_owned(),
            value: value.into(),
        });
        self
    }

    /// Instantiate a blueprint at this entity's position.
    pub fn spawn_blueprint(mut self, name: &str) -> Self {
        self.steps.push(SequenceStep::SpawnBlueprint {
            name: name.to_owned(),
        });
        self
    }

    /// Remove this entity.
    pub fn despawn(mut self) -> Self {
        self.steps.push(SequenceStep::Despawn);
        self
    }

    /// Repeat a sub-sequence N times.
    pub fn repeat(mut self, count: usize, sub: impl FnOnce(SequenceBuilder) -> SequenceBuilder) -> Self {
        let inner = sub(SequenceBuilder { steps: Vec::new() });
        self.steps.push(SequenceStep::Repeat {
            count,
            steps: inner.steps,
        });
        self
    }

    /// Set the easing function on the most recently added step.
    ///
    /// Only affects lerp steps (MoveTo, MoveBy, RotateTo, RotateBy, ScaleTo).
    /// Has no effect on instant steps (Wait, Emit, etc.).
    pub fn ease(mut self, ease: Ease) -> Self {
        if let Some(last) = self.steps.last_mut() {
            match last {
                SequenceStep::MoveTo { ease: e, .. }
                | SequenceStep::MoveBy { ease: e, .. }
                | SequenceStep::RotateTo { ease: e, .. }
                | SequenceStep::RotateBy { ease: e, .. }
                | SequenceStep::ScaleTo { ease: e, .. } => {
                    *e = ease;
                }
                _ => {} // no-op for non-lerp steps
            }
        }
        self
    }

    /// Build the sequence.
    pub fn build(self) -> Sequence {
        Sequence {
            steps: self.steps,
            current: 0,
            timer: 0.0,
        }
    }
}

/// Allow using `SequenceBuilder` directly where `Sequence` is expected.
impl From<SequenceBuilder> for Sequence {
    fn from(builder: SequenceBuilder) -> Self {
        builder.build()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{IVec3, Quat, Vec3};
    use std::f32::consts::PI;

    #[test]
    fn ease_linear() {
        assert!((Ease::Linear.eval(0.0) - 0.0).abs() < 1e-6);
        assert!((Ease::Linear.eval(0.5) - 0.5).abs() < 1e-6);
        assert!((Ease::Linear.eval(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ease_boundaries() {
        // All easing functions should map 0→0 and 1→1
        let eases = [
            Ease::Linear,
            Ease::InQuad,
            Ease::OutQuad,
            Ease::InOutQuad,
            Ease::InCubic,
            Ease::OutCubic,
            Ease::InOutCubic,
            Ease::InExpo,
            Ease::OutExpo,
            Ease::OutBounce,
            Ease::OutElastic,
            Ease::OutBack,
        ];
        for ease in &eases {
            assert!(
                (ease.eval(0.0) - 0.0).abs() < 1e-4,
                "{:?} at 0.0 = {}",
                ease,
                ease.eval(0.0)
            );
            assert!(
                (ease.eval(1.0) - 1.0).abs() < 1e-4,
                "{:?} at 1.0 = {}",
                ease,
                ease.eval(1.0)
            );
        }
    }

    #[test]
    fn ease_out_quad_midpoint() {
        // OutQuad at 0.5 = 1 - (1-0.5)^2 = 1 - 0.25 = 0.75
        assert!((Ease::OutQuad.eval(0.5) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn ease_clamps_input() {
        assert!((Ease::Linear.eval(-1.0) - 0.0).abs() < 1e-6);
        assert!((Ease::Linear.eval(2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn builder_basic_sequence() {
        let seq = Sequence::new()
            .wait(1.0)
            .move_by(Vec3::Y * 5.0, 2.0)
            .emit("done")
            .build();

        assert_eq!(seq.steps.len(), 3);
        assert!(!seq.is_complete());
        assert!((seq.total_duration() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn builder_door_sequence() {
        let open_pos = WorldPosition {
            chunk: IVec3::ZERO,
            local: Vec3::new(0.0, 3.0, 0.0),
        };
        let closed_pos = WorldPosition::default();

        let seq = Sequence::new()
            .move_to(open_pos, 1.0)
            .wait(3.0)
            .move_to(closed_pos, 1.0)
            .emit("door_closed")
            .build();

        assert_eq!(seq.steps.len(), 4);
        assert!((seq.total_duration() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn builder_ease_modifies_last_step() {
        let seq = Sequence::new()
            .move_to(WorldPosition::default(), 1.0)
            .ease(Ease::OutBounce)
            .build();

        match &seq.steps[0] {
            SequenceStep::MoveTo { ease, .. } => assert_eq!(*ease, Ease::OutBounce),
            _ => panic!("expected MoveTo"),
        }
    }

    #[test]
    fn builder_ease_no_op_on_instant_step() {
        let seq = Sequence::new()
            .emit("test")
            .ease(Ease::OutBounce) // should be silently ignored
            .build();

        match &seq.steps[0] {
            SequenceStep::Emit { name } => assert_eq!(name, "test"),
            _ => panic!("expected Emit"),
        }
    }

    #[test]
    fn builder_repeat() {
        let seq = Sequence::new()
            .repeat(3, |s| {
                s.set_state("fx/flash", true)
                    .wait(0.15)
                    .set_state("fx/flash", false)
                    .wait(0.15)
            })
            .despawn()
            .build();

        assert_eq!(seq.steps.len(), 2); // Repeat + Despawn
        match &seq.steps[0] {
            SequenceStep::Repeat { count, steps } => {
                assert_eq!(*count, 3);
                assert_eq!(steps.len(), 4);
            }
            _ => panic!("expected Repeat"),
        }
    }

    #[test]
    fn builder_emit_variants() {
        let seq = Sequence::new()
            .emit("simple")
            .emit_with("data", 42.0_f32)
            .emit_from("custom", None, Some(GameValue::Bool(true)))
            .build();

        assert_eq!(seq.steps.len(), 3);
        match &seq.steps[1] {
            SequenceStep::EmitWith { name, data } => {
                assert_eq!(name, "data");
                assert_eq!(*data, GameValue::Float(42.0));
            }
            _ => panic!("expected EmitWith"),
        }
    }

    #[test]
    fn builder_into_sequence() {
        // SequenceBuilder can be used where Sequence is expected via Into
        let builder = Sequence::new().wait(1.0);
        let seq: Sequence = builder.into();
        assert_eq!(seq.steps.len(), 1);
    }

    #[test]
    fn step_duration() {
        assert!((SequenceStep::Wait { duration: 2.0 }.duration() - 2.0).abs() < 1e-6);
        assert!(
            (SequenceStep::MoveTo {
                target: WorldPosition::default(),
                duration: 1.5,
                ease: Ease::Linear,
            }
            .duration()
                - 1.5)
                .abs()
                < 1e-6
        );
        assert_eq!(
            SequenceStep::Emit {
                name: "x".into()
            }
            .duration(),
            0.0
        );
        assert_eq!(SequenceStep::Despawn.duration(), 0.0);
    }

    #[test]
    fn repeat_duration() {
        let step = SequenceStep::Repeat {
            count: 3,
            steps: vec![
                SequenceStep::Wait { duration: 0.1 },
                SequenceStep::Wait { duration: 0.2 },
            ],
        };
        assert!((step.duration() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn sequence_default_is_empty() {
        let seq = Sequence::default();
        assert!(seq.steps.is_empty());
        assert!(seq.is_complete());
        assert_eq!(seq.current, 0);
        assert_eq!(seq.timer, 0.0);
    }

    #[test]
    fn rotate_steps() {
        let seq = Sequence::new()
            .rotate_to(Quat::from_rotation_y(PI), 1.0)
            .rotate_by(Quat::from_rotation_x(PI / 2.0), 0.5)
            .ease(Ease::InOutCubic)
            .build();

        assert_eq!(seq.steps.len(), 2);
        match &seq.steps[1] {
            SequenceStep::RotateBy { ease, .. } => assert_eq!(*ease, Ease::InOutCubic),
            _ => panic!("expected RotateBy"),
        }
    }

    #[test]
    fn scale_step() {
        let seq = Sequence::new()
            .scale_to(Vec3::ZERO, 0.3)
            .ease(Ease::InCubic)
            .build();

        match &seq.steps[0] {
            SequenceStep::ScaleTo {
                target,
                duration,
                ease,
            } => {
                assert_eq!(*target, Vec3::ZERO);
                assert!((duration - 0.3).abs() < 1e-6);
                assert_eq!(*ease, Ease::InCubic);
            }
            _ => panic!("expected ScaleTo"),
        }
    }

    #[test]
    fn spawn_blueprint_step() {
        let seq = Sequence::new().spawn_blueprint("guard").build();
        match &seq.steps[0] {
            SequenceStep::SpawnBlueprint { name } => assert_eq!(name, "guard"),
            _ => panic!("expected SpawnBlueprint"),
        }
    }

    #[test]
    fn serialization_roundtrip() {
        let seq = Sequence::new()
            .wait(1.0)
            .move_by(Vec3::Y, 2.0)
            .ease(Ease::OutBounce)
            .emit("done")
            .set_state("key", 42_i32)
            .build();

        let ron = ron::to_string(&seq).unwrap();
        let back: Sequence = ron::from_str(&ron).unwrap();
        assert_eq!(back.steps.len(), seq.steps.len());
        assert_eq!(back.current, 0);
        assert_eq!(back.timer, 0.0);
    }
}
