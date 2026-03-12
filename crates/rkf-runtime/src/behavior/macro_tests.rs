//! Tests for the `#[component]` proc macro.

#[cfg(test)]
mod tests {
    use crate::behavior::{ComponentEntry, ComponentMeta, FieldType, GameValue};
    use glam::Vec3;
    use rkf_macros::component;

    // ─── Test components ────────────────────────────────────────────────

    #[component]
    pub struct Health {
        pub current: f32,
        pub max: f32,
    }

    #[component]
    pub struct Named {
        pub label: String,
        pub priority: i32,
    }

    #[component]
    pub struct Spatial {
        pub velocity: Vec3,
        pub speed: f64,
    }

    #[component]
    pub struct WithTransient {
        pub visible: bool,
        #[serde(skip)]
        pub _cache: u32,
    }

    #[component]
    pub struct Counter {
        pub count: u64,
        pub active: bool,
    }

    // ─── ComponentMeta tests ────────────────────────────────────────────

    #[test]
    fn type_name_matches_struct() {
        assert_eq!(Health::type_name(), "Health");
        assert_eq!(Named::type_name(), "Named");
        assert_eq!(Spatial::type_name(), "Spatial");
        assert_eq!(WithTransient::type_name(), "WithTransient");
    }

    #[test]
    fn fields_count_and_names() {
        let fields = Health::fields();
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].name, "current");
        assert_eq!(fields[1].name, "max");
    }

    #[test]
    fn field_types_f32() {
        let fields = Health::fields();
        assert_eq!(fields[0].field_type, FieldType::Float);
        assert_eq!(fields[1].field_type, FieldType::Float);
    }

    #[test]
    fn field_types_string() {
        let fields = Named::fields();
        assert_eq!(fields[0].field_type, FieldType::String);
        assert_eq!(fields[0].name, "label");
    }

    #[test]
    fn field_types_int() {
        let fields = Named::fields();
        assert_eq!(fields[1].field_type, FieldType::Int);
        assert_eq!(fields[1].name, "priority");
    }

    #[test]
    fn field_types_vec3() {
        let fields = Spatial::fields();
        assert_eq!(fields[0].field_type, FieldType::Vec3);
        assert_eq!(fields[0].name, "velocity");
    }

    #[test]
    fn field_types_bool() {
        let fields = Counter::fields();
        assert_eq!(fields[1].field_type, FieldType::Bool);
        assert_eq!(fields[1].name, "active");
    }

    #[test]
    fn transient_field_detection() {
        let fields = WithTransient::fields();
        assert_eq!(fields.len(), 2);
        // visible is not transient
        assert!(!fields[0].transient);
        assert_eq!(fields[0].name, "visible");
        // _cache is transient (serde(skip))
        assert!(fields[1].transient);
        assert_eq!(fields[1].name, "_cache");
    }

    #[test]
    fn default_derived() {
        let h = Health::default();
        assert_eq!(h.current, 0.0);
        assert_eq!(h.max, 0.0);
    }

    // ─── Serialize / Deserialize roundtrip ──────────────────────────────

    #[test]
    fn serialize_deserialize_roundtrip() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Health {
            current: 75.0,
            max: 100.0,
        },));

        // Find the Health entry from inventory.
        let entry = find_entry("Health").expect("Health should be registered via inventory");

        // Serialize
        let ron_str = (entry.serialize)(&world, entity).expect("entity has Health");
        assert!(ron_str.contains("75"));
        assert!(ron_str.contains("100"));

        // Deserialize into a new entity.
        let entity2 = world.spawn(());
        (entry.deserialize_insert)(&mut world, entity2, &ron_str).unwrap();

        let h = world.get::<&Health>(entity2).unwrap();
        assert_eq!(h.current, 75.0);
        assert_eq!(h.max, 100.0);
    }

    #[test]
    fn serialize_returns_none_when_missing() {
        let mut world = hecs::World::new();
        let entity = world.spawn(());
        let entry = find_entry("Health").unwrap();
        assert!((entry.serialize)(&world, entity).is_none());
    }

    // ─── has / remove ───────────────────────────────────────────────────

    #[test]
    fn has_and_remove() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Counter {
            count: 42,
            active: true,
        },));

        let entry = find_entry("Counter").unwrap();

        assert!((entry.has)(&world, entity));
        (entry.remove)(&mut world, entity);
        assert!(!(entry.has)(&world, entity));
    }

    // ─── get_field / set_field ──────────────────────────────────────────

    #[test]
    fn get_field_f32() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Health {
            current: 50.0,
            max: 100.0,
        },));

        let entry = find_entry("Health").unwrap();
        let val = (entry.get_field)(&world, entity, "current").unwrap();
        assert_eq!(val, GameValue::Float(50.0));

        let val = (entry.get_field)(&world, entity, "max").unwrap();
        assert_eq!(val, GameValue::Float(100.0));
    }

    #[test]
    fn set_field_f32() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Health {
            current: 50.0,
            max: 100.0,
        },));

        let entry = find_entry("Health").unwrap();
        (entry.set_field)(&mut world, entity, "current", GameValue::Float(25.0)).unwrap();

        let h = world.get::<&Health>(entity).unwrap();
        assert_eq!(h.current, 25.0);
    }

    #[test]
    fn get_field_string() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Named {
            label: "test".into(),
            priority: 5,
        },));

        let entry = find_entry("Named").unwrap();
        let val = (entry.get_field)(&world, entity, "label").unwrap();
        assert_eq!(val, GameValue::String("test".into()));
    }

    #[test]
    fn set_field_string() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Named {
            label: "old".into(),
            priority: 1,
        },));

        let entry = find_entry("Named").unwrap();
        (entry.set_field)(
            &mut world,
            entity,
            "label",
            GameValue::String("new".into()),
        )
        .unwrap();

        let n = world.get::<&Named>(entity).unwrap();
        assert_eq!(n.label, "new");
    }

    #[test]
    fn get_set_field_i32() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Named {
            label: "x".into(),
            priority: 10,
        },));

        let entry = find_entry("Named").unwrap();
        let val = (entry.get_field)(&world, entity, "priority").unwrap();
        assert_eq!(val, GameValue::Int(10));

        (entry.set_field)(&mut world, entity, "priority", GameValue::Int(99)).unwrap();
        let n = world.get::<&Named>(entity).unwrap();
        assert_eq!(n.priority, 99);
    }

    #[test]
    fn get_set_field_vec3() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Spatial {
            velocity: Vec3::new(1.0, 2.0, 3.0),
            speed: 5.0,
        },));

        let entry = find_entry("Spatial").unwrap();
        let val = (entry.get_field)(&world, entity, "velocity").unwrap();
        assert_eq!(val, GameValue::Vec3(Vec3::new(1.0, 2.0, 3.0)));

        (entry.set_field)(
            &mut world,
            entity,
            "velocity",
            GameValue::Vec3(Vec3::ZERO),
        )
        .unwrap();

        let s = world.get::<&Spatial>(entity).unwrap();
        assert_eq!(s.velocity, Vec3::ZERO);
    }

    #[test]
    fn get_set_field_bool() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Counter {
            count: 1,
            active: true,
        },));

        let entry = find_entry("Counter").unwrap();
        let val = (entry.get_field)(&world, entity, "active").unwrap();
        assert_eq!(val, GameValue::Bool(true));

        (entry.set_field)(&mut world, entity, "active", GameValue::Bool(false)).unwrap();
        let c = world.get::<&Counter>(entity).unwrap();
        assert!(!c.active);
    }

    #[test]
    fn get_field_unknown_returns_error() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Health {
            current: 1.0,
            max: 2.0,
        },));

        let entry = find_entry("Health").unwrap();
        let err = (entry.get_field)(&world, entity, "nonexistent").unwrap_err();
        assert!(err.contains("unknown field"));
    }

    #[test]
    fn set_field_type_mismatch() {
        let mut world = hecs::World::new();
        let entity = world.spawn((Health {
            current: 1.0,
            max: 2.0,
        },));

        let entry = find_entry("Health").unwrap();
        let err =
            (entry.set_field)(&mut world, entity, "current", GameValue::Bool(true)).unwrap_err();
        assert!(err.contains("type mismatch"));
    }

    #[test]
    fn transient_field_not_accessible() {
        let mut world = hecs::World::new();
        let entity = world.spawn((WithTransient {
            visible: true,
            _cache: 99,
        },));

        let entry = find_entry("WithTransient").unwrap();
        // get_field on transient field should fail (not generated)
        // visible (non-transient) should work
        let val = (entry.get_field)(&world, entity, "visible").unwrap();
        assert_eq!(val, GameValue::Bool(true));

        // _cache (transient) should return unknown field error
        let err = (entry.get_field)(&world, entity, "_cache").unwrap_err();
        assert!(err.contains("unknown field"));
    }

    // ─── Inventory collection ───────────────────────────────────────────

    #[test]
    fn inventory_collects_all_test_components() {
        // All #[component] structs defined in this module should be in inventory.
        let names: Vec<&str> = inventory::iter::<ComponentEntry>()
            .map(|e| e.name)
            .collect();
        assert!(names.contains(&"Health"), "missing Health, got: {:?}", names);
        assert!(names.contains(&"Named"), "missing Named, got: {:?}", names);
        assert!(
            names.contains(&"Spatial"),
            "missing Spatial, got: {:?}",
            names
        );
        assert!(
            names.contains(&"WithTransient"),
            "missing WithTransient, got: {:?}",
            names
        );
        assert!(
            names.contains(&"Counter"),
            "missing Counter, got: {:?}",
            names
        );
    }

    // ─── Enum test components ──────────────────────────────────────────

    /// Unit-variant enum — default is first variant (Idle).
    #[component]
    #[derive(Debug, PartialEq)]
    pub enum GuardState {
        Idle,
        Patrolling,
        Alerted,
        Chasing,
    }

    /// Enum with tuple and struct variants — first variant has data, so no Default.
    #[component(no_default)]
    #[derive(Debug, PartialEq)]
    pub enum Action {
        MoveTo(f32, f32, f32),
        Attack { target_id: u32, damage: f32 },
        Wait,
    }

    // ─── Enum ComponentMeta tests ────────────────────────────────────────

    #[test]
    fn enum_type_name() {
        assert_eq!(GuardState::type_name(), "GuardState");
        assert_eq!(Action::type_name(), "Action");
    }

    #[test]
    fn enum_fields_empty() {
        assert!(GuardState::fields().is_empty());
        assert!(Action::fields().is_empty());
    }

    #[test]
    fn enum_default_first_unit_variant() {
        let gs = GuardState::default();
        assert_eq!(gs, GuardState::Idle);
    }

    // ─── Enum serialization round-trip ──────────────────────────────────

    #[test]
    fn enum_unit_variant_serialize_roundtrip() {
        let mut world = hecs::World::new();
        let entity = world.spawn((GuardState::Alerted,));

        let entry = find_entry("GuardState").expect("GuardState registered");

        let ron_str = (entry.serialize)(&world, entity).expect("entity has GuardState");
        assert!(ron_str.contains("Alerted"));

        let entity2 = world.spawn(());
        (entry.deserialize_insert)(&mut world, entity2, &ron_str).unwrap();

        let gs = world.get::<&GuardState>(entity2).unwrap();
        assert_eq!(*gs, GuardState::Alerted);
    }

    #[test]
    fn enum_data_variant_serialize_roundtrip() {
        let mut world = hecs::World::new();
        let entry = find_entry("Action").expect("Action registered");

        // Tuple variant
        let e1 = world.spawn((Action::MoveTo(1.0, 2.0, 3.0),));
        let ron_str = (entry.serialize)(&world, e1).unwrap();
        let e1b = world.spawn(());
        (entry.deserialize_insert)(&mut world, e1b, &ron_str).unwrap();
        {
            let a = world.get::<&Action>(e1b).unwrap();
            assert_eq!(*a, Action::MoveTo(1.0, 2.0, 3.0));
        }

        // Struct variant
        let e2 = world.spawn((Action::Attack {
            target_id: 42,
            damage: 10.5,
        },));
        let ron_str2 = (entry.serialize)(&world, e2).unwrap();
        let e2b = world.spawn(());
        (entry.deserialize_insert)(&mut world, e2b, &ron_str2).unwrap();
        {
            let a2 = world.get::<&Action>(e2b).unwrap();
            assert_eq!(
                *a2,
                Action::Attack {
                    target_id: 42,
                    damage: 10.5
                }
            );
        }

        // Unit variant within mixed enum
        let e3 = world.spawn((Action::Wait,));
        let ron_str3 = (entry.serialize)(&world, e3).unwrap();
        let e3b = world.spawn(());
        (entry.deserialize_insert)(&mut world, e3b, &ron_str3).unwrap();
        let a3 = world.get::<&Action>(e3b).unwrap();
        assert_eq!(*a3, Action::Wait);
    }

    // ─── Enum get_field / set_field (not applicable) ────────────────────

    #[test]
    fn enum_get_field_returns_error() {
        let mut world = hecs::World::new();
        let entity = world.spawn((GuardState::Idle,));

        let entry = find_entry("GuardState").unwrap();
        let err = (entry.get_field)(&world, entity, "anything").unwrap_err();
        assert!(err.contains("not supported"));
    }

    #[test]
    fn enum_set_field_returns_error() {
        let mut world = hecs::World::new();
        let entity = world.spawn((GuardState::Idle,));

        let entry = find_entry("GuardState").unwrap();
        let err =
            (entry.set_field)(&mut world, entity, "anything", GameValue::Int(1)).unwrap_err();
        assert!(err.contains("not supported"));
    }

    // ─── Enum has / remove ──────────────────────────────────────────────

    #[test]
    fn enum_has_and_remove() {
        let mut world = hecs::World::new();
        let entity = world.spawn((GuardState::Patrolling,));

        let entry = find_entry("GuardState").unwrap();
        assert!((entry.has)(&world, entity));
        (entry.remove)(&mut world, entity);
        assert!(!(entry.has)(&world, entity));
    }

    // ─── Enum inventory collection ──────────────────────────────────────

    #[test]
    fn inventory_collects_enum_components() {
        let names: Vec<&str> = inventory::iter::<ComponentEntry>()
            .map(|e| e.name)
            .collect();
        assert!(
            names.contains(&"GuardState"),
            "missing GuardState, got: {:?}",
            names
        );
        assert!(
            names.contains(&"Action"),
            "missing Action, got: {:?}",
            names
        );
    }

    // ─── Helper ─────────────────────────────────────────────────────────

    fn find_entry(name: &str) -> Option<&'static ComponentEntry> {
        inventory::iter::<ComponentEntry>().find(|e| e.name == name)
    }
}
