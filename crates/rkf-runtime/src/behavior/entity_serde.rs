//! Serde helpers for `hecs::Entity` fields in `#[component]` structs.
//!
//! `hecs::Entity` does not implement `Serialize`/`Deserialize`. The
//! `#[component]` proc macro injects `#[serde(serialize_with, deserialize_with)]`
//! attributes on `Entity` and `Option<Entity>` fields, pointing to the functions
//! in this module. They serialize the entity as raw `u64` bits (via
//! `Entity::to_bits` / `Entity::from_bits`), which is sufficient for
//! round-tripping through RON. The higher-level StableId UUID remapping happens
//! separately in `deserialize_insert`.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Serialize an `Entity` as its raw `u64` bits.
pub fn ser_entity<S: Serializer>(entity: &hecs::Entity, s: S) -> Result<S::Ok, S::Error> {
    entity.to_bits().serialize(s)
}

/// Deserialize an `Entity` from raw `u64` bits.
pub fn de_entity<'de, D: Deserializer<'de>>(d: D) -> Result<hecs::Entity, D::Error> {
    let bits = u64::deserialize(d)?;
    hecs::Entity::from_bits(bits)
        .ok_or_else(|| serde::de::Error::custom("invalid entity bits"))
}

/// Serialize an `Option<Entity>` as `Option<u64>`.
pub fn ser_opt_entity<S: Serializer>(
    entity: &Option<hecs::Entity>,
    s: S,
) -> Result<S::Ok, S::Error> {
    entity.map(|e| e.to_bits()).serialize(s)
}

/// Deserialize an `Option<Entity>` from `Option<u64>`.
pub fn de_opt_entity<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<Option<hecs::Entity>, D::Error> {
    let bits: Option<u64> = Option::deserialize(d)?;
    match bits {
        Some(b) => hecs::Entity::from_bits(b)
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("invalid entity bits")),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_roundtrip_ron() {
        // Create a world and spawn an entity to get a valid Entity handle.
        let mut world = hecs::World::new();
        let entity = world.spawn(());

        // Serialize via RON using our helper.
        let ron_str = ron::to_string(&EntityWrapper(entity)).unwrap();
        let deserialized: EntityWrapper = ron::from_str(&ron_str).unwrap();
        assert_eq!(deserialized.0, entity);
    }

    #[test]
    fn option_entity_some_roundtrip() {
        let mut world = hecs::World::new();
        let entity = world.spawn(());

        let ron_str = ron::to_string(&OptEntityWrapper(Some(entity))).unwrap();
        let deserialized: OptEntityWrapper = ron::from_str(&ron_str).unwrap();
        assert_eq!(deserialized.0, Some(entity));
    }

    #[test]
    fn option_entity_none_roundtrip() {
        let ron_str = ron::to_string(&OptEntityWrapper(None)).unwrap();
        let deserialized: OptEntityWrapper = ron::from_str(&ron_str).unwrap();
        assert_eq!(deserialized.0, None);
    }

    // Helper wrappers to test serde attributes
    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    struct EntityWrapper(
        #[serde(
            serialize_with = "ser_entity",
            deserialize_with = "de_entity"
        )]
        hecs::Entity,
    );

    #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
    struct OptEntityWrapper(
        #[serde(
            serialize_with = "ser_opt_entity",
            deserialize_with = "de_opt_entity"
        )]
        Option<hecs::Entity>,
    );
}
