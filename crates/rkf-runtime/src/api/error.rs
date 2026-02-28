//! Error types for the World API.

use super::entity::Entity;

/// Errors returned by [`super::World`] operations.
#[derive(Debug)]
pub enum WorldError {
    /// The entity does not exist (was despawned or never created).
    NoSuchEntity(Entity),
    /// The entity is missing the requested component.
    MissingComponent(Entity, &'static str),
    /// Reparenting would create a cycle in the hierarchy.
    CycleDetected,
    /// I/O error (file read/write).
    Io(std::io::Error),
    /// Scene parse error.
    Parse(String),
    /// Voxelization error.
    Voxelize(String),
}

impl std::fmt::Display for WorldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchEntity(e) => write!(f, "entity {} does not exist", e),
            Self::MissingComponent(e, name) => {
                write!(f, "missing component {} on {}", name, e)
            }
            Self::CycleDetected => write!(f, "reparent would create cycle"),
            Self::Io(err) => write!(f, "I/O: {}", err),
            Self::Parse(msg) => write!(f, "parse: {}", msg),
            Self::Voxelize(msg) => write!(f, "voxelization: {}", msg),
        }
    }
}

impl std::error::Error for WorldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for WorldError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::entity::Entity;

    #[test]
    fn no_such_entity_display() {
        let e = Entity::sdf(1, 0);
        let err = WorldError::NoSuchEntity(e);
        let s = format!("{}", err);
        assert!(s.contains("does not exist"));
    }

    #[test]
    fn missing_component_display() {
        let e = Entity::sdf(1, 0);
        let err = WorldError::MissingComponent(e, "Velocity");
        let s = format!("{}", err);
        assert!(s.contains("Velocity"));
    }

    #[test]
    fn cycle_detected_display() {
        let err = WorldError::CycleDetected;
        assert!(format!("{}", err).contains("cycle"));
    }

    #[test]
    fn io_error_from() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let err = WorldError::from(io_err);
        assert!(matches!(err, WorldError::Io(_)));
    }

    #[test]
    fn parse_error_display() {
        let err = WorldError::Parse("bad RON".to_string());
        assert!(format!("{}", err).contains("bad RON"));
    }
}
