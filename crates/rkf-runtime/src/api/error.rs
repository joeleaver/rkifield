//! Error types for the World API.

use uuid::Uuid;

/// Errors returned by [`super::World`] operations.
#[derive(Debug)]
pub enum WorldError {
    /// The entity does not exist (was despawned or never created).
    NoSuchEntity(Uuid),
    /// The entity is missing the requested component.
    MissingComponent(Uuid, &'static str),
    /// Reparenting would create a cycle in the hierarchy.
    CycleDetected,
    /// I/O error (file read/write).
    Io(std::io::Error),
    /// Scene parse error.
    Parse(String),
    /// Voxelization error.
    Voxelize(String),
    /// A scene node with the given name was not found.
    NodeNotFound(String),
    /// Scene index out of range.
    SceneOutOfRange(usize),
    /// Cannot remove the last scene.
    CannotRemoveLastScene,
}

impl std::fmt::Display for WorldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchEntity(id) => write!(f, "entity {} does not exist", id),
            Self::MissingComponent(id, name) => {
                write!(f, "missing component {} on {}", name, id)
            }
            Self::CycleDetected => write!(f, "reparent would create cycle"),
            Self::Io(err) => write!(f, "I/O: {}", err),
            Self::Parse(msg) => write!(f, "parse: {}", msg),
            Self::Voxelize(msg) => write!(f, "voxelization: {}", msg),
            Self::NodeNotFound(name) => write!(f, "node not found: {}", name),
            Self::SceneOutOfRange(idx) => write!(f, "scene index {} out of range", idx),
            Self::CannotRemoveLastScene => write!(f, "cannot remove the last scene"),
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

    #[test]
    fn no_such_entity_display() {
        let id = Uuid::from_u128(1);
        let err = WorldError::NoSuchEntity(id);
        let s = format!("{}", err);
        assert!(s.contains("does not exist"));
    }

    #[test]
    fn missing_component_display() {
        let id = Uuid::from_u128(1);
        let err = WorldError::MissingComponent(id, "Velocity");
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
