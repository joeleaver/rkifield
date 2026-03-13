//! Observation tool handlers and shared dispatch functions.
//!
//! Each tool calls the corresponding `AutomationApi` method and returns the result
//! as JSON. The shared functions `standard_tool_definitions()` and
//! `dispatch_tool_call()` allow any `AutomationApi` implementor to serve all
//! observation tools without needing a `ToolRegistry`.

mod dispatch;
mod handlers;
mod registration;

pub use dispatch::{dispatch_tool_call, standard_tool_definitions};
pub use registration::register_observation_tools;
