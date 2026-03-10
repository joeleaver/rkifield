//! Camera properties panel — FOV, fly speed, near/far, position readout.

use rinch::prelude::*;

use crate::editor_state::{SliderSignals, UiSignals};
use crate::CommandSender;

use super::slider_helpers::build_slider_row;
use super::{DIVIDER_STYLE, SECTION_STYLE, VALUE_STYLE};

/// Camera properties section: FOV, fly speed, near/far sliders + position readout.
#[component]
pub fn CameraProperties() -> NodeHandle {
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let ui = use_context::<UiSignals>();

    let pos = ui.camera_display_pos.get();

    // Build slider rows imperatively (build_slider_row appends to a container).
    let slider_section = __scope.create_element("div");
    build_slider_row(
        __scope, &slider_section, "FOV", "\u{00b0}",
        sliders.fov, 30.0, 120.0, 1.0, 0,
        { let cmd = cmd.clone(); move |_v| { sliders.send_camera_commands(&cmd); } },
    );
    build_slider_row(
        __scope, &slider_section, "Fly Speed", "",
        sliders.fly_speed, 0.5, 500.0, 0.5, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_camera_commands(&cmd); } },
    );
    build_slider_row(
        __scope, &slider_section, "Near Plane", "",
        sliders.near, 0.01, 10.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_camera_commands(&cmd); } },
    );
    build_slider_row(
        __scope, &slider_section, "Far Plane", "",
        sliders.far, 100.0, 10000.0, 100.0, 0,
        { let cmd = cmd.clone(); move |_v| { sliders.send_camera_commands(&cmd); } },
    );

    rsx! {
        div { style: "display:flex;flex-direction:column;",
            div { style: {SECTION_STYLE}, "Camera" }
            {slider_section}
            div { style: {DIVIDER_STYLE} }
            div { style: {VALUE_STYLE},
                {format!("Pos: ({:.1}, {:.1}, {:.1})", pos.x, pos.y, pos.z)}
            }
        }
    }
}
