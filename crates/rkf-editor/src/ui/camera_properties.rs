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

    let root = __scope.create_element("div");
    root.set_attribute("style", "display:flex;flex-direction:column;");

    // Section header.
    let name_row = __scope.create_element("div");
    name_row.set_attribute("style", SECTION_STYLE);
    name_row.append_child(&__scope.create_text("Camera"));
    root.append_child(&name_row);

    // Sliders.
    build_slider_row(
        __scope, &root, "FOV", "\u{00b0}",
        sliders.fov, 30.0, 120.0, 1.0, 0,
        { let cmd = cmd.clone(); move |_v| { sliders.send_camera_commands(&cmd); } },
    );
    build_slider_row(
        __scope, &root, "Fly Speed", "",
        sliders.fly_speed, 0.5, 500.0, 0.5, 1,
        { let cmd = cmd.clone(); move |_v| { sliders.send_camera_commands(&cmd); } },
    );
    build_slider_row(
        __scope, &root, "Near Plane", "",
        sliders.near, 0.01, 10.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_camera_commands(&cmd); } },
    );
    build_slider_row(
        __scope, &root, "Far Plane", "",
        sliders.far, 100.0, 10000.0, 100.0, 0,
        { let cmd = cmd.clone(); move |_v| { sliders.send_camera_commands(&cmd); } },
    );

    // Divider.
    let div = __scope.create_element("div");
    div.set_attribute("style", DIVIDER_STYLE);
    root.append_child(&div);

    // Position (read-only).
    let pos_row = __scope.create_element("div");
    pos_row.set_attribute("style", VALUE_STYLE);
    pos_row.append_child(
        &__scope.create_text(&format!("Pos: ({:.1}, {:.1}, {:.1})", pos.x, pos.y, pos.z)),
    );
    root.append_child(&pos_row);

    root.into()
}
