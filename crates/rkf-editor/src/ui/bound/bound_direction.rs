//! BoundDirection — Layer 2 widget that binds two angle sliders (azimuth + elevation)
//! to a UiStore path containing a Vec3 direction.

use rinch::prelude::*;

use crate::store::types::UiValue;
use crate::store::UiStore;
use crate::ui::widgets::float_slider::FloatSlider;

/// A direction input bound to a store path containing a Vec3.
///
/// Displays two sliders — Azimuth (0–360) and Elevation (-90–+90) — and
/// converts between spherical angles and the stored unit Vec3.
#[component]
pub fn BoundDirection(path: String, label: String) -> NodeHandle {
    let store = use_context::<UiStore>();
    let signal = store.read(&path);
    let dir = signal.get().as_vec3().unwrap_or([0.0, 1.0, 0.0]);

    // Vec3 → angles (degrees)
    let (dx, dy, dz) = (dir[0], dir[1], dir[2]);
    let az = (dx.atan2(dz)).to_degrees().rem_euclid(360.0);
    let el = dy.clamp(-1.0, 1.0).asin().to_degrees();

    let store_az = store.clone();
    let path_az = path.clone();
    let az_slider = FloatSlider {
        value: az,
        min: 0.0,
        max: 360.0,
        step: 1.0,
        decimals: 0,
        label: "Azimuth".into(),
        suffix: "\u{00b0}".into(),
        on_change: Some(ValueCallback::new(move |new_az: f64| {
            let cur = store_az.read(&path_az).get().as_vec3().unwrap_or([0.0, 1.0, 0.0]);
            let cur_el = cur[1].clamp(-1.0, 1.0).asin();
            let az_rad = new_az.to_radians();
            let x = az_rad.sin() * cur_el.cos();
            let y = cur_el.sin();
            let z = az_rad.cos() * cur_el.cos();
            let len = (x * x + y * y + z * z).sqrt();
            if len > 1e-12 {
                store_az.set(&path_az, UiValue::Vec3([x / len, y / len, z / len]));
            }
        })),
    }
    .render(__scope, &[]);

    let store_el = store.clone();
    let path_el = path.clone();
    let el_slider = FloatSlider {
        value: el,
        min: -90.0,
        max: 90.0,
        step: 1.0,
        decimals: 0,
        label: "Elevation".into(),
        suffix: "\u{00b0}".into(),
        on_change: Some(ValueCallback::new(move |new_el: f64| {
            let cur = store_el.read(&path_el).get().as_vec3().unwrap_or([0.0, 1.0, 0.0]);
            let (cx, _cy, cz) = (cur[0], cur[1], cur[2]);
            let cur_az = cx.atan2(cz);
            let el_rad = new_el.to_radians();
            let x = cur_az.sin() * el_rad.cos();
            let y = el_rad.sin();
            let z = cur_az.cos() * el_rad.cos();
            let len = (x * x + y * y + z * z).sqrt();
            if len > 1e-12 {
                store_el.set(&path_el, UiValue::Vec3([x / len, y / len, z / len]));
            }
        })),
    }
    .render(__scope, &[]);

    rsx! {
        div {
            div {
                style: "padding:3px 12px;font-size:11px;color:var(--rinch-color-dimmed);",
                {label}
            }
            {az_slider}
            {el_slider}
        }
    }
}
