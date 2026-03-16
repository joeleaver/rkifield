//! Vec3Editor — three DragValues in a row with colored X/Y/Z labels.

use rinch::prelude::*;

use super::DragValue;

/// Three DragValues for editing a 3D vector, with colored X (red), Y (green), Z (blue) labels.
pub struct Vec3Editor {
    pub x: Memo<f64>,
    pub y: Memo<f64>,
    pub z: Memo<f64>,
    /// Called on every drag tick with full [x, y, z] (live preview).
    pub on_change: Option<ValueCallback<[f64; 3]>>,
    /// Called on drag end / Enter (final command).
    pub on_commit: Option<ValueCallback<[f64; 3]>>,
    pub step: f64,
    pub min: f64,
    pub max: f64,
    pub decimals: u32,
    pub suffix: String,
}

impl std::fmt::Debug for Vec3Editor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vec3Editor").field("step", &self.step).finish()
    }
}

impl Default for Vec3Editor {
    fn default() -> Self {
        Self {
            x: Memo::new(|| 0.0),
            y: Memo::new(|| 0.0),
            z: Memo::new(|| 0.0),
            on_change: None,
            on_commit: None,
            step: 0.01,
            min: -1e9,
            max: 1e9,
            decimals: 2,
            suffix: String::new(),
        }
    }
}

impl Vec3Editor {
    /// Create from Signals (for environment/camera where UI owns the data).
    pub fn from_signals(
        x: Signal<f64>,
        y: Signal<f64>,
        z: Signal<f64>,
        on_change: Option<ValueCallback<f64>>,
    ) -> Self {
        Self {
            x: Memo::new(move || x.get()),
            y: Memo::new(move || y.get()),
            z: Memo::new(move || z.get()),
            on_change: {
                let on_change = on_change.clone();
                Some(ValueCallback::new(move |v: [f64; 3]| {
                    x.set(v[0]);
                    y.set(v[1]);
                    z.set(v[2]);
                    if let Some(cb) = &on_change { cb.0(v[0]); }
                }))
            },
            on_commit: None,
            ..Default::default()
        }
    }
}

impl Component for Vec3Editor {
    fn render(&self, __scope: &mut RenderScope, _children: &[NodeHandle]) -> NodeHandle {
        let axes: [(Memo<f64>, &str, &str); 3] = [
            (self.x, "X", "#e05050"),
            (self.y, "Y", "#50c050"),
            (self.z, "Z", "#5080e0"),
        ];

        let container = rsx! {
            div { style: "display: flex; gap: 4px; align-items: center;" }
        };

        // All three Memos are Copy — capture freely.
        let mx = self.x;
        let my = self.y;
        let mz = self.z;

        for (axis_idx, (memo, label, color)) in axes.into_iter().enumerate() {
            let on_change_axis = self.on_change.as_ref().map(|cb| {
                let cb = cb.clone();
                ValueCallback::new(move |v: f64| {
                    let mut arr = [mx.get(), my.get(), mz.get()];
                    arr[axis_idx] = v;
                    cb.0(arr);
                })
            });

            let on_commit_axis = self.on_commit.as_ref().map(|cb| {
                let cb = cb.clone();
                ValueCallback::new(move |v: f64| {
                    let mut arr = [mx.get(), my.get(), mz.get()];
                    arr[axis_idx] = v;
                    cb.0(arr);
                })
            });

            let dv = DragValue {
                value: memo,
                on_change: on_change_axis,
                on_commit: on_commit_axis,
                step: self.step,
                min: self.min,
                max: self.max,
                decimals: self.decimals,
                label: label.into(),
                label_color: color.into(),
                suffix: self.suffix.clone(),
            };
            container.append_child(&dv.render(__scope, &[]));
        }

        container
    }
}
