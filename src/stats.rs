//! Small on-screen stat helpers. Kept tiny and allocation-free per tick
//! so they never show up in profiles, even when updated every frame.

use egui::{Color32, Pos2, Rect, Stroke, Vec2};

/// Fixed-capacity sliding window of f32 samples. Bounded memory.
pub struct RollingStats {
    samples: Vec<f32>,
    head: usize,
    len: usize,
    capacity: usize,
}

impl RollingStats {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: vec![0.0; capacity],
            head: 0,
            len: 0,
            capacity,
        }
    }

    pub fn push(&mut self, v: f32) {
        self.samples[self.head] = v;
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn mean(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        let mut s = 0.0f32;
        for i in 0..self.len {
            s += self.samples[i];
        }
        s / self.len as f32
    }

    pub fn min_max(&self) -> (f32, f32) {
        if self.len == 0 {
            return (0.0, 0.0);
        }
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..self.len {
            let v = self.samples[i];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        (lo, hi)
    }

    /// Iterate the samples in insertion order, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        let start = if self.len < self.capacity {
            0
        } else {
            self.head
        };
        (0..self.len).map(move |i| self.samples[(start + i) % self.capacity])
    }
}

/// Minimalist sparkline for overlay stats. Draws within `rect` using a
/// single stroke so GPU load is independent of sample count.
pub struct SparkLine;

impl SparkLine {
    pub fn draw(painter: &egui::Painter, rect: Rect, stats: &RollingStats, color: Color32) {
        if stats.len() < 2 {
            return;
        }
        let (lo, hi) = stats.min_max();
        let span = (hi - lo).max(1e-6);
        let n = stats.len().max(1) - 1;
        let mut prev: Option<Pos2> = None;
        for (i, v) in stats.iter().enumerate() {
            let t = i as f32 / n as f32;
            let y = 1.0 - (v - lo) / span;
            let p = rect.min + Vec2::new(t * rect.width(), y * rect.height());
            if let Some(a) = prev {
                painter.line_segment([a, p], Stroke::new(1.0, color));
            }
            prev = Some(p);
        }
    }
}
