//! An orbit camera: turn about a target, pan it, dolly in and out.

use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self { target: Vec3::ZERO, distance: 4.0, yaw: 0.5, pitch: 0.35, fov_y: 0.7 }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let cp = self.pitch.cos();
        self.target + Vec3::new(self.yaw.sin() * cp, self.pitch.sin(), self.yaw.cos() * cp) * self.distance
    }

    fn right(&self) -> Vec3 {
        (self.target - self.eye()).cross(Vec3::Y).normalize_or(Vec3::X)
    }

    fn up(&self) -> Vec3 {
        self.right().cross((self.target - self.eye()).normalize_or(Vec3::NEG_Z)).normalize_or(Vec3::Y)
    }

    /// Clip planes that follow the distance, so a small model and a large one both fit.
    fn planes(&self) -> (f32, f32) {
        ((self.distance * 0.01).max(0.001), self.distance * 100.0)
    }

    pub fn proj(&self, aspect: f32) -> Mat4 {
        let (near, far) = self.planes();
        Mat4::perspective_rh_gl(self.fov_y, aspect.max(0.05), near, far)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj(aspect) * Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    /// Turns by a drag of `delta` pixels.
    pub fn orbit(&mut self, delta: glam::Vec2) {
        self.yaw -= delta.x * 0.01;
        self.pitch = (self.pitch + delta.y * 0.01).clamp(-1.5, 1.5);
    }

    /// Slides the target across the view by a drag of `delta` pixels, `height` being the
    /// viewport's, so the point under the cursor stays under it.
    pub fn pan(&mut self, delta: glam::Vec2, height: f32) {
        let per_pixel = 2.0 * self.distance * (self.fov_y * 0.5).tan() / height.max(1.0);
        self.target += (self.up() * delta.y - self.right() * delta.x) * per_pixel;
    }

    /// Dollies by scroll or pinch; each unit is a small fraction of the distance.
    pub fn zoom(&mut self, amount: f32) {
        self.distance = (self.distance * (-amount * 0.0015).exp()).clamp(0.01, 10_000.0);
    }

    /// Backs off until the box `min`..`max` fills the view.
    pub fn frame(&mut self, min: Vec3, max: Vec3) {
        self.target = (min + max) * 0.5;
        let radius = ((max - min).length() * 0.5).max(1e-3);
        self.distance = radius / (self.fov_y * 0.5).sin() * 1.15;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_centres_and_fits_the_box() {
        let mut cam = OrbitCamera::default();
        cam.frame(Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 2.0, 1.0));
        assert!((cam.target - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-5);
        // Every corner of the box lands inside the view.
        let vp = cam.view_proj(1.0);
        for x in [-1.0, 1.0] {
            for y in [0.0, 2.0] {
                for z in [-1.0, 1.0] {
                    let p = vp * glam::Vec4::new(x, y, z, 1.0);
                    let (nx, ny) = (p.x / p.w, p.y / p.w);
                    assert!(nx.abs() <= 1.0 && ny.abs() <= 1.0, "corner outside the view: {nx} {ny}");
                }
            }
        }
    }

    #[test]
    fn panning_keeps_the_view_direction() {
        let mut cam = OrbitCamera::default();
        let before = (cam.target - cam.eye()).normalize();
        cam.pan(glam::Vec2::new(40.0, -25.0), 900.0);
        let after = (cam.target - cam.eye()).normalize();
        assert!((before - after).length() < 1e-5);
        assert!(cam.target.length() > 0.0);
    }

    #[test]
    fn dragging_right_moves_the_scene_right() {
        // Grabbing the scene and pulling right has to move the target left.
        let mut cam = OrbitCamera { yaw: 0.0, pitch: 0.0, ..Default::default() };
        cam.pan(glam::Vec2::new(100.0, 0.0), 900.0);
        assert!(cam.target.x < 0.0);
    }

    #[test]
    fn zoom_stays_in_range_and_pitch_is_clamped() {
        let mut cam = OrbitCamera::default();
        for _ in 0..1000 {
            cam.zoom(500.0);
        }
        assert!(cam.distance >= 0.01);
        cam.orbit(glam::Vec2::new(0.0, 1e6));
        assert!(cam.pitch <= 1.5);
    }
}
