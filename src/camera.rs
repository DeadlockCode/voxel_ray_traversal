use nalgebra::{Matrix4, Vector3};

pub struct Camera {
    pub position: Vector3<f64>,
    pub rotation: Vector3<f64>,
    pub extent: [f64; 2],
    pub fov: f64,
}

impl Camera {
    pub fn new(position: Vector3<f64>, rotation: Vector3<f64>, extent: [f64; 2], fov: f64) -> Self {
        Camera {
            position,
            rotation,
            extent,
            fov,
        }
    }

    pub fn look_direction(&mut self, direction: Vector3<f64>) {
        let z = (-direction.x).atan2(direction.y);
        let x = direction.z.atan2(direction.xy().magnitude());
        self.rotation = Vector3::new(x, 0.0, z);
    }

    pub fn look_at(&mut self, target: Vector3<f64>) {
        let direction = target - self.position;
        self.look_direction(direction);
    }

    pub fn rotation_matrix(&self) -> Matrix4<f64> {
        let x = Matrix4::from_axis_angle(&Vector3::x_axis(), self.rotation.x);
        let y = Matrix4::from_axis_angle(&Vector3::y_axis(), self.rotation.y);
        let z = Matrix4::from_axis_angle(&Vector3::z_axis(), self.rotation.z);

        z * x * y
    }

    pub fn translation_matrix(&self) -> Matrix4<f64> {
        let mut translation = Matrix4::identity();
        translation.m14 = self.position.x;
        translation.m24 = self.position.y;
        translation.m34 = self.position.z;
        translation
    }

    pub fn pixel_to_ray_matrix(&self) -> Matrix4<f64> {
        let aspect = self.extent[0] / self.extent[1];
        let tan_fov = (self.fov.to_radians() * 0.5).tan();

        let mut center_pixel = Matrix4::identity();
        center_pixel.m13 = 0.5;
        center_pixel.m23 = 0.5;

        let mut pixel_to_uv = Matrix4::identity();
        pixel_to_uv.m11 = 2.0 / self.extent[0];
        pixel_to_uv.m22 = -2.0 / self.extent[1];
        pixel_to_uv.m13 = -1.0;
        pixel_to_uv.m23 = 1.0;

        let mut uv_to_view = Matrix4::identity();
        uv_to_view.m11 = tan_fov * aspect.max(1.0);
        uv_to_view.m22 = tan_fov / aspect.min(1.0);

        let mut swap_yz = Matrix4::identity();
        swap_yz.swap_columns(1, 2);

        let rotation = self.rotation_matrix();
        let translation = self.translation_matrix();

        translation * rotation * swap_yz * uv_to_view * pixel_to_uv * center_pixel
    }
}
