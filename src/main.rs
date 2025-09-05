use egui_winit_vulkano::{
    Gui, GuiConfig,
    egui::{self, Color32},
};
use nalgebra::{Matrix4, Vector3, Vector4};
use std::path::PathBuf;
use std::{
    f64::consts::{FRAC_PI_2, TAU},
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use vulkano::{
    Validated, Version, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, ClearColorImageInfo, CommandBufferUsage,
        CopyBufferToImageInfo, PrimaryCommandBufferAbstract,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    format::Format,
    image::{
        Image, ImageCreateInfo, ImageType, ImageUsage,
        sampler::Filter,
        view::{ImageView, ImageViewCreateInfo},
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    swapchain::{
        PresentMode, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
        acquire_next_image,
    },
    sync::GpuFuture,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::KeyCode,
    window::{CursorGrabMode, Icon, Window, WindowId},
};
use winit_input_helper::WinitInputHelper;

mod camera;
mod hot_reload;
mod voxelize;

use crate::camera::Camera;
use crate::hot_reload::HotReloadComputePipeline;

const INITIAL_VOXEL_RESOLUTION: u32 = 24;
const INITIAL_WINDOW_RESOLUTION: PhysicalSize<u32> = PhysicalSize::new(960, 960);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RenderMode {
    Coord,
    Steps,
    Normal,
    UV,
    Depth,
}

impl RenderMode {
    pub const ALL: &'static [RenderMode] = &[
        RenderMode::Coord,
        RenderMode::Steps,
        RenderMode::Normal,
        RenderMode::UV,
        RenderMode::Depth,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Model {
    Bunny,
    Dragon,
    Armadillo,
}

impl Model {
    pub const ALL: &'static [Model] = &[Model::Bunny, Model::Dragon, Model::Armadillo];

    pub fn path(&self) -> impl AsRef<Path> {
        let buf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
        match self {
            Model::Bunny => buf.join("bunny_remeshed.ply"),
            Model::Dragon => buf.join("dragon.ply"),
            Model::Armadillo => buf.join("armadillo.ply"),
        }
    }
}

fn get_allocators(
    device: &Arc<Device>,
) -> (
    Arc<StandardMemoryAllocator>,
    Arc<StandardDescriptorSetAllocator>,
    Arc<StandardCommandBufferAllocator>,
) {
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));
    (
        memory_allocator,
        descriptor_set_allocator,
        command_buffer_allocator,
    )
}

fn get_swapchain_images(
    device: &Arc<Device>,
    surface: &Arc<Surface>,
    window: &Window,
) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    let caps = device
        .physical_device()
        .surface_capabilities(surface, Default::default())
        .unwrap();

    let image_format = device
        .physical_device()
        .surface_formats(surface, Default::default())
        .unwrap()[0]
        .0;

    let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();

    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count.max(3),
            image_format,
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
            composite_alpha,
            present_mode: PresentMode::Immediate,
            ..Default::default()
        },
    )
    .unwrap()
}

fn get_render_image(
    memory_allocator: Arc<StandardMemoryAllocator>,
    extent: [u32; 2],
) -> (Arc<Image>, Arc<ImageView>) {
    let image = Image::new(
        memory_allocator,
        ImageCreateInfo {
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
            format: Format::R8G8B8A8_UNORM,
            extent: [extent[0], extent[1], 1],
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .unwrap();

    let image_view =
        ImageView::new(image.clone(), ImageViewCreateInfo::from_image(&image)).unwrap();

    (image, image_view)
}

fn get_resample_image(
    memory_allocator: Arc<StandardMemoryAllocator>,
    extent: [u32; 2],
) -> (Arc<Image>, Arc<ImageView>) {
    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
            format: Format::R8G8B8A8_UNORM,
            extent: [extent[0], extent[1], 1],
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .unwrap();

    let image_view =
        ImageView::new(image.clone(), ImageViewCreateInfo::from_image(&image)).unwrap();

    (image, image_view)
}

fn get_images_and_sets(
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    render_pipeline: &ComputePipeline,
    resample_pipeline: &ComputePipeline,
    render_extent: [u32; 2],
    window_extent: [u32; 2],
) -> (
    Arc<Image>,
    Arc<DescriptorSet>,
    Arc<Image>,
    Arc<DescriptorSet>,
) {
    let (render_image, render_image_view) =
        get_render_image(memory_allocator.clone(), render_extent);

    let layout = render_pipeline.layout().set_layouts()[0].clone();
    let render_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout,
        [WriteDescriptorSet::image_view(0, render_image_view.clone())],
        [],
    )
    .unwrap();

    let (resample_image, resample_image_view) = get_resample_image(memory_allocator, window_extent);

    let layout = resample_pipeline.layout().set_layouts()[0].clone();
    let resample_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout,
        [
            WriteDescriptorSet::image_view(0, render_image_view.clone()),
            WriteDescriptorSet::image_view(1, resample_image_view.clone()),
        ],
        [],
    )
    .unwrap();

    (render_image, render_set, resample_image, resample_set)
}

fn get_voxel_set(
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    render_pipeline: &ComputePipeline,
    queue: &Arc<Queue>,
    voxels: Vec<u128>,
    resolution: u32,
) -> Arc<DescriptorSet> {
    // Each texel is a 4x4x8 block of
    // voxels where each bit is one voxel.
    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim3d,
            format: Format::R32G32B32A32_UINT,
            extent: [resolution / 4, resolution / 4, resolution / 8],
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .unwrap();

    let src_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        voxels,
    )
    .unwrap();

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    command_buffer_builder
        .clear_color_image(ClearColorImageInfo::image(image.clone()))
        .unwrap()
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            src_buffer,
            image.clone(),
        ))
        .unwrap();

    let _ = command_buffer_builder
        .build()
        .unwrap()
        .execute(queue.clone())
        .unwrap();

    let image_view =
        ImageView::new(image.clone(), ImageViewCreateInfo::from_image(&image)).unwrap();

    let layout = render_pipeline
        .layout()
        .set_layouts()
        .get(1)
        .unwrap()
        .clone();
    DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout.clone(),
        [WriteDescriptorSet::image_view(0, image_view)],
        [],
    )
    .unwrap()
}

fn load_icon(icon: &[u8]) -> Icon {
    let (icon_rgba, icon_width, icon_height) = {
        let image = image::load_from_memory(icon).unwrap().to_rgba8();
        let (width, height) = image.dimensions();
        let rgba = image.into_raw();
        (rgba, width, height)
    };
    Icon::from_rgba(icon_rgba, icon_width, icon_height).unwrap()
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    render_pipeline: HotReloadComputePipeline,
    resample_pipeline: HotReloadComputePipeline,

    voxel_set: Arc<DescriptorSet>,
    voxel_resolution: u32,
    future_voxel_resolution: u32,
    model: Model,

    camera: Camera,
    render_mode: RenderMode,
    render_scale: f32,

    input: WinitInputHelper,
    focused: bool,
    last_second: Instant,
    frames_since_last_second: u32,
    fps: u32,

    rcx: Option<RenderContext>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    image_views: Vec<Arc<ImageView>>,

    render_image: Arc<Image>,
    render_set: Arc<DescriptorSet>,
    resample_image: Arc<Image>,
    resample_set: Arc<DescriptorSet>,

    gui: Gui,

    recreate_swapchain: bool,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        let library = VulkanLibrary::new().unwrap();

        let mut required_extensions = Surface::required_extensions(event_loop).unwrap();

        required_extensions.ext_debug_utils = true;

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
            })
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop).unwrap()
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
        }

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    dynamic_rendering: true,
                    ..DeviceFeatures::empty()
                },
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let (memory_allocator, descriptor_set_allocator, command_buffer_allocator) =
            get_allocators(&device);

        let shaders_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
        let render_pipeline =
            HotReloadComputePipeline::new(device.clone(), &shaders_dir.join("traverse.comp"));
        let resample_pipeline =
            HotReloadComputePipeline::new(device.clone(), &shaders_dir.join("resample.comp"));

        let voxel_resolution = INITIAL_VOXEL_RESOLUTION;
        let model = Model::Bunny;
        let voxel_set = {
            let voxels = voxelize::ply_to_voxels(model.path(), voxel_resolution);

            get_voxel_set(
                memory_allocator.clone(),
                command_buffer_allocator.clone(),
                descriptor_set_allocator.clone(),
                &render_pipeline,
                &queue,
                voxels,
                voxel_resolution,
            )
        };

        let input = WinitInputHelper::new();

        let mut camera = Camera::new(
            Vector3::new(2.0, 3.0, 1.0),
            Vector3::zeros(),
            INITIAL_WINDOW_RESOLUTION.into(),
            20.0,
        );
        camera.look_at(Vector3::zeros());

        App {
            instance,
            device,
            queue,

            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,

            render_pipeline,
            resample_pipeline,

            voxel_set,
            voxel_resolution,
            future_voxel_resolution: voxel_resolution,
            model,

            camera,
            render_mode: RenderMode::Coord,
            render_scale: 1.0,

            input,
            focused: false,
            last_second: Instant::now(),
            frames_since_last_second: 0,
            fps: 0,

            rcx: None,
        }
    }

    fn update(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        if now.duration_since(self.last_second) > Duration::from_secs(1) {
            self.fps = self.frames_since_last_second;
            self.frames_since_last_second = 0;
            self.last_second = now;
        }
        self.frames_since_last_second += 1;

        let Some(delta_time) = self.input.delta_time().as_ref().map(Duration::as_secs_f64) else {
            return;
        };

        if self.input.close_requested() {
            event_loop.exit();
            return;
        }

        if self.focused {
            let t = |k: KeyCode| self.input.key_held(k) as u8 as f64;

            let v = Vector3::new(KeyCode::KeyD, KeyCode::KeyW, KeyCode::KeyQ).map(t)
                - Vector3::new(KeyCode::KeyA, KeyCode::KeyS, KeyCode::KeyE).map(t);

            self.camera.position +=
                (self.camera.rotation_matrix() * v.push(0.0) * delta_time).xyz();

            let sens = 0.001 * (self.camera.fov.to_radians() * 0.5).tan();

            let (dx, dy) = self.input.mouse_diff();
            self.camera.rotation.z -= dx as f64 * sens;
            self.camera.rotation.x -= dy as f64 * sens;
            self.camera.rotation.x = self.camera.rotation.x.clamp(-FRAC_PI_2, FRAC_PI_2);
            self.camera.rotation.y = self.camera.rotation.y.rem_euclid(TAU);

            let ds = self.input.scroll_diff();
            let tanfov = (self.camera.fov.to_radians() * 0.5).tan();
            self.camera.fov = ((tanfov * (ds.1 as f64 * -0.1).exp()).atan() * 2.0).to_degrees();
        }

        let rcx = self.rcx.as_mut().unwrap();

        if self.input.mouse_pressed(MouseButton::Left) {
            self.focused = true;
            rcx.window
                .set_cursor_grab(CursorGrabMode::Confined)
                .unwrap();
            rcx.window.set_cursor_visible(false);
        }
        if self.input.key_pressed(KeyCode::Escape) {
            self.focused = false;
            rcx.window.set_cursor_grab(CursorGrabMode::None).unwrap();
            rcx.window.set_cursor_visible(true);
        }
    }

    fn render(&mut self, _event_loop: &ActiveEventLoop) {
        self.render_pipeline.maybe_reload();
        self.resample_pipeline.maybe_reload();

        let rcx = self.rcx.as_mut().unwrap();

        if self.input.window_resized().is_some() {
            rcx.recreate_swapchain = true;
        }

        let window_size = rcx.window.inner_size();

        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        if rcx.recreate_swapchain {
            let images;
            (rcx.swapchain, images) = rcx
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: window_size.into(),
                    ..rcx.swapchain.create_info()
                })
                .unwrap();

            rcx.image_views = images
                .iter()
                .map(|i| ImageView::new(i.clone(), ImageViewCreateInfo::from_image(i)).unwrap())
                .collect();

            let window_extent: [u32; 2] = window_size.into();
            let render_extent = [
                (window_extent[0] as f32 * self.render_scale) as u32,
                (window_extent[1] as f32 * self.render_scale) as u32,
            ];
            (
                rcx.render_image,
                rcx.render_set,
                rcx.resample_image,
                rcx.resample_set,
            ) = get_images_and_sets(
                self.memory_allocator.clone(),
                self.descriptor_set_allocator.clone(),
                &self.render_pipeline,
                &self.resample_pipeline,
                render_extent,
                window_extent,
            );

            rcx.recreate_swapchain = false;
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    rcx.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            rcx.recreate_swapchain = true;
        }

        rcx.gui.immediate_ui(|gui| {
            let ctx = gui.context();

            egui::Window::new("Settings").show(&ctx, |ui| {
                ui.style_mut().spacing.slider_width = 250.0;
                ui.horizontal(|ui| {
                    for &mode in RenderMode::ALL {
                        ui.selectable_value(&mut self.render_mode, mode, format!("{:?}", mode));
                    }
                });
                ui.separator();
                ui.add(egui::Slider::new(&mut self.camera.fov, 0.0..=180.0).text("FOV"));
                if ui
                    .add(
                        egui::Slider::new(&mut self.render_scale, 0.125..=8.0).text("Render Scale"),
                    )
                    .changed()
                {
                    let window_extent: [u32; 2] = rcx.window.inner_size().into();
                    let render_extent = [
                        (window_extent[0] as f32 * self.render_scale) as u32,
                        (window_extent[1] as f32 * self.render_scale) as u32,
                    ];
                    (
                        rcx.render_image,
                        rcx.render_set,
                        rcx.resample_image,
                        rcx.resample_set,
                    ) = get_images_and_sets(
                        self.memory_allocator.clone(),
                        self.descriptor_set_allocator.clone(),
                        &self.render_pipeline,
                        &self.resample_pipeline,
                        render_extent,
                        window_extent,
                    );
                }

                ui.add(
                    egui::Slider::new(&mut self.future_voxel_resolution, 8..=3200)
                    .custom_formatter(|n, _| format!("{}", (n as u32).div_ceil(8) * 8))
                    .custom_parser(|s| s.parse::<u32>().ok().map(|n| n.div_ceil(8) * 8).map(|n| n as f64))
                    .text("Voxel Resolution")
                );
                self.future_voxel_resolution = self.future_voxel_resolution.div_ceil(8) * 8;
                ui.colored_label(Color32::LIGHT_RED, "Warning: Setting voxel resolution too high might crash the program.");
                ui.label("The maximum possible resolution will depend on your GPU's Vulkan limits. This could be mitigated by splitting the world into multiple textures but that's not included in this simple example.");
                ui.horizontal(|ui| {
                    for &model in Model::ALL {
                        ui.selectable_value(&mut self.model, model, format!("{:?}", model));
                    }
                });
                if ui.button("Generate Voxel Grid").clicked() {
                    self.voxel_resolution = self.future_voxel_resolution;
                    let voxels = voxelize::ply_to_voxels(self.model.path(), self.voxel_resolution);
                    self.voxel_set = get_voxel_set(
                        self.memory_allocator.clone(),
                        self.command_buffer_allocator.clone(),
                        self.descriptor_set_allocator.clone(),
                        &self.render_pipeline,
                        &self.queue,
                        voxels,
                        self.voxel_resolution,
                    );
                }
            });
            egui::Window::new("Stats").show(&ctx, |ui| {
                fn format_with_commas(n: u64) -> String {
                    let mut s = n.to_string();

                    let mut i = 3;
                    while i < s.len() {
                        s.insert(s.len() - i, ',');
                        i += 4;
                    }

                    s
                }

                ui.label(format!("FPS: {}", self.fps));

                let voxel_resolution = self.voxel_resolution as u64;
                ui.label(format!(
                    "Voxels: {}³ = {}",
                    format_with_commas(voxel_resolution),
                    format_with_commas(voxel_resolution.pow(3))
                ));
                let window_extent: [u32; 2] = rcx.window.inner_size().into();
                let render_extent = [
                    (window_extent[0] as f32 * self.render_scale) as u32,
                    (window_extent[1] as f32 * self.render_scale) as u32,
                ];
                ui.label(format!(
                    "Pixels: {}×{} = {}",
                    format_with_commas(window_extent[0] as u64),
                    format_with_commas(window_extent[1] as u64),
                    format_with_commas((window_extent[0] * window_extent[1]) as u64)
                ));
                ui.label(format!(
                    "Rays: {}×{} = {}",
                    format_with_commas(render_extent[0] as u64),
                    format_with_commas(render_extent[1] as u64),
                    format_with_commas((render_extent[0] * render_extent[1]) as u64)
                ));
            });
        });

        let render_extent = rcx.render_image.extent();
        let resample_extent = rcx.resample_image.extent();
        self.camera.extent = [render_extent[0] as f64, render_extent[1] as f64];

        let pixel_to_ray = self.camera.pixel_to_ray_matrix();

        let size = self.voxel_resolution as f64;
        let mut scale_and_center = Matrix4::from_diagonal(&Vector4::from_element(size));
        scale_and_center.set_column(3, &Vector3::from_element(0.5 * size).push(1.0));
        let pixel_to_ray = scale_and_center * pixel_to_ray;

        #[derive(BufferContents)]
        #[repr(C)]
        struct PushConstants {
            pixel_to_ray: Matrix4<f32>,
            voxel_resolution: u32,
            render_mode: u32,
        }
        let push_constants = PushConstants {
            pixel_to_ray: pixel_to_ray.cast(),
            voxel_resolution: self.voxel_resolution,
            render_mode: self.render_mode as u32,
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .clear_color_image(ClearColorImageInfo::image(rcx.render_image.clone()))
            .unwrap();

        builder
            .bind_pipeline_compute(self.render_pipeline.clone())
            .unwrap()
            .push_constants(self.render_pipeline.layout().clone(), 0, push_constants)
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.render_pipeline.layout().clone(),
                0,
                vec![rcx.render_set.clone(), self.voxel_set.clone()],
            )
            .unwrap();
        unsafe {
            builder
                .dispatch([
                    render_extent[0].div_ceil(8),
                    render_extent[1].div_ceil(8),
                    1,
                ])
                .unwrap();
        }

        builder
            .bind_pipeline_compute(self.resample_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.resample_pipeline.layout().clone(),
                0,
                vec![rcx.resample_set.clone()],
            )
            .unwrap();

        unsafe {
            builder
                .dispatch([
                    resample_extent[0].div_ceil(8),
                    resample_extent[1].div_ceil(8),
                    1,
                ])
                .unwrap();
        }

        let mut info = BlitImageInfo::images(
            rcx.resample_image.clone(),
            rcx.image_views[image_index as usize].image().clone(),
        );
        info.filter = Filter::Nearest;
        builder.blit_image(info).unwrap();

        let command_buffer = builder.build().unwrap();

        let render_future = acquire_future
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap();

        let gui_future = rcx
            .gui
            .draw_on_image(render_future, rcx.image_views[image_index as usize].clone());

        gui_future
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(rcx.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_inner_size(INITIAL_WINDOW_RESOLUTION)
                        .with_window_icon(Some(load_icon(include_bytes!("../assets/icon.png"))))
                        .with_title("Voxel Ray Traversal"),
                )
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();

        let (swapchain, images) = get_swapchain_images(&self.device, &surface, &window);
        let image_views = images
            .iter()
            .map(|i| ImageView::new(i.clone(), ImageViewCreateInfo::from_image(i)).unwrap())
            .collect::<Vec<_>>();

        let window_extent: [u32; 2] = window.inner_size().into();
        let render_extent = [
            (window_extent[0] as f32 * self.render_scale) as u32,
            (window_extent[1] as f32 * self.render_scale) as u32,
        ];
        let (render_image, render_set, resample_image, resample_set) = get_images_and_sets(
            self.memory_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            &self.render_pipeline,
            &self.resample_pipeline,
            render_extent,
            window_extent,
        );

        let gui = Gui::new(
            event_loop,
            surface,
            self.queue.clone(),
            swapchain.image_format(),
            GuiConfig {
                is_overlay: true,
                ..Default::default()
            },
        );

        let recreate_swapchain = false;

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            image_views,

            render_image,
            render_set,
            resample_image,
            resample_set,

            gui,

            recreate_swapchain,
        });
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {
        self.input.step();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if !self.rcx.as_mut().unwrap().gui.update(&event) {
            self.input.process_window_event(&event);
        }

        if event == WindowEvent::RedrawRequested {
            self.render(event_loop);
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        self.input.process_device_event(&event);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.input.end_step();
        self.update(event_loop);
        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app).unwrap();
}
