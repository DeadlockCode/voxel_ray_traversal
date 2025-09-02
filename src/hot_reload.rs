use std::{
    fs::File,
    io::Read,
    ops::Deref,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use vulkano::{
    device::{Device, DeviceOwned},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
};

fn compile_to_spirv(
    path: &Path,
    kind: shaderc::ShaderKind,
    entry_point_name: &str,
) -> Result<shaderc::CompilationArtifact, shaderc::Error> {
    let mut f = File::open(path).unwrap();
    let mut source_text = String::new();
    f.read_to_string(&mut source_text).unwrap();

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    options.add_macro_definition("EP", Some(entry_point_name));
    compiler.compile_into_spirv(
        &source_text,
        kind,
        path.to_str().unwrap(),
        entry_point_name,
        Some(&options),
    )
}

fn get_pipeline(shader_module: Arc<ShaderModule>) -> Arc<ComputePipeline> {
    let device = shader_module.device().clone();
    let entry_point = shader_module.entry_point("main").unwrap();

    let stage = PipelineShaderStageCreateInfo::new(entry_point);

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

pub struct HotReloadComputePipeline {
    pipeline: Arc<ComputePipeline>,
    reload: Arc<AtomicBool>,
    path: PathBuf,
    _watcher: RecommendedWatcher,
}

impl Deref for HotReloadComputePipeline {
    type Target = Arc<ComputePipeline>;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

impl HotReloadComputePipeline {
    pub fn new(device: Arc<Device>, path: &Path) -> Self {
        let reload = Arc::<AtomicBool>::default();
        let cloned_reload = reload.clone();
        let mut watcher =
            notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
                if let Ok(event) = res {
                    if event.kind.is_modify() {
                        cloned_reload.store(true, Ordering::Relaxed);
                    }
                }
            })
            .unwrap();

        watcher.watch(path, RecursiveMode::NonRecursive).unwrap();

        let artifact = compile_to_spirv(path, shaderc::ShaderKind::Compute, "main").unwrap();

        let shader_module = unsafe {
            ShaderModule::new(device, ShaderModuleCreateInfo::new(artifact.as_binary())).unwrap()
        };

        let pipeline = get_pipeline(shader_module);

        Self {
            pipeline,
            reload,
            path: path.to_path_buf(),
            _watcher: watcher,
        }
    }

    pub fn maybe_reload(&mut self) {
        if self.reload.swap(false, Ordering::Relaxed) {
            let artifact = match compile_to_spirv(&self.path, shaderc::ShaderKind::Compute, "main")
            {
                Ok(artifact) => artifact,
                Err(e) => {
                    eprint!("{}", e);
                    return;
                }
            };

            let shader_module = unsafe {
                ShaderModule::new(
                    self.pipeline.device().clone(),
                    ShaderModuleCreateInfo::new(artifact.as_binary()),
                )
                .unwrap()
            };

            let new_pipeline = get_pipeline(shader_module);

            let num_sets = self.pipeline.layout().set_layouts().len() as u32;
            let new_num_sets = new_pipeline.layout().set_layouts().len() as u32;
            if num_sets != new_num_sets
                || !new_pipeline
                    .layout()
                    .is_compatible_with(self.pipeline.layout(), num_sets)
            {
                eprintln!(
                    "{} layout is not compatible with pipeline.",
                    self.path.display()
                );
                return;
            }

            self.pipeline = new_pipeline;
            eprintln!("{} successfully reloaded", self.path.display());
        }
    }
}
