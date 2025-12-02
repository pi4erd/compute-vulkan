pub use shaderc;

use std::{error::Error, iter::zip, sync::{Arc, RwLock}};

use bytemuck::NoUninit;
use vkl::{VklResult, vk};
use shaderc::{CompilationArtifact, Result as ShaderResult};

pub trait BatchDataTrait {
    fn get_data_bytes(&self) -> &[u8];
}

pub struct BatchData<T: NoUninit> {
    pub array: Vec<T>,
}

impl<T: NoUninit> BatchDataTrait for BatchData<T> {
    fn get_data_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.array)
    }
}

impl<T: bytemuck::NoUninit> BatchData<T> {
    pub fn size_of(&self) -> usize {
        self.array.len() * size_of::<T>()
    }
}

pub struct BatchBufferInfo<'a> {
    pub buffer_binding: u32,
    pub buffer_size: u64,
    pub input: Option<&'a dyn BatchDataTrait>,
    pub host_mapped: bool,
}

pub struct BatchCode<'a> {
    pub code: &'a [u8],
    pub batch_group_count: (u32, u32, u32),
}

pub struct BatchInfo<'a> {
    pub code: &'a [BatchCode<'a>],
    pub buffers: &'a [BatchBufferInfo<'a>],
}

pub struct ComputeState {
    piler: vkl::PipelineManager,
    allocator: vkl::DefaultAllocator,

    instance: vkl::Instance,
}

impl ComputeState {
    fn debug_callback(
        severity: vkl::DebugUtilsMessageSeverity,
        type_flags: vkl::DebugUtilsMessageType,
        data: &vkl::DebugUtilsCallbackData,
    ) -> bool {
        match severity {
            vkl::DebugUtilsMessageSeverity::ERROR =>
                log::error!("({:?}): {}", type_flags, data.message),
            vkl::DebugUtilsMessageSeverity::WARNING =>
                log::warn!("({:?}): {}", type_flags, data.message),
            vkl::DebugUtilsMessageSeverity::INFO =>
                log::debug!("({:?}): {}", type_flags, data.message),
            vkl::DebugUtilsMessageSeverity::VERBOSE =>
                log::trace!("({:?}): {}", type_flags, data.message),
            _ => log::error!("Invalid severity. ({:?}): {}", type_flags, data.message),
        }

        false
    }

    pub fn new(validation: bool) -> Result<Self, Box<dyn Error>> {
        let entry = vkl::Entry::dynamic()?;

        let mut instance_extensions = vec![];
        let mut layers = vec![];

        if validation {
            instance_extensions.push(c"VK_EXT_debug_utils");
            layers.push(c"VK_LAYER_KHRONOS_validation");
        }

        let mut instance = vkl::Instance::new(
            entry,
            instance_extensions,
            layers,
            false
        )?;

        if validation {
            instance.create_messenger(
                vkl::DebugUtilsMessageSeverity::VERBOSE
                    | vkl::DebugUtilsMessageSeverity::INFO
                    | vkl::DebugUtilsMessageSeverity::WARNING
                    | vkl::DebugUtilsMessageSeverity::ERROR,
                vkl::DebugUtilsMessageType::GENERAL
                    | vkl::DebugUtilsMessageType::PERFORMANCE
                    | vkl::DebugUtilsMessageType::VALIDATION,
                Self::debug_callback,
            )?;
        }

        let device_extensions = vkl::DeviceExtensions {
            ..Default::default()
        };
        let device_features = vkl::DeviceFeatures {
            ..Default::default()
        };
        instance.create_device(device_extensions, device_features)?;

        let device = instance.device();

        let piler = device.create_pipeline_manager();

        let allocator: vkl::DefaultAllocator = Arc::new(
            RwLock::new(vkl::Allocator::new(&instance, device))
        );

        Ok(Self {
            instance,
            piler,
            allocator,
        })
    }

    fn new_buffer_from_info(&self, info: &BatchBufferInfo) -> VklResult<vkl::Buffer> {
        let buffer_usage = if info.input.is_none() {
            vk::BufferUsageFlags::STORAGE_BUFFER
        } else {
            vk::BufferUsageFlags::STORAGE_BUFFER |
                vk::BufferUsageFlags::TRANSFER_DST
        };
        let mut buffer = vkl::Buffer::create(
            self.allocator.clone(),
            self.instance.device(),
            &[vkl::QueueType::Compute],
            buffer_usage,
            vk::BufferCreateFlags::empty(),
            info.buffer_size
        )?;
        buffer.allocate_memory(
            self.instance.device(),
            if !info.host_mapped {
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            } else {
                vk::MemoryPropertyFlags::HOST_COHERENT |
                    vk::MemoryPropertyFlags::HOST_VISIBLE
            },
            vk::MemoryAllocateFlags::empty(),
        )?;

        if let Some(input) = info.input {
            buffer.write_data_staged(
                self.instance.device(),
                input.get_data_bytes(),
            )?;
        }

        Ok(buffer)
    }

    pub fn prepare_buffers(&self, batch: &BatchInfo) -> VklResult<Vec<vkl::Buffer>> {
        batch.buffers
            .iter()
            .map(|b| Ok(self.new_buffer_from_info(b)?))
            .collect::<VklResult<Vec<_>>>()
    }

    pub fn map_buffer<'a>(&'a self, buffer: &'a vkl::Buffer) -> VklResult<vkl::MemMap<'a>> {
        buffer.map(self.instance.device())
    }

    pub fn dispatch_compute(&mut self, batch: &BatchInfo, buffers: &[vkl::Buffer]) -> VklResult<()> {
        // TODO: Optimize dispatch compute by only performing the dispatch task here
        // without needless allocations. 
        // Probably will need some kind of shared descriptor pool management?
        let pool = {
            let pool_sizes = [
                vk::DescriptorPoolSize::default()
                    .descriptor_count(batch.buffers.len() as u32)
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
            ];
            self.instance
                .device()
                .create_descriptor_pool(
                    1,
                    &pool_sizes,
                    vk::DescriptorPoolCreateFlags::empty()
                )?
        };

        let descriptor_set_layout = {
            let bindings = batch.buffers
                .iter()
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b.buffer_binding)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect::<Vec<_>>();
            let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(&bindings);
            self.instance
                .device()
                .create_descriptor_set_layout(&layout_info)?
        };

        let descriptor_set = pool.allocate_descriptor_set(&descriptor_set_layout)?;

        {
            let buffer_infos = buffers
                .iter()
                // NOTE: Array type used below is to use this in WriteDescriptorSet
                //       as buffer_info 
                .map(|b| [b.get_description()])
                .collect::<Vec<_>>();
            let writes = zip(buffer_infos.iter(), batch.buffers.iter())
                .map(|(bi, ba)| {
                    vk::WriteDescriptorSet::default()
                        .buffer_info(bi)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .dst_binding(ba.buffer_binding)
                        .dst_set(descriptor_set)
                })
                .collect::<Vec<_>>();
            pool.write_descriptor_sets(&writes, &[]);
        }

        let pipeline_layout = self.piler
            .create_pipeline_layout(
                &[descriptor_set_layout.layout()],
                &[]
            )?;

        let shader_modules = batch.code
            .iter()
            .map(|code| {
                let module = self.piler.create_shader_module(code.code)?;
                Ok(module)
            })
            .collect::<VklResult<Vec<_>>>()?;
        let compute_pipelines = shader_modules
            .into_iter()
            .map(|shader_module| {
                let pipeline_info = vkl::ComputePipelineInfo {
                    layout_ref: pipeline_layout,
                    stage: vkl::PipelineStage {
                        module: &shader_module,
                        entrypoint: c"main",
                        stage: vk::ShaderStageFlags::COMPUTE,
                        flags: vk::PipelineShaderStageCreateFlags::empty(),
                    },
                    flags: vk::PipelineCreateFlags::empty(),
                };

                let pipeline = self.piler.create_compute_pipeline(&pipeline_info)?;
                Ok(pipeline)
            })
            .collect::<VklResult<Vec<_>>>()?;

        let cmd_buffer = self.instance
            .device()
            .allocate_command_buffer(vkl::QueueType::Compute, vk::CommandBufferLevel::PRIMARY)?;

        {
            let encoder = self.instance
                .device()
                .create_command_encoder(*cmd_buffer, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;

            for (pipeline, code) in zip(compute_pipelines, batch.code) {
                let compute_pass = encoder.begin_compute_pass();

                compute_pass.bind_descriptor_sets(
                    self.piler.get_layout(pipeline_layout),
                    vk::PipelineBindPoint::COMPUTE,
                    0, &[descriptor_set]
                );

                compute_pass.bind_pipeline(self.piler.get_pipeline(pipeline));

                compute_pass.dispatch(code.batch_group_count);
            }
        }

        let submit_cmd_buffers = [*cmd_buffer];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&submit_cmd_buffers);
        self.instance
            .device()
            .queue_submit(
                vkl::QueueType::Compute,
                &[submit_info],
                None
            )?;
        self.instance
            .device()
            .queue_wait_idle(vkl::QueueType::Compute)?;
        
        // Clear after compute is done!
        self.piler.clear();

        Ok(())
    }
}

pub fn compile_shader(filename: &str, code: &str, shader_kind: shaderc::ShaderKind) -> ShaderResult<CompilationArtifact> {
    let compiler = shaderc::Compiler::new()?; 

    compiler.compile_into_spirv(
        code,
        shader_kind,
        filename,
        "main",
        None
    )
}
