use std::sync::{Arc, RwLock};

use vkl::vk;

extern crate vkl;

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

fn main() {
    pretty_env_logger::init();

    let entry = vkl::Entry::dynamic().expect("Failed to initialize Vulkan entry");

    let instance_extensions = vec![
        c"VK_EXT_debug_utils"
    ];
    let layers = vec![
        c"VK_LAYER_KHRONOS_validation"
    ];
    let mut instance = vkl::Instance::new(entry, instance_extensions, layers, false)
        .expect("Failed to initialize Vulkan instance");

    instance.create_messenger(
        vkl::DebugUtilsMessageSeverity::VERBOSE
            | vkl::DebugUtilsMessageSeverity::INFO
            | vkl::DebugUtilsMessageSeverity::WARNING
            | vkl::DebugUtilsMessageSeverity::ERROR,
        vkl::DebugUtilsMessageType::GENERAL
            | vkl::DebugUtilsMessageType::PERFORMANCE
            | vkl::DebugUtilsMessageType::VALIDATION,
        debug_callback,
    ).expect("Failed to initialize debug messenger");

    let device_extensions = vkl::DeviceExtensions {
        ..Default::default()
    };
    let device_features = vkl::DeviceFeatures {
        ..Default::default()
    };
    instance.create_device(device_extensions, device_features)
        .expect("Failed to initialize Vulkan device");

    let device = instance.device();

    let mut piler = device.create_pipeline_manager();

    let cmd_buffer = device.allocate_command_buffer(
        vkl::QueueType::Compute,
        vk::CommandBufferLevel::PRIMARY
    ).unwrap();

    let allocator: vkl::DefaultAllocator = Arc::new(RwLock::new(vkl::Allocator::new(&instance, device)));

    let array = (0..100000).collect::<Vec<_>>();
    let buffer = {
        let buffer_info = vkl::BufferInfo {
            data: &array,
            queue_types: &[vkl::QueueType::Compute],
            buffer_usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            buffer_flags: vk::BufferCreateFlags::empty(),
            memory_properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            alloc_flags: vk::MemoryAllocateFlags::empty(),
        };

        vkl::Buffer::new(allocator.clone(), device, &buffer_info).unwrap()
    };
 
    let descriptor_pool = {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(10),
        ];

        vkl::DescriptorPool::new(
            device,
            &pool_sizes,
            1,
            vk::DescriptorPoolCreateFlags::empty(),
        ).unwrap()
    };

    let (set_layout, descriptor_set) = {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_count(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);
        let set_layout = vkl::DescriptorSetLayout::new(device, &set_layout_info).unwrap();
        let pool = descriptor_pool.allocate_descriptor_set(&set_layout).unwrap();

        (set_layout, pool)
    };

    {
        let buffer_info = [buffer.get_description()];
        let descriptor_set_write = vk::WriteDescriptorSet::default()
            .buffer_info(&buffer_info)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .dst_set(descriptor_set)
            .dst_binding(0);
        descriptor_pool.write_descriptor_sets(&[descriptor_set_write], &[]);
    }

    let (compute_pipeline, layout) = {
        let code = include_bytes!("../shaders/test.comp.spv");
        let module = piler.create_shader_module(code).unwrap();

        let compute_stage = vkl::PipelineStage {
            module: &module,
            entrypoint: c"main",
            stage: vk::ShaderStageFlags::COMPUTE,
            flags: vk::PipelineShaderStageCreateFlags::empty(),
        };

        let layout = piler.create_pipeline_layout(
            &[set_layout.layout()],
            &[]
        )
            .expect("Failed to create pipeline layout");
        let compute_pipeline_info = vkl::ComputePipelineInfo {
            layout_ref: layout,
            stage: compute_stage,
            flags: vk::PipelineCreateFlags::empty(),
        };

        (
            piler.create_compute_pipeline(&compute_pipeline_info)
                .expect("Failed to create compute pipeline"),
            layout,
        )
    };

    {
        let encoder = device.create_command_encoder(
            *cmd_buffer,
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        ).unwrap();

        let compute = encoder.begin_compute_pass();

        compute.bind_pipeline(piler.get_pipeline(compute_pipeline));
        compute.bind_descriptor_sets(
            piler.get_layout(layout),
            vk::PipelineBindPoint::COMPUTE,
            0,
            &[descriptor_set]
        );
        compute.dispatch((array.len() as u32, 1, 1));

        compute.finish();
        encoder.finish();
    }

    let cmd_buffers_to_submit = [*cmd_buffer];
    let submit_info = vk::SubmitInfo::default()
        .command_buffers(&cmd_buffers_to_submit);
    device.queue_submit(
        vkl::QueueType::Compute,
        &[submit_info],
        None
    ).unwrap();
    device.queue_wait_idle(vkl::QueueType::Compute).unwrap();

    log::info!("Finished compute!");

    let mut end_buffer = vkl::Buffer::create(
        allocator.clone(),
        device,
        &[vkl::QueueType::Compute],
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::BufferCreateFlags::empty(),
        (array.len() * size_of::<i32>()) as u64,
    ).unwrap();
    end_buffer.allocate_memory(
        device,
        vk::MemoryPropertyFlags::HOST_COHERENT |
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        vk::MemoryAllocateFlags::empty(),
    ).unwrap();

    {
        let encoder = device.create_command_encoder(
            *cmd_buffer,
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        ).unwrap();

        encoder.copy_buffer_full(&buffer, &end_buffer);
    }

    let cmd_buffers_to_submit = [*cmd_buffer];
    let submit_info = vk::SubmitInfo::default()
        .command_buffers(&cmd_buffers_to_submit);
    device.queue_submit(
        vkl::QueueType::Compute,
        &[submit_info],
        None
    ).unwrap();
    device.queue_wait_idle(vkl::QueueType::Compute).unwrap();

    log::info!("Finished copy!");

    {
        let map = end_buffer.map(device).unwrap();
        
        let result = map.read::<i32>();

        assert!(!array.eq(result));
        
        log::info!("{:?}", result);
    }
}
