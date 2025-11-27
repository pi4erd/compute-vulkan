use vkl::{piler::{
    PipelineCreateFlags, PipelineShaderStageCreateFlags, ShaderStageFlags
}, vk};

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

    let code = include_bytes!("../shaders/test.comp.spv");
    let module = piler.create_shader_module(code).unwrap();

    let compute_stage = vkl::PipelineStage {
        module: &module,
        entrypoint: c"main",
        stage: ShaderStageFlags::COMPUTE,
        flags: PipelineShaderStageCreateFlags::empty(),
    };

    let layout = piler.create_pipeline_layout(&[], &[])
        .expect("Failed to create pipeline layout");
    let compute_pipeline_info = vkl::ComputePipelineInfo {
        layout_ref: layout,
        stage: compute_stage,
        flags: PipelineCreateFlags::empty(),
    };
    let compute_pipeline = piler.create_compute_pipeline(&compute_pipeline_info)
        .expect("Failed to create compute pipeline");

    let cmd_buffer = device.allocate_command_buffer(
        vkl::QueueType::Compute,
        vk::CommandBufferLevel::PRIMARY
    ).unwrap();
}
