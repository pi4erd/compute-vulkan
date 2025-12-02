use compute_runtime::*;

fn main() {
    pretty_env_logger::init();

    let debug = cfg!(debug_assertions);

    let mut state = ComputeState::new(debug)
        .expect("Failed to initialize Vulkan compute instance");
    
    let code_text = include_str!("../shaders/test_compute.comp");
    
    let code_binary = compile_shader(
        "",
        &code_text,
        shaderc::ShaderKind::Compute,
    ).expect("Failed to compile shader");

    let data: BatchData<i32> = BatchData {
        array: (0..(1000 * 100000)).collect::<Vec<_>>(),
    };

    let batch = BatchInfo {
        code: &[
            BatchCode {
                code: code_binary.as_binary_u8(),
                batch_group_count: (100000, 1, 1),
            }
        ],
        buffers: &[
            BatchBufferInfo {
                buffer_binding: 0,
                buffer_size: data.size_of() as u64,
                input: Some(&data),
                host_mapped: true,
            },
        ]
    };

    let buffers = state
        .prepare_buffers(&batch)
        .expect("Failed to create buffers");

    log::info!("Dispatching compute!");
    for i in 0..100 {
        state
            .dispatch_compute(&batch, &buffers)
            .expect("Failed to execute batch");
        log::info!("Finished iteration {i}");
    }

    let data = {
        let map = state.map_buffer(&buffers[0]).unwrap();
        
        map.read::<i32>().to_vec()
    };

    println!("{:?}", &data[data.len() - 100..data.len()]);

    println!("Finished batch execution!");
}
