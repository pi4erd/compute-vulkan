# vkc-lib

A vulkan compute library.

## Warning: *very* early development

This crate is still in **extremely** early development stage.
Things **WILL** break.

Still, I'd appreciate any contributions from people who found this small
project useful.

## Usage

All the structures necessary for the job are in the top-level
module of `vkc-lib`. This means you can just

```rust
use vkc_lib::*;
```

to include everything you will need.

Start by compiling your shader(s) into `spirv` format by using
`compile_shader` function.

```rust
let code_text = "#version 460
layout (binding = 0) buffer SSBONumbers {
    int numbers[];
};

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint i = gl_GlobalInvocationID.x;

    numbers[i] += 100;
}"; // A compute shader. Can be anything

// compile_shader is part of vkc_lib crate
let code_binary = compile_shader(
    "add_number.glsl",
    code_text,
    shaderc::ShaderKind::Compute,
).expect("Failed to compile shader");
```

Create a definition for your compute task. For this you can use `BatchInfo`.

```rust
let batch = BatchInfo {
    code: &[
        BatchCode {
            code: code_binary.as_binary_u8(),
            batch_group_count: (10, 1, 1),
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
```

As you can see in `buffers` definition (`input` field), you can add
input (init) data to the task by using `BatchData` structure.

```rust
let data: BatchData<i32> = BatchData {
    array: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
};
```

After definition is done, it is time to dispatch compute.

First, initialize the compute state:

```rust
// Keep it mutable because dispatch function will require mutability.
let mut state = ComputeState::new(debug)
    .expect("Failed to initialize Vulkan compute instance");
```

Then use the next function to prepare buffers for future compute job based on
your batch info:

```rust
let buffers = state
    .prepare_buffers(&batch)
    .expect("Failed to create buffers");
```

Then, whenever you want, dispatch by using next function:

```rust
// You can dispatch same shaders multiple times by calling this function.
// Note that shaders are dispatched in order of their definition inside BatchInfo
state
    .dispatch_compute(&batch, &buffers)
    .expect("Failed to execute batch");
```

To read data back from buffers, use `map_buffer` function:

```rust
let data = {
    // Choose the buffer. Note that only buffers marked as `host_mapped`
    // can be mapped.
    let map = state.map_buffer(&buffers[0]).unwrap();
    
    map.read::<i32>().to_vec()
};
```
