use std::fs;
use std::io::{Result as IOResult, Error as IOError};
use std::path::Path;

extern crate shaderc;

use shaderc::Result as ShaderResult;

fn compile_shader<P>(path: P, shader_kind: shaderc::ShaderKind) -> ShaderResult<()>
    where P: AsRef<Path>
{
    let compiler = shaderc::Compiler::new()?; 

    let text = fs::read_to_string(path.as_ref()).unwrap();
    let artifact = compiler.compile_into_spirv(
        &text,
        shader_kind,
        &path.as_ref().to_string_lossy(),
        "main",
        None
    )?;

    let new_path = path
        .as_ref()
        .to_path_buf()
        .with_added_extension("spv");

    fs::write(new_path, artifact.as_binary_u8())
        .expect("Failed to write artifact of compilation");

    Ok(())
}

fn main() -> IOResult<()> {
    let shaders = fs::read_dir("shaders")?;

    let mut failed = false;
    for shader in shaders {
        let shader = shader?;

        if shader.file_type()?.is_file() {
            let shader_path = shader.path();

            let extension = shader_path.extension()
                .map_or(String::from(""), |s| s.to_string_lossy().to_string());

            let shader_kind = match extension.as_str() {
                "comp" => shaderc::ShaderKind::Compute,
                _ => continue,
            };

            match compile_shader(&shader_path, shader_kind) {
                Ok(_) => {},
                Err(e) => {
                    println!("Failed to compile shader {}: {e}", shader_path.to_string_lossy());
                    failed = true;
                }
            }
        }
    }

    if failed {
        return Err(IOError::new(
            std::io::ErrorKind::Other,
            "Failed to compile 1 or more shaders."
        ));
    }

    Ok(())
}