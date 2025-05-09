use anyhow::{Context, Result};
use glob::{glob, GlobError};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <glob-pattern>", args[0]);
        std::process::exit(1);
    }
    let pattern = &args[1];

    let files: Vec<Result<PathBuf, GlobError>> = glob(pattern)
        .expect("Failed to read glob pattern")
        .collect();
    let n_files = files.len();
    for (i, entry) in files.into_iter().enumerate() {
        match entry {
            Ok(path) => {
                println!(
                    "[{:03}/{:03}] Processing file: {}",
                    i,
                    n_files,
                    path.display(),
                );
                if let Err(e) = process_file(&path) {
                    eprintln!("Failed to process file {}: {:?}", path.display(), e);
                } else {
                    println!(
                        "[{:03}/{:03}] Finished processing file: {}",
                        i + 1,
                        n_files,
                        path.display(),
                    );
                }
            }
            Err(e) => eprintln!("Error reading path: {e:?}"),
        }
    }
}

fn process_file(path: &Path) -> Result<()> {
    let file = File::open(path).context(format!("Failed to open file: {}", path.display()))?;
    let reader = BufReader::new(file);

    let output_dir = path
        .parent()
        .ok_or(anyhow::anyhow!("Failed to get parent directory"))?
        .join("binary");
    std::fs::create_dir_all(&output_dir).context(format!(
        "Failed to create output directory: {}",
        output_dir.display()
    ))?;
    let output_filename = output_dir.join({
        let mut name = path
            .file_stem()
            .ok_or(anyhow::anyhow!("Failed to get file stem"))?
            .to_os_string();
        name.push(".bin");
        name
    });
    let output_file = File::create(output_filename.as_path()).context(format!(
        "Failed to create output file: {}",
        output_filename.display()
    ))?;
    let mut writer = BufWriter::new(output_file);

    let process = || {
        for (line_num, line) in reader.lines().enumerate() {
            let line = line.context("Failed to read line")?;
            let floats: Result<Vec<_>, _> = line
                .split(',')
                .filter_map(|s| {
                    if s.is_empty() {
                        None
                    } else {
                        Some(
                            s.trim()
                                .parse::<f64>()
                                .context(format!("Failed to parse float: {}:{s}", line_num + 1)),
                        )
                    }
                })
                .collect();

            match floats {
                Ok(numbers) => {
                    for num in numbers {
                        writer
                            .write_all(&num.to_le_bytes())
                            .context("Failed to write to output file")?;
                    }
                }
                Err(e) => {
                    return Err(
                        e.context(format!("Error parsing line in file: {}", path.display()))
                    );
                }
            }
        }

        Ok(())
    };

    match process() {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Error processing file: {}", path.display());
            if output_filename.exists() {
                std::fs::remove_file(output_filename).context("Failed to remove output file")?;
            }
            return Err(e);
        }
    }

    Ok(())
}
