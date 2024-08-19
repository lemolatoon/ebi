fn main() -> miette::Result<()> {
    let path0 = std::path::PathBuf::from("ALP/include"); // include path
    let path1 = std::path::PathBuf::from("include"); // include path
    let mut b = autocxx_build::Builder::new("src/lib.rs", [&path0, &path1])
        .extra_clang_args(&["-std=c++17"])
        .build()?;
    // if CXX is not specified, and clang++ available
    if std::env::var("CXX").is_err() && which::which("clang++").is_ok() {
        b.compiler("clang++");
    }

    // This assumes all your C++ bindings are in main.rs
    b.compiler("clang++")
        .flag("-std=c++17")
        .flag_if_supported("-Wno-unused-parameter")
        .file("ALP/src/falp.cpp")
        .file("ALP/src/fastlanes_ffor.cpp")
        .file("ALP/src/fastlanes_unffor.cpp")
        .file("ALP/src/fastlanes_generated_ffor.cpp")
        .file("ALP/src/fastlanes_generated_unffor.cpp")
        .compile("alp"); // arbitrary library name, pick anything

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:return-if-changed=ALP/src/falp.cpp");
    println!("cargo:return-if-changed=ALP/src/fastlanes_ffor.cpp");
    println!("cargo:return-if-changed=ALP/src/fastlanes_generated_ffor.cpp");
    println!("cargo:return-if-changed=ALP/src/fastlanes_generated_unffor.cpp");
    println!("cargo:return-if-changed=ALP/src/fastlanes_ffor.cpp");
    // Add instructions to link to any C++ libraries you need.
    Ok(())
}
