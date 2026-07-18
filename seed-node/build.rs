fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(&["../proto/inference.proto", "../proto/governance.proto"], &["../proto"])?;
    Ok(())
}
