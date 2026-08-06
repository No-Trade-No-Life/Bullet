use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    if let Err(error) = run() {
        eprintln!("bullet: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse(std::env::args().skip(1))?;
    let strategy = fs::canonicalize(&arguments.strategy)?;
    let config = fs::canonicalize(&arguments.config)?;
    let cache = strategy_cache(&strategy)?;
    let package = strategy_package(&cache);
    let target = cache_root().join("target");
    fs::create_dir_all(cache.join("src"))?;
    fs::write(cache.join("Cargo.toml"), manifest(&package))?;
    fs::write(cache.join("src/main.rs"), wrapper(&strategy))?;
    let status = Command::new("cargo")
        .args(["build", "--release", "--manifest-path"])
        .arg(cache.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", &target)
        .status()?;
    if !status.success() {
        return Err("strategy compilation failed".into());
    }
    let binary = cache.join("bullet-strategy");
    fs::copy(target.join("release").join(package), &binary)?;
    println!("strategy_binary: {}", binary.display());
    let status = Command::new(binary).arg(config).status()?;
    if !status.success() {
        return Err("strategy execution failed".into());
    }
    Ok(())
}

fn manifest(package: &str) -> String {
    format!(
        "[package]\nname = \"{package}\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[dependencies]\nbullet = {{ path = \"{}\" }}\n",
        escape(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("cli has workspace parent")
                .join("bullet")
        )
    )
}
fn wrapper(strategy: &Path) -> String {
    format!(
        "#[path = \"{}\"]\nmod user_strategy;\n\nfn main() {{\n    let config = std::env::args().nth(1).expect(\"usage: bullet-strategy <config.toml>\");\n    let mut strategy = user_strategy::strategy();\n    if let Err(error) = bullet::run(config, &mut strategy) {{\n        eprintln!(\"bullet: {{error}}\");\n        std::process::exit(1);\n    }}\n}}\n",
        escape(strategy)
    )
}
fn escape(path: impl AsRef<Path>) -> String {
    path.as_ref()
        .display()
        .to_string()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
}
fn cache_root() -> PathBuf {
    std::env::temp_dir().join("bullet")
}

fn strategy_package(cache: &Path) -> String {
    format!(
        "bullet-strategy-{}",
        cache
            .file_name()
            .expect("strategy cache has a hash directory")
            .to_string_lossy()
    )
}

fn strategy_cache(strategy: &Path) -> Result<PathBuf, std::io::Error> {
    let mut hash = DefaultHasher::new();
    fs::read(strategy)?.hash(&mut hash);
    env!("CARGO_PKG_VERSION").hash(&mut hash);
    Ok(cache_root().join(format!("{:016x}", hash.finish())))
}

struct Arguments {
    strategy: PathBuf,
    config: PathBuf,
}
impl Arguments {
    fn parse(mut values: impl Iterator<Item = String>) -> Result<Self, ArgumentError> {
        if values.next().as_deref() != Some("run") {
            return Err(ArgumentError::Usage);
        }
        let strategy = values.next().ok_or(ArgumentError::Usage)?;
        if values.next().as_deref() != Some("--config") {
            return Err(ArgumentError::Usage);
        }
        let config = values.next().ok_or(ArgumentError::Usage)?;
        if values.next().is_some() {
            return Err(ArgumentError::Usage);
        }
        Ok(Self {
            strategy: strategy.into(),
            config: config.into(),
        })
    }
}
#[derive(Debug)]
enum ArgumentError {
    Usage,
}
impl std::fmt::Display for ArgumentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("usage: bullet run <strategy.rs> --config <backtest.toml>")
    }
}
impl Error for ArgumentError {}
