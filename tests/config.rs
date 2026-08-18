//! Cross-stage tests for configuration: the file layer on disk, and the real binary
//! booting (or refusing to boot) on what the layers resolved to.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};

use penr_oz_ai_rag_service::{
    Config, EmbeddingProviderKind, LlmProviderKind, LogFormat, LogLevel, VectorStoreKind,
};
use tempfile::tempdir;

/// Kills the served binary when the test ends, pass or fail.
struct ServerGuard(Child);

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Write `contents` to `<dir>/rag.json` and return the path.
fn write_config(dir: &std::path::Path, contents: &str) -> std::path::PathBuf {
    let path = dir.join("rag.json");
    std::fs::write(&path, contents).expect("write config file");
    path
}

/// Write a one-file corpus and return its path.
fn write_corpus(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("corpus.txt");
    std::fs::write(
        &path,
        "Retrieval augmented generation grounds answers in sources.",
    )
    .expect("write corpus");
    path
}

// --- the file layer -----------------------------------------------------------------

#[test]
fn a_partial_file_overrides_only_what_it_names() {
    let dir = tempdir().unwrap();
    let path = write_config(
        dir.path(),
        r#"{
            "server": { "port": 9137 },
            "logging": { "level": "debug", "format": "json" }
        }"#,
    );

    let config = Config::from_file(&path).expect("partial config loads");

    assert_eq!(config.server.port, 9137);
    assert_eq!(config.logging.level, LogLevel::Debug);
    assert_eq!(config.logging.format, LogFormat::Json);
    // Untouched sections keep their defaults.
    assert_eq!(config.server.host, "127.0.0.1");
    assert_eq!(config.embedding.provider, EmbeddingProviderKind::Mock);
    assert_eq!(config.llm.provider, LlmProviderKind::Mock);
    assert_eq!(config.vector_store.kind, VectorStoreKind::InMemory);
    config.validate().expect("partial config is valid");
}

#[test]
fn an_empty_file_is_the_same_as_no_file() {
    let dir = tempdir().unwrap();
    let path = write_config(dir.path(), "{}");
    assert_eq!(Config::from_file(&path).unwrap(), Config::default());
}

#[test]
fn a_full_file_round_trips_through_the_config_it_produces() {
    let dir = tempdir().unwrap();
    let written = Config::default();
    let path = write_config(
        dir.path(),
        &serde_json::to_string_pretty(&written).expect("config serializes"),
    );
    assert_eq!(Config::from_file(&path).unwrap(), written);
}

#[test]
fn a_malformed_file_names_the_path_and_the_problem() {
    let dir = tempdir().unwrap();
    let path = write_config(dir.path(), "{ not json");

    let message = Config::from_file(&path).unwrap_err().to_string();

    assert!(message.contains("failed to parse config file"), "{message}");
    assert!(message.contains("rag.json"), "{message}");
}

#[test]
fn a_typo_in_a_key_is_reported_rather_than_ignored() {
    let dir = tempdir().unwrap();
    let path = write_config(dir.path(), r#"{"server": {"prot": 8080}}"#);

    let message = Config::from_file(&path).unwrap_err().to_string();

    assert!(message.contains("unknown field `prot`"), "{message}");
}

#[test]
fn an_unknown_provider_lists_what_is_available() {
    let dir = tempdir().unwrap();
    let path = write_config(dir.path(), r#"{"llm": {"provider": "gpt"}}"#);

    let message = Config::from_file(&path).unwrap_err().to_string();

    assert!(message.contains("unknown variant `gpt`"), "{message}");
    assert!(message.contains("mock"), "{message}");
}

#[test]
fn load_reports_a_named_file_that_is_not_there() {
    let dir = tempdir().unwrap();
    let missing = dir.path().join("absent.json");

    let message = Config::load(Some(&missing)).unwrap_err().to_string();

    assert!(message.contains("failed to read config file"), "{message}");
    assert!(message.contains("absent.json"), "{message}");
}

#[test]
fn layers_apply_in_order_with_env_over_file() {
    let dir = tempdir().unwrap();
    let path = write_config(
        dir.path(),
        r#"{"server": {"port": 1111}, "retrieval": {"min_score": 0.1}}"#,
    );

    let mut config = Config::from_file(&path).unwrap();
    config
        .apply_env_with(|name| match name {
            "RAG_SERVER_PORT" => Some("2222".to_string()),
            _ => None,
        })
        .unwrap();

    assert_eq!(config.server.port, 2222, "env wins over the file");
    assert_eq!(
        config.retrieval.min_score, 0.1,
        "the file wins over defaults"
    );
    assert_eq!(config.server.host, "127.0.0.1", "defaults fill the rest");
}

// --- the binary booting on what the layers resolved to -------------------------------

/// Spawn `penr-oz-rag serve` with `args` and `envs`, returning the process and the lines
/// it printed before either binding or exiting.
fn spawn_serve(args: &[&std::ffi::OsStr], envs: &[(&str, &str)]) -> (ServerGuard, Vec<String>) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_penr-oz-rag"));
    command.arg("serve").args(args);
    for (name, value) in envs {
        command.env(name, value);
    }
    let child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("serve binary spawns");
    let mut guard = ServerGuard(child);

    let stdout = guard.0.stdout.take().expect("stdout is piped");
    let mut lines = BufReader::new(stdout).lines();
    let mut seen = Vec::new();
    for line in lines.by_ref() {
        let line = line.expect("server stdout is utf-8");
        let done = line.starts_with("Serving POST");
        seen.push(line);
        if done {
            break;
        }
    }
    // Keep draining stdout: dropping the pipe would make a later `println!` hit EPIPE
    // and panic, killing the server mid-test.
    std::thread::spawn(move || for _ in lines {});
    (guard, seen)
}

#[test]
fn the_server_binds_the_address_from_its_config_file() {
    let dir = tempdir().unwrap();
    let corpus = write_corpus(dir.path());
    // Port 0 so the test cannot collide with anything else on the machine.
    let config = write_config(
        dir.path(),
        r#"{"server": {"host": "127.0.0.1", "port": 0}}"#,
    );

    let (_guard, lines) = spawn_serve(
        &[corpus.as_os_str(), "--config".as_ref(), config.as_os_str()],
        &[],
    );

    let serving = lines
        .iter()
        .find(|line| line.starts_with("Serving POST"))
        .expect("server reports the address it bound");
    assert!(serving.contains("http://127.0.0.1:"), "{serving}");
}

#[test]
fn the_server_reports_the_settings_it_resolved() {
    let dir = tempdir().unwrap();
    let corpus = write_corpus(dir.path());
    let config = write_config(
        dir.path(),
        r#"{"server": {"port": 0}, "retrieval": {"min_score": 0.35, "max_query_chars": 512}}"#,
    );

    let (_guard, lines) = spawn_serve(
        &[corpus.as_os_str(), "--config".as_ref(), config.as_os_str()],
        &[],
    );

    let reported = lines
        .iter()
        .find(|line| line.starts_with("Config:"))
        .expect("server reports its resolved config");
    assert!(reported.contains("min_score=0.35"), "{reported}");
    assert!(reported.contains("max_query_chars=512"), "{reported}");
    assert!(reported.contains("embedding=mock"), "{reported}");
    assert!(reported.contains("vector_store=in_memory"), "{reported}");
}

#[test]
fn an_environment_variable_overrides_the_config_file() {
    let dir = tempdir().unwrap();
    let corpus = write_corpus(dir.path());
    let config = write_config(
        dir.path(),
        r#"{"server": {"port": 0}, "retrieval": {"min_score": 0.35}}"#,
    );

    let (_guard, lines) = spawn_serve(
        &[corpus.as_os_str(), "--config".as_ref(), config.as_os_str()],
        &[("RAG_RETRIEVAL_MIN_SCORE", "0.75")],
    );

    let reported = lines
        .iter()
        .find(|line| line.starts_with("Config:"))
        .expect("server reports its resolved config");
    assert!(reported.contains("min_score=0.75"), "{reported}");
}

#[test]
fn a_command_line_flag_overrides_the_environment_and_the_file() {
    let dir = tempdir().unwrap();
    let corpus = write_corpus(dir.path());
    let config = write_config(
        dir.path(),
        r#"{"server": {"port": 0}, "retrieval": {"min_score": 0.35}}"#,
    );

    let (_guard, lines) = spawn_serve(
        &[
            corpus.as_os_str(),
            "--config".as_ref(),
            config.as_os_str(),
            "--min-score".as_ref(),
            "0.9".as_ref(),
        ],
        &[("RAG_RETRIEVAL_MIN_SCORE", "0.75")],
    );

    let reported = lines
        .iter()
        .find(|line| line.starts_with("Config:"))
        .expect("server reports its resolved config");
    assert!(reported.contains("min_score=0.9"), "{reported}");
}

#[test]
fn an_invalid_setting_stops_startup_with_an_informative_error() {
    let dir = tempdir().unwrap();
    let corpus = write_corpus(dir.path());
    let config = write_config(dir.path(), r#"{"retrieval": {"min_score": 5.0}}"#);

    let output = Command::new(env!("CARGO_BIN_EXE_penr-oz-rag"))
        .arg("serve")
        .arg(&corpus)
        .arg("--config")
        .arg(&config)
        .output()
        .expect("serve binary runs");

    assert!(!output.status.success(), "startup should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid configuration"), "{stderr}");
    assert!(stderr.contains("min_score"), "{stderr}");
    assert!(stderr.contains("between -1 and 1"), "{stderr}");
    // It must fail before doing any work, not after ingesting and binding.
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("Serving POST"), "{stdout}");
    assert!(!stdout.contains("Ingested"), "{stdout}");
}

#[test]
fn an_unusable_environment_variable_stops_startup() {
    let dir = tempdir().unwrap();
    let corpus = write_corpus(dir.path());

    let output = Command::new(env!("CARGO_BIN_EXE_penr-oz-rag"))
        .arg("serve")
        .arg(&corpus)
        .env("RAG_LLM_PROVIDER", "gpt-5")
        .output()
        .expect("serve binary runs");

    assert!(!output.status.success(), "startup should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("RAG_LLM_PROVIDER"), "{stderr}");
    assert!(stderr.contains("unknown LLM provider"), "{stderr}");
    assert!(stderr.contains("mock"), "{stderr}");
}
