//! Structured configuration for the service.
//!
//! [`Config`] is the single place that answers "what should the service do?" — which
//! address to bind, which embedding provider, LLM, and vector store to build, how much
//! to log, and how retrieval should behave. It is assembled from three layers, each
//! overriding the one before it:
//!
//! 1. **Defaults**, from [`Config::default`]. Every field has one, so a service with no
//!    config file and no environment variables still starts.
//! 2. **A config file**, JSON, loaded by [`Config::from_file`]. Any key may be omitted;
//!    what is missing keeps its default.
//! 3. **Environment variables**, applied by [`Config::apply_env`]. Every setting has a
//!    `RAG_`-prefixed variable, so a deployment can override one value without shipping
//!    a file.
//!
//! Command-line flags, where the binary offers them, sit above all three: they are the
//! most specific statement of intent, so they win.
//!
//! [`Config::validate`] is the startup gate. It is called by [`Config::load`] and should
//! be called again after applying any CLI overrides, so that a misconfigured service
//! fails immediately with a message naming the offending setting — rather than binding a
//! port and failing on the first request.
//!
//! ## Example
//!
//! ```
//! use penr_oz_ai_rag_service::{Config, EmbeddingProviderKind, LogLevel};
//!
//! let mut config = Config::default();
//! assert_eq!(config.server.port, 8080);
//! assert_eq!(config.embedding.provider, EmbeddingProviderKind::Mock);
//!
//! // Environment variables override whatever the defaults and file produced.
//! config
//!     .apply_env_with(|name| match name {
//!         "RAG_SERVER_PORT" => Some("9000".to_string()),
//!         "RAG_LOG_LEVEL" => Some("debug".to_string()),
//!         _ => None,
//!     })
//!     .unwrap();
//!
//! assert_eq!(config.server.port, 9000);
//! assert_eq!(config.logging.level, LogLevel::Debug);
//! config.validate().unwrap();
//! ```

use std::fmt;
use std::io;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::generation::DEFAULT_MIN_SCORE;
use crate::retrieval::DEFAULT_MAX_QUERY_CHARS;

/// The config file consulted when the caller does not name one.
///
/// A missing file at this path is not an error — it means "use defaults". A file the
/// caller asked for by name, and which is missing, *is* an error: they meant it.
pub const DEFAULT_CONFIG_PATH: &str = "rag.config.json";

/// The environment variable naming a config file, overriding [`DEFAULT_CONFIG_PATH`].
pub const CONFIG_PATH_ENV: &str = "RAG_CONFIG";

/// Anything that stops a usable [`Config`] from being assembled.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The config file could not be read.
    #[error("failed to read config file `{path}`: {source}")]
    Read {
        /// The file that could not be read.
        path: PathBuf,
        /// The underlying I/O failure.
        source: io::Error,
    },

    /// The config file was read but is not valid JSON, or has an unknown/ill-typed key.
    #[error("failed to parse config file `{path}`: {source}")]
    Parse {
        /// The file that could not be parsed.
        path: PathBuf,
        /// The underlying deserialization failure.
        source: serde_json::Error,
    },

    /// An environment variable was set to something unusable.
    #[error("environment variable {name}=`{value}` is not valid: {reason}")]
    Env {
        /// The variable that was set.
        name: String,
        /// What it was set to.
        value: String,
        /// Why that value was rejected, including the accepted alternatives.
        reason: String,
    },

    /// The assembled config is internally invalid.
    #[error("invalid configuration: {0}")]
    Invalid(String),
}

/// Every setting the service reads, with a default for each.
///
/// Deserialization fills omitted keys from [`Default`] and rejects unknown ones, so a
/// typo in a config file is reported instead of silently ignored.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Where the HTTP server listens.
    pub server: ServerConfig,
    /// Which embedding backend to build.
    pub embedding: EmbeddingConfig,
    /// Which language-model backend to build.
    pub llm: LlmConfig,
    /// Which vector store to index into.
    pub vector_store: VectorStoreConfig,
    /// How much to log, and in what shape.
    pub logging: LoggingConfig,
    /// How retrieval behaves by default.
    pub retrieval: RetrievalConfig,
}

impl Config {
    /// Assemble a config from defaults, then a file, then the environment, and validate it.
    ///
    /// The file is `path` when given; otherwise the one named by `RAG_CONFIG`; otherwise
    /// [`DEFAULT_CONFIG_PATH`] if it happens to exist. A file named explicitly — by
    /// argument or by `RAG_CONFIG` — must exist, because asking for a file that is not
    /// there is a mistake worth reporting rather than silently ignoring.
    pub fn load(path: Option<&Path>) -> Result<Self, ConfigError> {
        let explicit = path.map(PathBuf::from).or_else(|| {
            std::env::var(CONFIG_PATH_ENV)
                .ok()
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        });

        let mut config = match explicit {
            Some(path) => Self::from_file(&path)?,
            None => {
                let default_path = Path::new(DEFAULT_CONFIG_PATH);
                if default_path.is_file() {
                    Self::from_file(default_path)?
                } else {
                    Self::default()
                }
            }
        };

        config.apply_env()?;
        config.validate()?;
        Ok(config)
    }

    /// Read a JSON config file, filling omitted keys from [`Default`].
    ///
    /// Does not consult the environment and does not validate; [`Config::load`] does both.
    pub fn from_file(path: &Path) -> Result<Self, ConfigError> {
        let contents = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        serde_json::from_str(&contents).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Override settings from `RAG_`-prefixed environment variables.
    pub fn apply_env(&mut self) -> Result<(), ConfigError> {
        self.apply_env_with(|name| std::env::var(name).ok())
    }

    /// Override settings from `lookup` instead of the real environment.
    ///
    /// The process environment is global and shared between concurrently running tests,
    /// so taking the lookup as a parameter is what makes environment handling testable
    /// without serializing the suite.
    ///
    /// An empty value is treated as unset: exporting `RAG_SERVER_HOST=` is far more often
    /// a shell interpolation that produced nothing than a deliberate request to bind the
    /// empty string.
    pub fn apply_env_with(
        &mut self,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<(), ConfigError> {
        let get = |name: &str| lookup(name).filter(|value| !value.trim().is_empty());

        if let Some(value) = get("RAG_SERVER_HOST") {
            self.server.host = value;
        }
        if let Some(value) = get("RAG_SERVER_PORT") {
            self.server.port = parse_env("RAG_SERVER_PORT", &value, "a port between 0 and 65535")?;
        }
        if let Some(value) = get("RAG_EMBEDDING_PROVIDER") {
            self.embedding.provider = parse_env_choice("RAG_EMBEDDING_PROVIDER", &value)?;
        }
        if let Some(value) = get("RAG_LLM_PROVIDER") {
            self.llm.provider = parse_env_choice("RAG_LLM_PROVIDER", &value)?;
        }
        if let Some(value) = get("RAG_VECTOR_STORE") {
            self.vector_store.kind = parse_env_choice("RAG_VECTOR_STORE", &value)?;
        }
        if let Some(value) = get("RAG_LOG_LEVEL") {
            self.logging.level = parse_env_choice("RAG_LOG_LEVEL", &value)?;
        }
        if let Some(value) = get("RAG_LOG_FORMAT") {
            self.logging.format = parse_env_choice("RAG_LOG_FORMAT", &value)?;
        }
        if let Some(value) = get("RAG_RETRIEVAL_MIN_SCORE") {
            self.retrieval.min_score = parse_env(
                "RAG_RETRIEVAL_MIN_SCORE",
                &value,
                "a number between -1 and 1",
            )?;
        }
        if let Some(value) = get("RAG_RETRIEVAL_MAX_QUERY_CHARS") {
            self.retrieval.max_query_chars = parse_env(
                "RAG_RETRIEVAL_MAX_QUERY_CHARS",
                &value,
                "a positive integer",
            )?;
        }

        Ok(())
    }

    /// Reject a config the service could not honour, naming the offending setting.
    ///
    /// Called by [`Config::load`]; call it again after applying CLI overrides, since
    /// those bypass the layers that were already checked.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.addr()?;

        if self.retrieval.max_query_chars == 0 {
            return Err(ConfigError::Invalid(
                "retrieval.max_query_chars must be at least 1, or every query is rejected"
                    .to_string(),
            ));
        }
        if !self.retrieval.min_score.is_finite() {
            return Err(ConfigError::Invalid(format!(
                "retrieval.min_score must be a finite number, got {}",
                self.retrieval.min_score
            )));
        }
        if !(-1.0..=1.0).contains(&self.retrieval.min_score) {
            return Err(ConfigError::Invalid(format!(
                "retrieval.min_score must be between -1 and 1 (the range of cosine \
                 similarity), got {}",
                self.retrieval.min_score
            )));
        }

        Ok(())
    }

    /// The address the server should bind, built from `server.host` and `server.port`.
    ///
    /// Returns [`ConfigError::Invalid`] if the host is not an IP address.
    pub fn addr(&self) -> Result<SocketAddr, ConfigError> {
        let text = if self.server.host.contains(':') {
            // An IPv6 literal needs brackets before a port can be appended to it.
            format!("[{}]:{}", self.server.host, self.server.port)
        } else {
            format!("{}:{}", self.server.host, self.server.port)
        };
        text.parse().map_err(|_| {
            ConfigError::Invalid(format!(
                "server.host must be an IP address such as 127.0.0.1 or 0.0.0.0, got `{}`",
                self.server.host
            ))
        })
    }
}

/// Where the HTTP server listens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// IP address to bind. Defaults to loopback, so an unconfigured service is not
    /// exposed to the network by accident.
    pub host: String,
    /// Port to bind. `0` picks a free port, which is how tests avoid collisions.
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
        }
    }
}

/// Which embedding backend to build.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// The provider implementation.
    pub provider: EmbeddingProviderKind,
}

/// Which language-model backend to build.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmConfig {
    /// The provider implementation.
    pub provider: LlmProviderKind,
}

/// Which vector store to index into.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VectorStoreConfig {
    /// The store implementation.
    pub kind: VectorStoreKind,
}

/// How much to log, and in what shape.
///
/// The settings are defined and validated here; wiring them to an actual subscriber is
/// the job of the tracing work tracked separately.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggingConfig {
    /// The most verbose level that is emitted.
    pub level: LogLevel,
    /// Human-readable lines or one JSON object per record.
    pub format: LogFormat,
}

/// How the served retriever and answer generator are built.
///
/// `top_k` is deliberately absent: it is chosen per request, so a server-wide setting
/// for it would be config that never takes effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RetrievalConfig {
    /// The similarity a chunk must reach to be used as answer context. Requests may
    /// override it per call.
    pub min_score: f32,
    /// The longest query accepted, in characters, before a request is rejected.
    pub max_query_chars: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            min_score: DEFAULT_MIN_SCORE,
            max_query_chars: DEFAULT_MAX_QUERY_CHARS,
        }
    }
}

/// Generate an enum whose variants name interchangeable backends.
///
/// Each gets `Default`, serde in `snake_case`, a `Display`/`FromStr` pair that agree with
/// the serde spelling, and a `VARIANTS` list so an error can tell the reader what was
/// actually on offer.
macro_rules! choice_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident => $text:literal,
            )+
        }
        default = $default:ident;
        label = $label:literal;
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(rename_all = "snake_case")]
        pub enum $name {
            $(
                $(#[$variant_meta])*
                $variant,
            )+
        }

        impl $name {
            /// Every accepted spelling, in declaration order.
            pub const VARIANTS: &'static [&'static str] = &[$($text),+];

            /// What this kind is called in config files and environment variables.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $text,)+
                }
            }

            /// The noun used when reporting an unknown value.
            pub const fn label() -> &'static str {
                $label
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::$default
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl FromStr for $name {
            type Err = String;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match value {
                    $($text => Ok(Self::$variant),)+
                    _ => Err(format!(
                        "unknown {} `{}`; expected one of: {}",
                        $label,
                        value,
                        Self::VARIANTS.join(", ")
                    )),
                }
            }
        }
    };
}

choice_enum! {
    /// The embedding backend to build.
    ///
    /// Only the deterministic mock exists today; a real provider is added by
    /// implementing [`EmbeddingProvider`](crate::EmbeddingProvider) and giving it a
    /// variant here.
    pub enum EmbeddingProviderKind {
        /// [`MockEmbeddingProvider`](crate::MockEmbeddingProvider): deterministic
        /// hash-based vectors, so similarity is reproducible but not semantic.
        Mock => "mock",
    }
    default = Mock;
    label = "embedding provider";
}

choice_enum! {
    /// The language-model backend to build.
    pub enum LlmProviderKind {
        /// [`MockLlmProvider`](crate::MockLlmProvider): echoes the grounded prompt back.
        Mock => "mock",
    }
    default = Mock;
    label = "LLM provider";
}

choice_enum! {
    /// The vector store to index into.
    pub enum VectorStoreKind {
        /// [`InMemoryVectorStore`](crate::InMemoryVectorStore): in-process, lost on exit.
        InMemory => "in_memory",
    }
    default = InMemory;
    label = "vector store";
}

choice_enum! {
    /// The most verbose log level that is emitted.
    pub enum LogLevel {
        /// Failures only.
        Error => "error",
        /// Failures and warnings.
        Warn => "warn",
        /// Lifecycle events: startup, shutdown, per-request summaries.
        Info => "info",
        /// Detail useful while developing.
        Debug => "debug",
        /// Everything, including per-stage internals.
        Trace => "trace",
    }
    default = Info;
    label = "log level";
}

choice_enum! {
    /// The shape of each log record.
    pub enum LogFormat {
        /// Human-readable lines.
        Text => "text",
        /// One JSON object per record, for log shipping.
        Json => "json",
    }
    default = Text;
    label = "log format";
}

/// Parse an environment variable into `T`, reporting what was expected when it fails.
fn parse_env<T: FromStr>(name: &str, value: &str, expected: &str) -> Result<T, ConfigError> {
    value.trim().parse().map_err(|_| ConfigError::Env {
        name: name.to_string(),
        value: value.to_string(),
        reason: format!("expected {expected}"),
    })
}

/// Parse an environment variable into one of a fixed set of choices.
fn parse_env_choice<T: FromStr<Err = String>>(name: &str, value: &str) -> Result<T, ConfigError> {
    value
        .trim()
        .parse()
        .map_err(|reason: String| ConfigError::Env {
            name: name.to_string(),
            value: value.to_string(),
            reason,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let owned: Vec<(String, String)> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name| {
            owned
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
        }
    }

    #[test]
    fn defaults_are_usable_without_any_configuration() {
        let config = Config::default();
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.embedding.provider, EmbeddingProviderKind::Mock);
        assert_eq!(config.llm.provider, LlmProviderKind::Mock);
        assert_eq!(config.vector_store.kind, VectorStoreKind::InMemory);
        assert_eq!(config.logging.level, LogLevel::Info);
        assert_eq!(config.logging.format, LogFormat::Text);
        assert_eq!(config.retrieval.max_query_chars, DEFAULT_MAX_QUERY_CHARS);
        config.validate().unwrap();
    }

    #[test]
    fn default_addr_is_loopback() {
        assert_eq!(
            Config::default().addr().unwrap().to_string(),
            "127.0.0.1:8080"
        );
    }

    #[test]
    fn omitted_file_keys_keep_their_defaults() {
        let config: Config = serde_json::from_str(r#"{"server": {"port": 9999}}"#).unwrap();
        assert_eq!(config.server.port, 9999);
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.retrieval.max_query_chars, DEFAULT_MAX_QUERY_CHARS);
    }

    #[test]
    fn unknown_file_key_is_rejected_rather_than_ignored() {
        let err = serde_json::from_str::<Config>(r#"{"serverr": {"port": 1}}"#).unwrap_err();
        assert!(err.to_string().contains("unknown field"), "{err}");
    }

    #[test]
    fn unknown_provider_in_file_names_the_alternatives() {
        let err =
            serde_json::from_str::<Config>(r#"{"embedding": {"provider": "ollama"}}"#).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("unknown variant"), "{message}");
        assert!(message.contains("mock"), "{message}");
    }

    #[test]
    fn config_round_trips_through_json() {
        let config = Config::default();
        let text = serde_json::to_string(&config).unwrap();
        assert_eq!(serde_json::from_str::<Config>(&text).unwrap(), config);
    }

    #[test]
    fn env_overrides_every_setting() {
        let mut config = Config::default();
        config
            .apply_env_with(env(&[
                ("RAG_SERVER_HOST", "0.0.0.0"),
                ("RAG_SERVER_PORT", "9000"),
                ("RAG_EMBEDDING_PROVIDER", "mock"),
                ("RAG_LLM_PROVIDER", "mock"),
                ("RAG_VECTOR_STORE", "in_memory"),
                ("RAG_LOG_LEVEL", "trace"),
                ("RAG_LOG_FORMAT", "json"),
                ("RAG_RETRIEVAL_MIN_SCORE", "0.25"),
                ("RAG_RETRIEVAL_MAX_QUERY_CHARS", "1024"),
            ]))
            .unwrap();

        assert_eq!(config.addr().unwrap().to_string(), "0.0.0.0:9000");
        assert_eq!(config.logging.level, LogLevel::Trace);
        assert_eq!(config.logging.format, LogFormat::Json);
        assert_eq!(config.retrieval.min_score, 0.25);
        assert_eq!(config.retrieval.max_query_chars, 1024);
        config.validate().unwrap();
    }

    #[test]
    fn env_overrides_the_file_layer() {
        let mut config: Config = serde_json::from_str(r#"{"server": {"port": 1234}}"#).unwrap();
        config
            .apply_env_with(env(&[("RAG_SERVER_PORT", "4321")]))
            .unwrap();
        assert_eq!(config.server.port, 4321);
    }

    #[test]
    fn unset_env_leaves_settings_untouched() {
        let mut config = Config::default();
        config.apply_env_with(|_| None).unwrap();
        assert_eq!(config, Config::default());
    }

    #[test]
    fn empty_env_value_is_treated_as_unset() {
        let mut config = Config::default();
        config
            .apply_env_with(env(&[("RAG_SERVER_HOST", "   ")]))
            .unwrap();
        assert_eq!(config.server.host, "127.0.0.1");
    }

    #[test]
    fn surrounding_whitespace_in_env_values_is_tolerated() {
        let mut config = Config::default();
        config
            .apply_env_with(env(&[
                ("RAG_SERVER_PORT", " 9100 "),
                ("RAG_LOG_LEVEL", " debug "),
            ]))
            .unwrap();
        assert_eq!(config.server.port, 9100);
        assert_eq!(config.logging.level, LogLevel::Debug);
    }

    #[test]
    fn unparseable_env_number_names_the_variable_and_expectation() {
        let mut config = Config::default();
        let err = config
            .apply_env_with(env(&[("RAG_SERVER_PORT", "http")]))
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("RAG_SERVER_PORT"), "{message}");
        assert!(message.contains("http"), "{message}");
        assert!(message.contains("between 0 and 65535"), "{message}");
    }

    #[test]
    fn unknown_env_choice_lists_the_accepted_values() {
        let mut config = Config::default();
        let err = config
            .apply_env_with(env(&[("RAG_LOG_LEVEL", "verbose")]))
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("RAG_LOG_LEVEL"), "{message}");
        assert!(message.contains("unknown log level"), "{message}");
        assert!(
            message.contains("error, warn, info, debug, trace"),
            "{message}"
        );
    }

    #[test]
    fn validate_rejects_a_non_ip_host() {
        let mut config = Config::default();
        config.server.host = "localhost".to_string();
        let message = config.validate().unwrap_err().to_string();
        assert!(
            message.contains("server.host must be an IP address"),
            "{message}"
        );
        assert!(message.contains("localhost"), "{message}");
    }

    #[test]
    fn validate_rejects_zero_max_query_chars() {
        let mut config = Config::default();
        config.retrieval.max_query_chars = 0;
        let message = config.validate().unwrap_err().to_string();
        assert!(
            message.contains("retrieval.max_query_chars must be at least 1"),
            "{message}"
        );
    }

    #[test]
    fn validate_rejects_a_min_score_outside_cosine_range() {
        let mut config = Config::default();
        config.retrieval.min_score = 1.5;
        let message = config.validate().unwrap_err().to_string();
        assert!(message.contains("between -1 and 1"), "{message}");
    }

    #[test]
    fn validate_rejects_a_non_finite_min_score() {
        let mut config = Config::default();
        config.retrieval.min_score = f32::NAN;
        let message = config.validate().unwrap_err().to_string();
        assert!(message.contains("must be a finite number"), "{message}");
    }

    #[test]
    fn ipv6_hosts_are_bracketed_before_binding() {
        let mut config = Config::default();
        config.server.host = "::1".to_string();
        config.server.port = 8080;
        assert_eq!(config.addr().unwrap().to_string(), "[::1]:8080");
    }

    #[test]
    fn port_zero_is_allowed_so_tests_can_bind_a_free_port() {
        let mut config = Config::default();
        config.server.port = 0;
        config.validate().unwrap();
    }

    #[test]
    fn missing_file_reports_the_path() {
        let err = Config::from_file(Path::new("definitely/not/here.json")).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("failed to read config file"), "{message}");
        assert!(message.contains("definitely/not/here.json"), "{message}");
    }

    #[test]
    fn choice_enums_round_trip_through_their_text_form() {
        assert_eq!("in_memory".parse(), Ok(VectorStoreKind::InMemory));
        assert_eq!(VectorStoreKind::InMemory.to_string(), "in_memory");
        assert_eq!(LogFormat::Json.to_string(), "json");
        assert_eq!(LogLevel::VARIANTS.len(), 5);
    }
}
