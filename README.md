# penr-oz-ai-rag-service

Implementation of a Retrieval-Augmented Generation (RAG) service for AI.

This repository provides the **document ingestion pipeline** — the stage that turns raw
source files into metadata-rich, retrievable chunks — the **retrieval service** that
answers queries over them through a `POST /retrieve` endpoint, and the **answer
generation service** that grounds a language model in the retrieved context through a
`POST /answer` endpoint. It is written in Rust as a reusable library
(`penr_oz_ai_rag_service`) with a command-line front-end (`penr-oz-rag`) that both
ingests and serves.

## Overview

Ingestion is modeled as three decoupled stages, each behind a trait so it can be
replaced or extended independently:

```
 file ──▶ Loader ──▶ Document ──▶ Chunker ──▶ Chunk[] ──▶ ChunkStore ──▶ storage
          (load)                  (split +              (persist)
                                   metadata)
```

1. **Load** — a [`Loader`] reads a file and normalizes it into a `Document`. The only
   built-in loader today is `TextLoader` (`.txt`, `.text`). New formats are added by
   implementing `Loader` and registering it with a `LoaderRegistry` — **nothing
   downstream changes**, which is what keeps PDF / HTML / Markdown loaders a drop-in
   addition later.
2. **Chunk** — a `Chunker` splits a `Document` into ordered `Chunk`s and attaches
   positional and provenance metadata. The built-in `FixedSizeChunker` produces
   fixed-size, overlapping, character-based windows and prefers word boundaries.
3. **Store** — a `ChunkStore` persists the chunks. Built-in backends are
   `InMemoryStorage` and `JsonlStorage` (one JSON object per line).

`IngestionPipeline` composes the three stages and reports what it did.

## Features

- Ingest a single text file or a directory tree (walked recursively, deterministic
  order).
- Character-based chunking (correct for multi-byte/Unicode text) with configurable
  size and overlap, and optional word-boundary awareness.
- Rich per-chunk metadata: source id, chunk index, total chunks, character offsets, and
  provenance propagated from the loader (e.g. `loader`, `filename`).
- Pluggable loaders, chunkers, and storage backends via small traits.
- Meaningful, specific error messages for invalid input (missing file, unsupported
  format, non-UTF-8 data, empty document, bad chunker configuration).
- Grounded answer generation over the indexed corpus: retrieval, a confidence gate that
  keeps low-scoring chunks out of the prompt, a pluggable `LlmProvider`, and source
  references on every answer.
- Layered configuration — defaults, a JSON file, `RAG_*` environment variables, then CLI
  flags — validated at startup so a misconfigured service fails immediately.

## Requirements

- Rust 1.74 or newer (stable). Install via [rustup](https://rustup.rs/).

## Build & test

```bash
cargo build --release   # binary at target/release/penr-oz-rag
cargo test              # unit + integration + doc tests
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

## Command-line usage

```bash
penr-oz-rag ingest <INPUT> [OPTIONS]   # chunk documents (optionally persist as JSONL)
penr-oz-rag serve  <INPUT> [OPTIONS]   # chunk + index documents, then serve POST /retrieve and POST /answer
```

Options common to both commands:

| Option | Default | Description |
| --- | --- | --- |
| `<INPUT>` | — | Path to a file or directory to ingest. |
| `--chunk-size <N>` | `800` | Maximum number of characters per chunk. |
| `--overlap <N>` | `100` | Characters shared between consecutive chunks (must be `< chunk-size`). |
| `--no-word-aware` | off | Make exact character cuts instead of preferring word boundaries. |

`ingest` only:

| Option | Default | Description |
| --- | --- | --- |
| `-o, --output <FILE>` | _(none)_ | Write chunks as JSON Lines to `FILE`. If omitted, chunks are produced and counted in memory but not persisted. |

`serve` only:

| Option | Default | Description |
| --- | --- | --- |
| `--config <PATH>` | `$RAG_CONFIG`, then `./rag.config.json` | JSON config file. See [Configuration](#configuration). |
| `--addr <ADDR>` | `server.host`/`server.port` from config (`127.0.0.1:8080`) | Address to bind the HTTP server to (port `0` picks a free port). |
| `--min-score <F>` | `retrieval.min_score` from config (`0`) | Minimum similarity score a retrieved chunk must reach to be used as answer context. Requests can override it per call via `min_score`. |

### Examples

```bash
# Ingest one file and just report how many chunks it produces (in memory)
penr-oz-rag ingest ./docs/notes.txt

# Ingest a whole directory and persist chunks as JSON Lines
penr-oz-rag ingest ./docs --output chunks.jsonl --chunk-size 800 --overlap 100
```

Example run:

```text
$ penr-oz-rag ingest ./docs --output out/chunks.jsonl --chunk-size 120 --overlap 24
Ingested 2 file(s), skipped 1, created 4 chunk(s).
  ./docs/intro.txt -> 3 chunk(s)
  ./docs/notes.text -> 1 chunk(s)
Wrote 4 chunk(s) to out/chunks.jsonl
```

When ingesting a **directory**, files whose format has no registered loader are skipped
and counted (`skipped`). When ingesting a **single file**, an unsupported format is an
error, since you asked for that file explicitly. The process exits non-zero on any
error.

## Configuration

`serve` resolves its settings from four layers, each overriding the one before it:

1. **Built-in defaults** — the service starts with no configuration at all.
2. **A JSON config file** — `--config <PATH>`, else `$RAG_CONFIG`, else
   `./rag.config.json` if it exists. Any key may be omitted; what is missing keeps its
   default. A file you name explicitly must exist; the implicit `./rag.config.json` may
   not.
3. **Environment variables** — `RAG_`-prefixed, one per setting. An empty value counts
   as unset.
4. **Command-line flags** — the most specific statement of intent, so they win.

The resolved result is validated once, before ingesting or binding anything, so a
mistake surfaces at startup rather than on the first request.

| Config key | Environment variable | Default | Accepted values |
| --- | --- | --- | --- |
| `server.host` | `RAG_SERVER_HOST` | `127.0.0.1` | Any IP address (`0.0.0.0`, `::1`, …). |
| `server.port` | `RAG_SERVER_PORT` | `8080` | `0`–`65535`; `0` picks a free port. |
| `embedding.provider` | `RAG_EMBEDDING_PROVIDER` | `mock` | `mock` |
| `llm.provider` | `RAG_LLM_PROVIDER` | `mock` | `mock` |
| `vector_store.kind` | `RAG_VECTOR_STORE` | `in_memory` | `in_memory` |
| `logging.level` | `RAG_LOG_LEVEL` | `info` | `error`, `warn`, `info`, `debug`, `trace` |
| `logging.format` | `RAG_LOG_FORMAT` | `text` | `text`, `json` |
| `retrieval.min_score` | `RAG_RETRIEVAL_MIN_SCORE` | `0` | `-1` to `1` (the range of cosine similarity). |
| `retrieval.max_query_chars` | `RAG_RETRIEVAL_MAX_QUERY_CHARS` | `8192` | Any positive integer. |

`mock` and `in_memory` are the only backends that exist today; a real provider becomes a
new variant here once it implements the corresponding trait. There is no `retrieval.top_k`
setting because `top_k` is chosen per request.

`logging.*` is defined and validated but not yet wired to a subscriber — that arrives with
tracing and request logging.

```jsonc
// rag.config.json — every key is optional
{
  "server": { "host": "0.0.0.0", "port": 8080 },
  "embedding": { "provider": "mock" },
  "llm": { "provider": "mock" },
  "vector_store": { "kind": "in_memory" },
  "logging": { "level": "info", "format": "text" },
  "retrieval": { "min_score": 0.2, "max_query_chars": 8192 }
}
```

```bash
# File, overridden by an environment variable, overridden by a flag
penr-oz-rag serve ./docs --config rag.config.json
RAG_SERVER_PORT=9000 penr-oz-rag serve ./docs
penr-oz-rag serve ./docs --addr 0.0.0.0:9100 --min-score 0.3
```

Startup echoes what it resolved, so a surprising run can be traced to a layer without
re-deriving the precedence by hand:

```text
$ penr-oz-rag serve ./docs --config rag.config.json
Ingested 2 file(s), skipped 0, created 7 chunk(s).
  ./docs/intro.txt -> 4 chunk(s)
  ./docs/notes.txt -> 3 chunk(s)
Config: embedding=mock, llm=mock, vector_store=in_memory, min_score=0.2, max_query_chars=8192, log=info/text
Indexed 7 chunk(s) for retrieval.
Serving POST /retrieve and POST /answer on http://0.0.0.0:8080
```

Misconfiguration is reported with the offending setting named and the accepted values
listed, and the process exits non-zero:

```text
$ RAG_LLM_PROVIDER=gpt-5 penr-oz-rag serve ./docs
error: environment variable RAG_LLM_PROVIDER=`gpt-5` is not valid: unknown LLM provider `gpt-5`; expected one of: mock

$ penr-oz-rag serve ./docs --min-score 5
error: invalid configuration: retrieval.min_score must be between -1 and 1 (the range of cosine similarity), got 5
```

Read the same layers from library code with `Config::load`:

```rust
use penr_oz_ai_rag_service::{Config, ConfigError};

fn main() -> Result<(), ConfigError> {
    // Defaults -> file -> environment, then validated.
    let config = Config::load(None)?;
    println!("binding {}", config.addr()?);
    Ok(())
}
```

## Output format

Each persisted chunk is one JSON object (pretty-printed here for readability):

```json
{
  "id": "docs/intro.txt#0",
  "content": "Retrieval augmented generation grounds a language model in external knowledge. The ingestion pipeline loads raw",
  "metadata": {
    "source": "docs/intro.txt",
    "chunk_index": 0,
    "total_chunks": 3,
    "start_char": 0,
    "end_char": 111,
    "extra": {
      "filename": "intro.txt",
      "loader": "text"
    }
  }
}
```

`start_char` / `end_char` are character (not byte) offsets into the source document.

## Library usage

Add the crate to another workspace member or use it directly:

```rust
use penr_oz_ai_rag_service::{
    FixedSizeChunker, IngestionPipeline, InMemoryStorage,
};

fn main() -> penr_oz_ai_rag_service::Result<()> {
    let chunker = FixedSizeChunker::new(800, 100)?; // chunk_size, overlap (in chars)

    let mut pipeline = IngestionPipeline::builder(InMemoryStorage::new())
        .chunker(chunker)
        .build();

    let report = pipeline.ingest_path("docs")?;
    println!("created {} chunks", report.chunks_created);

    for chunk in pipeline.into_store().chunks() {
        println!("{} ({} chars)", chunk.id, chunk.content.chars().count());
    }
    Ok(())
}
```

To persist instead, swap the store for `JsonlStorage::create("chunks.jsonl")?` and call
`pipeline.flush()?` when done.

## Extending the pipeline

The pipeline is built to grow. To add support for a new format, implement `Loader` and
register it:

```rust
use std::path::Path;
use std::sync::Arc;
use penr_oz_ai_rag_service::{Document, Loader, LoaderRegistry, Result};

struct MarkdownLoader;

impl Loader for MarkdownLoader {
    fn extensions(&self) -> &[&str] {
        &["md", "markdown"]
    }

    fn load(&self, path: &Path) -> Result<Document> {
        // read `path`, strip markup, and return a normalized `Document`
        todo!()
    }
}

let mut loaders = LoaderRegistry::with_defaults();
loaders.register(Arc::new(MarkdownLoader));
// IngestionPipeline::builder(store).loaders(loaders).build()
```

The same pattern applies to chunking (implement `Chunker`) and storage (implement
`ChunkStore`, e.g. to write to a vector database) — each is selected on the
`IngestionPipeline` builder without changing the other stages.

## Embeddings

Turning chunks into vectors is decoupled from ingestion behind the `EmbeddingProvider`
trait, so the embedding backend (a hosted API, a local model, …) can be swapped without
touching the rest of the service. The trait embeds a **batch** at a time, is object-safe
(usable as `Box<dyn EmbeddingProvider>`), and surfaces a dedicated `EmbeddingError`:

```rust
use penr_oz_ai_rag_service::{EmbeddingError, EmbeddingProvider, MockEmbeddingProvider};

#[tokio::main]
async fn main() -> Result<(), EmbeddingError> {
    // `MockEmbeddingProvider` produces deterministic vectors with no network — handy in
    // tests and examples. Swap in a real provider behind the same trait.
    let provider = MockEmbeddingProvider::new();
    let vectors = provider.embed(&["hello", "world"]).await?;

    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0].len(), provider.dimensions());
    Ok(())
}
```

Provider-specific code (HTTP, auth, request shaping) lives inside each implementation, so
adding a real provider is a matter of implementing `EmbeddingProvider` and returning
`EmbeddingError` for failures.

## Vector search

Embedded chunks are indexed and retrieved through the `VectorStore` trait, the storage
abstraction for the retrieval half of RAG: insert `EmbeddedChunk`s (a chunk plus its
vector), then ask for the top-k most similar chunks to a query vector. Each
`SearchResult` carries the matching chunk's text and metadata alongside a similarity
score, so a retriever has everything it needs without a second lookup. Like
`EmbeddingProvider`, the trait is async and object-safe (usable as
`Arc<dyn VectorStore>`) and surfaces a dedicated `VectorStoreError`.

`InMemoryVectorStore` is the built-in backend for development and tests: it keeps vectors
in memory and answers searches with an exact cosine-similarity scan. A production
deployment swaps in an approximate-nearest-neighbor service (Qdrant, Pinecone, pgvector,
…) behind the same trait.

```rust
use penr_oz_ai_rag_service::{
    EmbeddingProvider, EmbeddedChunk, InMemoryVectorStore, MockEmbeddingProvider, VectorStore,
    VectorStoreError,
};

#[tokio::main]
async fn main() -> Result<(), VectorStoreError> {
    let provider = MockEmbeddingProvider::new();
    let store = InMemoryVectorStore::new();

    // `chunks: Vec<Chunk>` comes from the ingestion pipeline. Embed each chunk's content
    // and index it alongside the chunk.
    let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
    let vectors = provider.embed(&texts).await.expect("embed");
    let items: Vec<EmbeddedChunk> = chunks
        .into_iter()
        .zip(vectors)
        .map(|(chunk, vector)| EmbeddedChunk::new(chunk, vector))
        .collect();
    store.insert(&items).await?;

    // Embed the query the same way, then retrieve the 5 closest chunks.
    let query = provider.embed(&["how does retrieval work?"]).await.expect("embed");
    for hit in store.search(&query[0], 5).await? {
        println!("{:.3}  {}", hit.score, hit.content());
    }
    Ok(())
}
```

The first inserted vector fixes the store's dimensionality; later vectors and query
vectors must match it, or the store returns `VectorStoreError::DimensionMismatch`.

## Retrieval

`Retriever` composes the embedding and vector-store layers into the read half of RAG —
the engine behind a `POST /retrieve` endpoint. Given a query it **validates** it, embeds
it with an `EmbeddingProvider`, and searches a `VectorStore`, returning the top matching
chunks with their similarity scores. Empty (or whitespace-only) and oversized queries are
rejected with a dedicated `RetrievalError` *before* any embedding or search work happens,
so bad input never reaches the backend.

```rust
use penr_oz_ai_rag_service::{
    Chunk, InMemoryVectorStore, MockEmbeddingProvider, RetrievalError, Retriever,
};

#[tokio::main]
async fn main() -> Result<(), RetrievalError> {
    let retriever = Retriever::new(MockEmbeddingProvider::new(), InMemoryVectorStore::new());

    // `chunks: Vec<Chunk>` comes from the ingestion pipeline. `index` embeds each chunk's
    // content and moves it into the store, so it becomes retrievable without a clone.
    let chunks: Vec<Chunk> = Vec::new();
    retriever.index(chunks).await?;

    // Retrieve the 5 chunks most relevant to a query, each with its similarity score.
    for hit in retriever.retrieve("how does retrieval work?", 5).await? {
        println!("{:.3}  {}", hit.score, hit.content());
    }
    Ok(())
}
```

The `RetrievalRequest` / `RetrievalResponse` pair is the JSON wire shape of the endpoint:
deserialize the `POST` body into a `RetrievalRequest` (`top_k` defaults to `5` when
omitted), call `Retriever::handle`, and serialize the `RetrievalResponse` back. The
library stays web-framework-free — retrieval is a runtime-agnostic layer — while the
`penr-oz-rag` binary hosts the endpoint with axum via the `serve` command. Validation
errors map to `400 Bad Request` and backend failures to `5xx`.

### The `POST /retrieve` endpoint

`serve` ingests a file or directory, indexes every chunk, and answers retrieval queries:

```bash
# Start the service over a corpus
penr-oz-rag serve ./docs --addr 127.0.0.1:8080

# Top matching chunks with similarity scores
curl -s localhost:8080/retrieve \
  -H 'content-type: application/json' \
  -d '{"query": "how does chunking work?", "top_k": 3}'

# top_k omitted -> defaults to 5
curl -s localhost:8080/retrieve -H 'content-type: application/json' \
  -d '{"query": "cosine similarity"}'

# Validation: empty or oversized queries -> 400 with an error message
curl -si localhost:8080/retrieve -H 'content-type: application/json' \
  -d '{"query": "   "}'
```

A `200` response carries the ranked results, most similar first:

```json
{
  "results": [
    {
      "chunk": {
        "id": "docs/intro.txt#0",
        "content": "Retrieval augmented generation grounds a language model...",
        "metadata": { "source": "docs/intro.txt", "chunk_index": 0, "...": "..." }
      },
      "score": 0.83
    }
  ]
}
```

> **Note:** the service currently embeds with the deterministic, in-process
> `MockEmbeddingProvider` — the only provider in the crate so far — so similarity is
> hash-based rather than semantic. The endpoint mechanics (validation, ranking, `top_k`,
> scores, error mapping) are all real; retrieval becomes semantic the moment a real
> `EmbeddingProvider` implementation lands, with no changes to the handler.

## Answer generation

`AnswerGenerator` composes a `Retriever` with an `LlmProvider` into the generative half
of RAG — the engine behind a `POST /answer` endpoint. Given a question it:

1. **retrieves** the top-k chunks (validation included, exactly as `/retrieve` does),
2. **gates by confidence** — chunks scoring below a minimum similarity are dropped, so
   unrelated chunks never reach the model when retrieval confidence is low,
3. **builds a grounded prompt** (`build_prompt`) from the surviving passages, numbered
   and attributed, instructing the model to answer only from them,
4. **calls the configured `LlmProvider`**, and
5. returns the answer together with **source references** (`SourceRef`: chunk id, source
   document, chunk index, similarity score) for the exact chunks the prompt used.

If *no* chunk clears the gate, the model is not called at all: the response carries the
sentinel `NO_CONTEXT_ANSWER` and an empty source list, so the generator refuses to
answer from context it does not trust rather than prompting the model with noise.

The `LlmProvider` trait mirrors `EmbeddingProvider`: async, object-safe
(`Box<dyn LlmProvider>`), with a dedicated `LlmError` — so a hosted model is a drop-in
implementation. `MockLlmProvider` is the built-in, deterministic stand-in: by default it
echoes the prompt back (which lets tests assert on exactly what the model was shown),
`with_reply` fixes a canned answer, and `failing` exercises error paths.

```rust
use penr_oz_ai_rag_service::{
    AnswerGenerator, Chunk, GenerationError, InMemoryVectorStore, MockEmbeddingProvider,
    MockLlmProvider, Retriever,
};

#[tokio::main]
async fn main() -> Result<(), GenerationError> {
    let retriever = Retriever::new(MockEmbeddingProvider::new(), InMemoryVectorStore::new());

    // `chunks: Vec<Chunk>` comes from the ingestion pipeline.
    let chunks: Vec<Chunk> = Vec::new();
    retriever.index(chunks).await?;

    // `with_min_score` sets the default confidence gate; tune it per embedding model.
    let generator = AnswerGenerator::new(retriever, MockLlmProvider::new());

    // Retrieve up to 5 chunks, keep those scoring >= 0.25, prompt the model, answer.
    let response = generator.answer("how does retrieval work?", 5, 0.25).await?;
    println!("{}", response.answer);
    for source in &response.sources {
        println!("  [{}] {} (score {:.3})", source.id, source.source, source.score);
    }
    Ok(())
}
```

The `AnswerRequest` / `AnswerResponse` pair is the JSON wire shape of the endpoint:
deserialize the `POST` body into an `AnswerRequest` (`top_k` defaults to `5` and
`min_score` to the generator's threshold when omitted), call `AnswerGenerator::handle`,
and serialize the `AnswerResponse` back. Validation errors map to `400 Bad Request`,
embedding/LLM backend failures to `502`, and vector-store failures to `500`.

### The `POST /answer` endpoint

`serve` hosts answering next to retrieval, over the same indexed corpus:

```bash
# Start the service over a corpus (optionally tune the confidence gate)
penr-oz-rag serve ./docs --addr 127.0.0.1:8080 --min-score 0.25

# A grounded answer with source references
curl -s localhost:8080/answer \
  -H 'content-type: application/json' \
  -d '{"query": "how does chunking work?", "top_k": 3}'

# Per-request confidence gate: exclude weakly related chunks from the prompt
curl -s localhost:8080/answer -H 'content-type: application/json' \
  -d '{"query": "how does chunking work?", "min_score": 0.5}'

# Validation: empty or oversized queries -> 400 with an error message
curl -si localhost:8080/answer -H 'content-type: application/json' \
  -d '{"query": "   "}'
```

A `200` response carries the answer and the chunks it was grounded in:

```json
{
  "answer": "Chunking splits each document into fixed-size, overlapping windows...",
  "sources": [
    {
      "id": "docs/intro.txt#0",
      "source": "docs/intro.txt",
      "chunk_index": 0,
      "score": 0.83
    }
  ]
}
```

When no retrieved chunk clears the minimum score, the response is still `200`, with
`"sources": []` and the fixed `NO_CONTEXT_ANSWER` text as the answer — the model is
never shown low-confidence context.

> **Note:** the service currently generates with the deterministic, in-process
> `MockLlmProvider` — the only provider in the crate so far — which echoes the grounded
> prompt rather than writing prose. The endpoint mechanics (retrieval, confidence
> gating, prompt building, source attribution, error mapping) are all real; answers
> become fluent the moment a real `LlmProvider` implementation lands, with no changes to
> the handler.

## Project layout

```
src/
├── lib.rs            crate root and re-exports
├── main.rs           `penr-oz-rag` CLI (ingest + serve, POST /retrieve + /answer handlers)
├── config.rs         Config, ConfigError, provider/store/logging kinds
├── error.rs          RagError / Result
├── document.rs       Document, Chunk, ChunkMetadata
├── loader/           Loader trait, LoaderRegistry, TextLoader
├── chunker/          Chunker trait, FixedSizeChunker
├── storage/          ChunkStore trait, InMemoryStorage, JsonlStorage
├── embedding/        EmbeddingProvider trait, EmbeddingError, MockEmbeddingProvider
├── vector/           VectorStore trait, VectorStoreError, InMemoryVectorStore
├── retrieval.rs      Retriever, RetrievalRequest/Response, RetrievalError
├── llm/              LlmProvider trait, LlmError, MockLlmProvider
├── generation.rs     AnswerGenerator, AnswerRequest/Response, SourceRef, build_prompt
└── pipeline.rs       IngestionPipeline + builder
tests/
├── ingestion.rs      end-to-end ingestion tests
├── embedding.rs      embedding abstraction tests
├── vector_search.rs  end-to-end embed-index-retrieve tests
├── retrieval.rs      end-to-end retriever tests (validate, embed, search)
├── generation.rs     end-to-end answer-generation tests (gate, prompt, attribute)
├── serve.rs          end-to-end HTTP tests against the served /retrieve + /answer endpoints
└── config.rs         config file layer plus the binary booting on what it resolved
```

## License

Licensed under the [MIT License](LICENSE).

[`Loader`]: src/loader/mod.rs
