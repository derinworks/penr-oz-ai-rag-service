//! Integration test for the basic RAG flow, end to end through the library's public
//! API: start with a small document on disk, ingest it through the real
//! [`IngestionPipeline`] (loader → chunker → store), index the stored chunks, query
//! for relevant information, verify the correct chunk comes back, and generate an
//! answer grounded in it.
//!
//! Only the built-in mock providers are used — [`MockEmbeddingProvider`] embeds
//! deterministically (identical text yields an identical vector, so querying a chunk's
//! exact words is a near-perfect cosine match) and [`MockLlmProvider`] echoes the
//! prompt it was shown — so the whole flow runs offline, without paid APIs.

use std::fs;

use penr_oz_ai_rag_service::{
    AnswerGenerator, FixedSizeChunker, InMemoryStorage, InMemoryVectorStore, IngestionPipeline,
    MockEmbeddingProvider, MockLlmProvider, Retriever,
};
use tempfile::tempdir;

/// Ingest `input` the way the `serve` command does — into memory with `chunker` — and
/// return the chunks ready for indexing.
fn ingest(
    input: &std::path::Path,
    chunker: FixedSizeChunker,
) -> Vec<penr_oz_ai_rag_service::Chunk> {
    let mut pipeline = IngestionPipeline::builder(InMemoryStorage::new())
        .chunker(chunker)
        .build();
    let report = pipeline.ingest_path(input).expect("ingestion succeeds");
    assert!(report.files_ingested >= 1);
    assert!(report.chunks_created >= 1);

    let chunks = pipeline.into_store().into_chunks();
    assert_eq!(chunks.len(), report.chunks_created);
    chunks
}

/// Isolate the numbered context passages of a prompt built by `build_prompt` — the
/// span between the `Context:` header and the trailing `Question:` line.
///
/// Grounding assertions run against this section rather than the whole prompt so they
/// cannot be satisfied by the question that `build_prompt` always appends.
fn context_section(prompt: &str) -> &str {
    const HEADER: &str = "Context:\n";
    let start = prompt.find(HEADER).expect("prompt has a context section") + HEADER.len();
    let end = start
        + prompt[start..]
            .find("Question:")
            .expect("prompt has a question line");
    &prompt[start..end]
}

#[tokio::test]
async fn ingested_documents_flow_through_retrieval_to_a_grounded_answer() {
    // Two small documents, each shorter than the 64-character chunk size, so each
    // becomes exactly one chunk whose content is the whole file — which makes
    // retrieval assertable: querying one file's exact text must rank its chunk first.
    let dir = tempdir().unwrap();
    let docs = dir.path().join("docs");
    fs::create_dir(&docs).unwrap();
    fs::write(
        docs.join("rag.txt"),
        "retrieval augmented generation grounds answers",
    )
    .unwrap();
    fs::write(
        docs.join("search.txt"),
        "cosine similarity ranks vector search hits",
    )
    .unwrap();

    // Ingest: file → Document → Chunk[] → store, through the real pipeline.
    let chunks = ingest(&docs, FixedSizeChunker::new(64, 8).unwrap());
    assert_eq!(chunks.len(), 2);

    // Index every stored chunk for retrieval, as `serve` does at startup.
    let retriever = Retriever::new(MockEmbeddingProvider::new(), InMemoryVectorStore::new());
    let indexed = retriever.index(chunks).await.expect("indexing succeeds");
    assert_eq!(indexed, 2);

    // Query for relevant information: the exact text of one document must retrieve
    // that document's chunk, first and with a (near-)perfect score.
    let query = "retrieval augmented generation grounds answers";
    let results = retriever
        .retrieve(query, 2)
        .await
        .expect("retrieval succeeds");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].chunk.content, query);
    assert!((results[0].score - 1.0).abs() < 1e-6);
    assert!(results[1].score < results[0].score);

    // Provenance attached at ingestion survived the whole flow into the result.
    let metadata = &results[0].chunk.metadata;
    assert!(metadata.source.ends_with("rag.txt"));
    assert_eq!(
        metadata.extra.get("filename").map(String::as_str),
        Some("rag.txt")
    );
    assert_eq!(
        metadata.extra.get("loader").map(String::as_str),
        Some("text")
    );
    assert_eq!((metadata.chunk_index, metadata.total_chunks), (0, 1));
    let retrieved_id = results[0].chunk.id.clone();

    // Generate an answer with the mock LLM. The echoing mock returns the prompt it was
    // shown, proving the ingested document's text and the question both reached the
    // model; the sources attribute the answer to the retrieved chunk.
    let generator = AnswerGenerator::new(retriever, MockLlmProvider::new());
    let response = generator
        .answer(query, 1, 0.0)
        .await
        .expect("generation succeeds");

    // Assert the passage reached the model's *context*, not merely that its words
    // appear somewhere in the prompt: the query here is the document's exact text, so
    // a bare `contains` would also be satisfied by the trailing question line and would
    // stay green even if `build_prompt` stopped including retrieved passages entirely.
    let context = context_section(&response.answer);
    assert!(
        context.contains("retrieval augmented generation grounds answers"),
        "retrieved passage missing from the prompt's context section: {context:?}"
    );
    assert!(
        context.contains(&format!("[1] {retrieved_id}")),
        "retrieved chunk not cited as context passage 1: {context:?}"
    );
    assert!(response.answer.contains(&format!("Question: {query}")));
    assert_eq!(response.sources.len(), 1);
    assert_eq!(response.sources[0].id, retrieved_id);
    assert!(response.sources[0].source.ends_with("rag.txt"));
    assert!((response.sources[0].score - 1.0).abs() < 1e-6);
}

#[tokio::test]
async fn retrieval_pinpoints_the_correct_chunk_of_a_multi_chunk_document() {
    // One document long enough to split into several overlapping chunks, so the test
    // proves retrieval distinguishes *chunks*, not just documents.
    let dir = tempdir().unwrap();
    let path = dir.path().join("guide.txt");
    fs::write(
        &path,
        "Loaders normalize raw files into documents with provenance metadata. \
         Chunkers split documents into overlapping character windows. \
         Vector stores rank indexed chunks by cosine similarity for retrieval.",
    )
    .unwrap();

    let chunks = ingest(&path, FixedSizeChunker::new(48, 8).unwrap());
    assert!(
        chunks.len() >= 2,
        "document must split into multiple chunks"
    );
    let total = chunks.len();

    // Remember a middle chunk to query for; every chunk window has distinct text, so
    // the deterministic mock embedding is unique per chunk.
    let target = &chunks[total / 2];
    let query = target.content.clone();
    let expected_id = target.id.clone();
    let expected_index = target.metadata.chunk_index;

    let retriever = Retriever::new(MockEmbeddingProvider::new(), InMemoryVectorStore::new());
    let indexed = retriever.index(chunks).await.expect("indexing succeeds");
    assert_eq!(indexed, total);

    // Querying the target chunk's exact text retrieves that chunk — and only that
    // chunk — with a (near-)perfect score; its siblings score strictly lower.
    let results = retriever
        .retrieve(&query, total)
        .await
        .expect("retrieval succeeds");
    assert_eq!(results.len(), total);
    assert_eq!(results[0].chunk.id, expected_id);
    assert_eq!(results[0].chunk.metadata.chunk_index, expected_index);
    assert_eq!(results[0].chunk.metadata.total_chunks, total);
    assert!((results[0].score - 1.0).abs() < 1e-6);
    for other in &results[1..] {
        assert!(other.score < results[0].score);
    }
}
