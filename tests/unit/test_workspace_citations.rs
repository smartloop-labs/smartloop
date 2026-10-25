//! Files from the project's folder are cited like documents: one citation
//! per file, with its path to open and the text it gave the model.

use smartloop::agents::mcp::explorer::{ConversationTurn, DocumentGroups, Explorer, ExplorerSearchService};
use smartloop::parsers::document_chunk::DocumentChunk;

/// A service whose folder search returns `chunks`.
struct FolderSearch(Vec<DocumentChunk>);

impl ExplorerSearchService for FolderSearch {
    fn get_document_groups(
        &self,
        _query: Option<&str>,
        _skill_instruction: Option<&str>,
        _conversation_history: Option<&[ConversationTurn]>,
    ) -> anyhow::Result<Option<DocumentGroups>> {
        Ok(None)
    }
    fn search_feedback(&self, _query: &str, _n_results: usize) -> anyhow::Result<Vec<DocumentChunk>> {
        Ok(Vec::new())
    }
    fn search_workspace(
        &self,
        _query: &str,
        _work_dir: &str,
        _n_results: usize,
        _conversation_history: Option<&[ConversationTurn]>,
    ) -> anyhow::Result<Vec<DocumentChunk>> {
        Ok(self.0.clone())
    }
}

fn chunk(file: &str, text: &str, distance: f64) -> DocumentChunk {
    DocumentChunk { document_id: file.to_string(), text: text.to_string(), distance: Some(distance), ..Default::default() }
}

#[test]
fn test_each_file_is_cited_once_with_its_path_and_extract() {
    let service = FolderSearch(vec![
        chunk("Fund supplement.pdf", "Class A shares carry voting rights.", 0.2),
        chunk("Fund supplement.pdf", "Class B shares do not.", 0.4),
        chunk("notes/terms.md", "Terms apply.", 0.3),
    ]);

    let (text, citations) = Explorer::search_workspace("share classes", &service, "/Users/me/lexic", None);

    assert!(text.contains("--- Fund supplement.pdf ---\nClass A shares carry voting rights."));
    let names: Vec<&str> = citations.iter().map(|c| c["document_name"].as_str().unwrap()).collect();
    assert_eq!(names, ["Fund supplement.pdf", "notes/terms.md"]);
    assert_eq!(citations[0]["path"], "/Users/me/lexic/Fund supplement.pdf");
    assert_eq!(citations[0]["_extract"], "Class A shares carry voting rights.");
    assert_eq!(citations[1]["path"], "/Users/me/lexic/notes/terms.md");
}

#[test]
fn test_no_hits_no_citations() {
    let (text, citations) = Explorer::search_workspace("anything", &FolderSearch(Vec::new()), "/tmp", None);

    assert!(text.is_empty());
    assert!(citations.is_empty());
}
