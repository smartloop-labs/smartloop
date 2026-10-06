//! Port of `tests/unit/test_citation_verification.py`.
//!
//! Citations credit only the sources the answer actually used.
//!
//! Retrieval returning a document does not mean the answer used it: nearest
//! neighbour always returns *something*, so a project holding a single
//! document surfaces it for every question however unrelated.  That is how a
//! `good-morning` workout skill ended up citing an invoice — the invoice was
//! the only document in the project, so it was necessarily the closest match.
//!
//! Matching is on document *content*.  An earlier revalidation matched on the
//! filename and silently dropped real sources whose names say nothing about
//! their contents ("sp46.pdf" for an index fund); see the comment at the
//! emission site.
//!
//! The evidence is a shared run of consecutive words, or a shared precise
//! figure.  Single shared words were tried first and are far too weak:
//! measured on the transcripts below, a workout answer and an unrelated fund
//! prospectus share 7-12 ordinary English words, which is exactly how five
//! fund PDFs came to be credited to a `good-morning` routine.
//!
//! Every fixture below is invented.  The behaviour under test came from real
//! transcripts, but a fixture is read far more often than the transcript it
//! came from, so no real person, employer, document or product name appears
//! here.
//!
//! Where the Python tests monkeypatch `smartloop.db.embeddings.encode_texts`,
//! these install the equivalent override with
//! `embeddings::set_encode_texts_hook` (serialised under the `embeddings`
//! key, since the override is process-wide).  The lexical-section tests run
//! with an override that reports "no embedder", which is what the real
//! `encode_texts` returns in the Python test environment (no GGUF available).

use std::collections::HashSet;
use std::sync::Arc;

use ndarray::{Array1, Array2};
use parking_lot::Mutex;
use serde_json::{Map, Value};
use serial_test::serial;
use smartloop::api::routers::chat::{
    citation_sources, citation_used, figures, phrases, semantic_distances, verified_citations, windows,
    CITATION_CUTOFF_MARGIN, CITATION_MAX_DISTANCE, MAX_WINDOWS, WINDOW_CHARS,
};
use smartloop::db::embeddings::{self, blob_to_value, embedding_to_blob};

type Cit = Map<String, Value>;

const INVOICE_DOC: &str = "invoice-contoso-jordan-rivera.pdf";
const INVOICE: &str = concat!(
    "=== Source: invoice-contoso-jordan-rivera.pdf ===\n",
    "Invoice from Contoso Ltd. to Jordan Rivera. Amount $1,000.00 for ",
    "contract work services. Invoice date August 2026."
);

fn cit_with(extract: Option<&str>, doc_id: &str, name: &str) -> Cit {
    let mut c = Map::new();
    c.insert("document_id".into(), Value::String(doc_id.into()));
    c.insert("document_name".into(), Value::String(name.into()));
    if let Some(extract) = extract {
        c.insert("_extract".into(), Value::String(extract.into()));
    }
    c
}

/// `_cit()` with its defaults.
fn cit() -> Cit {
    cit_with(Some(INVOICE), "d1", INVOICE_DOC)
}

/// `_cit(extract=...)`.
fn cit_extract(extract: Option<&str>) -> Cit {
    cit_with(extract, "d1", INVOICE_DOC)
}

/// `_verified_citations(citations, response)` with its defaults
/// (`base_dir=None`, `embed=True`).
fn verified(citations: &[Cit], response: &str) -> Vec<Cit> {
    verified_citations(citations, response, None, true)
}

fn names(out: &[Cit]) -> Vec<String> {
    out.iter()
        .map(|c| c["document_name"].as_str().unwrap().to_string())
        .collect()
}

fn set(items: &[&str]) -> HashSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

fn no_private_keys(c: &Cit) -> bool {
    c.keys().all(|k| !k.starts_with('_'))
}

/// Clears the `encode_texts` override however the test ends (monkeypatch's
/// undo).
#[must_use]
struct EncoderGuard;

impl Drop for EncoderGuard {
    fn drop(&mut self) {
        embeddings::set_encode_texts_hook(None);
    }
}

/// `encode_texts` returning `None`: no embedding model on this install.
fn no_embedder() -> EncoderGuard {
    embeddings::set_encode_texts_hook(Some(Arc::new(|_texts, _base_dir, _timeout| None)));
    EncoderGuard
}

// --------------------------------------------------------------------------
// the reported bug
// --------------------------------------------------------------------------

#[test]
#[serial(embeddings)]
fn test_unrelated_answer_does_not_cite_the_document() {
    let _enc = no_embedder();
    let answer = "Good morning! Here are 10 minute workout routines focusing on upper \
                  body. Push-ups (3 sets of 8-12 reps), dumbbell rows, shoulder presses.";
    assert!(verified(&[cit()], answer).is_empty());
}

#[test]
#[serial(embeddings)]
fn test_answer_that_uses_the_document_still_cites_it() {
    let _enc = no_embedder();
    let answer = "The invoice is for $1,000.00 paid by Contoso Ltd. to \
                  Jordan Rivera for contract work services.";
    let out = verified(&[cit()], answer);
    assert_eq!(names(&out), vec![INVOICE_DOC]);
}

// --------------------------------------------------------------------------
// what still counts as use
// --------------------------------------------------------------------------

/// Reworded around the source's own phrase — still a real use.
#[test]
fn test_paraphrase_keeping_the_wording_is_kept() {
    assert!(citation_used(&cit(), "Rivera was engaged for contract work."));
}

/// '$1,000.00' in the source must match '1000' in the answer.
///
/// A translated or fully reworded answer keeps none of the source's phrasing
/// but still carries its figures, so a shared figure alone qualifies.
#[test]
fn test_numeric_match_survives_formatting() {
    assert!(citation_used(&cit(), "The total came to 1000 dollars."));
}

/// "3 sets of 10 reps" must not match every document containing a 3.
#[test]
fn test_small_numbers_are_not_evidence() {
    let c = cit_extract(Some("The fund held 3 positions across 10 sectors last year."));
    assert!(!citation_used(&c, "Do 3 sets of 10 reps with a 5 second hold."));
}

/// The measured failure: unrelated texts share ordinary vocabulary.
#[test]
fn test_incidental_common_words_are_not_evidence() {
    let c = cit_extract(Some(
        "The Fund holds securities that may decline in value. Shares are \
         available to investors since inception; performance is not otherwise \
         guaranteed. Three different share classes are offered.",
    ));
    let answer = "Since it is Wednesday, here are three different options. Hold each \
                  plank for 45 seconds and perform arm raises. Focus on full range.";
    assert!(!citation_used(&c, answer));
}

/// Absence of evidence is not evidence of misuse.
#[test]
fn test_citation_without_an_extract_is_always_kept() {
    assert!(citation_used(&cit_extract(None), "anything at all"));
}

#[test]
fn test_empty_response_keeps_the_citation() {
    assert!(citation_used(&cit(), ""));
}

/// Nothing to compare against is not proof of misuse.
#[test]
fn test_extract_too_short_to_judge_keeps_the_citation() {
    assert!(citation_used(&cit_extract(Some("Rye flour")), "unrelated"));
}

/// The regression that got the old filename check reverted, inverted.
///
/// Echoing the filename is not evidence the *content* was used.
#[test]
#[serial(embeddings)]
fn test_filename_alone_does_not_justify_a_citation() {
    let _enc = no_embedder();
    let answer = format!("I could not read {INVOICE_DOC}.");
    assert!(verified(&[cit_extract(Some("Northwind Growth Fund quarterly holdings"))], &answer).is_empty());
}

/// 'sp46.pdf' says nothing about its contents; the content still matches.
#[test]
fn test_opaque_filename_is_no_obstacle() {
    let c = cit_with(Some("Northwind Growth Fund quarterly holdings report"), "d1", "sp46.pdf");
    assert!(citation_used(&c, "This is the Northwind Growth Fund."));
}

// --------------------------------------------------------------------------
// hygiene
// --------------------------------------------------------------------------

#[test]
#[serial(embeddings)]
fn test_private_extract_never_reaches_the_client() {
    let _enc = no_embedder();
    let out = verified(&[cit()], "Jordan Rivera was paid $1,000.00");
    assert!(!out.is_empty() && no_private_keys(&out[0]));
}

#[test]
#[serial(embeddings)]
fn test_duplicates_are_collapsed() {
    let _enc = no_embedder();
    let answer = "Jordan Rivera was paid $1,000.00";
    let out = verified(&[cit(), cit()], answer);
    assert_eq!(out.len(), 1);
}

#[test]
#[serial(embeddings)]
fn test_documents_are_judged_independently() {
    let _enc = no_embedder();
    let used = cit_with(Some("Quarterly revenue reached 4.25 million euros"), "a", "a.pdf");
    let unused = cit_with(Some("Recipe for sourdough bread with rye flour"), "b", "b.pdf");
    let out = verified(&[used, unused], "Revenue reached 4.25 million euros.");
    let ids: Vec<&str> = out.iter().map(|c| c["document_id"].as_str().unwrap()).collect();
    assert_eq!(ids, vec!["a"]);
}

#[test]
#[serial(embeddings)]
fn test_no_citations_in_no_citations_out() {
    let _enc = no_embedder();
    assert!(verified(&[], "anything").is_empty());
    // Python also accepts `None`; the slice API has only the empty case.
    let none: Option<Vec<Cit>> = None;
    assert!(verified(none.as_deref().unwrap_or_default(), "anything").is_empty());
}

#[test]
#[serial(embeddings)]
fn test_dedup_falls_back_to_name_when_id_missing() {
    let _enc = no_embedder();
    let mut c = Map::new();
    c.insert("document_name".into(), Value::String("x.pdf".into()));
    c.insert("_extract".into(), Value::String("widget assembly torque specification is 12Nm".into()));
    let out = verified(&[c.clone(), c.clone()], "The widget assembly torque specification is 12Nm.");
    assert_eq!(out.len(), 1);
}

// --------------------------------------------------------------------------
// the matching primitives
// --------------------------------------------------------------------------

#[test]
fn test_phrases_are_runs_of_three_words() {
    assert_eq!(phrases("alpha beta gamma delta"), set(&["alpha beta gamma", "beta gamma delta"]));
}

#[test]
fn test_phrases_ignore_case_and_punctuation() {
    assert!(phrases("The total, came to: 1000").contains("total came to"));
}

#[test]
fn test_text_shorter_than_a_phrase_yields_nothing() {
    assert!(phrases("two words").is_empty());
}

#[test]
fn test_figures_normalise_formatting_currency() {
    assert_eq!(figures("$1,000.00"), set(&["1000"]));
}

#[test]
fn test_figures_normalise_formatting_plain() {
    assert_eq!(figures("1000"), set(&["1000"]));
}

#[test]
fn test_figures_normalise_formatting_percentage() {
    assert_eq!(figures("0.36%"), set(&["0.36"]));
}

/// Too few digits to be evidence.
#[test]
fn test_figures_normalise_formatting_small_numbers() {
    assert_eq!(figures("3 sets of 10 reps"), set(&[]));
}

#[test]
fn test_figures_normalise_formatting_page_numbers() {
    assert_eq!(figures("page 5 of 12"), set(&[]));
}

/// A 400-digit run overflows float() to inf; int(inf) used to raise
/// OverflowError inside the streaming generator and abort the response.
#[test]
fn test_figures_survive_a_number_too_long_for_a_float() {
    let huge = "9".repeat(400);
    assert_eq!(figures(&huge), set(&[&huge]));
    assert!(citation_used(
        &cit_extract(Some(&format!("total {huge}"))),
        &format!("The total is {huge}")
    ));
}

/// The Python test patches `_verify_citations` to raise; here the encoder it
/// calls raises (panics) instead, which is the same failure of the fallible
/// core.
#[test]
#[serial(embeddings)]
fn test_verification_failure_keeps_sources_instead_of_raising() {
    embeddings::set_encode_texts_hook(Some(Arc::new(|_texts, _base_dir, _timeout| panic!("boom"))));
    let _enc = EncoderGuard;
    let out = verified(&[cit(), cit()], "any answer");
    assert_eq!(out.len(), 1);
    assert!(!out[0].contains_key("_extract"));
}

// --------------------------------------------------------------------------
// grammar is not attribution
// --------------------------------------------------------------------------
//
// A citation exists so a reader can trace a statement back to the text it
// came from, so the check is a text comparison and nothing else — no model
// call, nothing for the user to wait through once the answer is on screen.
//
// That puts the whole weight on what counts as a shared run.  Every English
// text contains "here are the" and "in the last"; a run built only out of
// grammar is a coincidence of the language, not a trace of one text in
// another, and each such run used to be enough to credit a document the
// answer had never opened.

#[test]
fn test_a_run_of_pure_grammar_is_not_evidence() {
    assert!(phrases("here are the").is_empty());
    assert!(phrases("as well as").is_empty());
    assert!(phrases("one of the").is_empty());
}

/// The shape of an incidental collision: grammar around a single word.
#[test]
fn test_a_run_with_one_content_word_is_not_evidence() {
    assert!(phrases("here are the").is_empty());
    assert!(phrases("in the last").is_empty());
}

#[test]
fn test_a_run_with_two_content_words_is_evidence() {
    assert!(phrases("paid for contract work").contains("for contract work"));
    assert!(phrases("The total, came to: 1000").contains("total came to"));
}

/// Two texts about nothing in common, sharing only sentence scaffolding.
#[test]
fn test_grammar_only_overlap_no_longer_credits_a_document() {
    let c = cit_extract(Some(
        "Here are the terms. As well as the above, one of the parties shall \
         in the last instance be liable.",
    ));
    let answer = "Here are the steps. As well as the setup, one of the options is to \
                  in the last resort restart.";
    assert!(!citation_used(&c, answer));
}

/// The filter must not cost a real trace.
#[test]
#[serial(embeddings)]
fn test_shared_phrasing_still_credits_the_document() {
    let _enc = no_embedder();
    let answer = "Jordan Rivera was invoiced $1,000.00 for contract work services.";
    let out = verified(&[cit()], answer);
    assert_eq!(names(&out), vec![INVOICE_DOC]);
}

// --------------------------------------------------------------------------
// the reported cases
// --------------------------------------------------------------------------

// A stand-in for the indexed personal document in the reported cases.  Real
// names and employers are deliberately absent — a fixture is read far more
// often than the transcript it came from.
const RESUME: &str = concat!(
    "Principal Widget Architect at Acme Corp. Designed and led ",
    "production-grade widget assembly lines for industrial customers. ",
    "Tech Lead and Co-Founder, Example Industries."
);

fn resume_cit() -> Cit {
    cit_with(Some(RESUME), "r1", "candidate-cv.pdf")
}

/// A camera answer, with a CV as the only indexed document.
///
/// The text comparison always got this one right — the answer and the CV
/// have nothing whatever in common.  It was an LLM judge, since removed, that
/// put the CV back: asked whether the answer had drawn on it, the small model
/// said yes.
#[test]
#[serial(embeddings)]
fn test_an_unrelated_document_is_not_a_source_for_a_camera_answer() {
    let _enc = no_embedder();
    let answer = "Based on the provided search results, here are the key features for \
                  the Zenith QX-9 camera: it is equipped with a medium format BSI CMOS \
                  sensor featuring 100 megapixels and 5-axis stabilisation.";
    assert!(verified(&[resume_cit()], answer).is_empty());
}

/// The other half: when the answer really is the document, say so.
///
/// The same CV, on the request whose answer genuinely was built from it — the
/// citation is the only thing telling the reader where those job titles came
/// from.
#[test]
#[serial(embeddings)]
fn test_a_document_the_answer_was_written_from_is_kept() {
    let _enc = no_embedder();
    let answer = "Based on the provided documents, here are the best jobs these days: \
                  Principal Widget Architect, designing and leading production-grade \
                  widget assembly lines for industrial customers.";
    let out = verified(&[resume_cit()], answer);
    assert_eq!(names(&out), vec!["candidate-cv.pdf"]);
}

// --------------------------------------------------------------------------
// the check has to work in the language the answer is written in
// --------------------------------------------------------------------------
//
// The tokenizer used to be `[a-z0-9][a-z0-9.,%/-]*`.  Outside English that did
// not weaken the check, it switched it off: no ASCII letters meant no words,
// no words meant no runs, no runs meant no evidence either way — and "no
// evidence" is deliberately read as "keep the source".  Every citation on
// every Arabic, Chinese, Hebrew, Russian, Hindi or Thai answer was credited
// unconditionally.  Accented Latin was worse than useless: "über" tokenised
// to "ber", "für" to "f" and "r", and those fragments appear in no stopword
// list, so mangled German counted as three content words.

fn src(text: &str) -> Cit {
    cit_with(Some(text), "s1", "s.pdf")
}

/// Not "ber" and not "f" + "r".
#[test]
fn test_accented_words_survive_tokenisation() {
    let p = phrases("Die Rechnung über tausend Euro für Vertragsarbeit");
    assert!(p.contains("rechnung über tausend"));
    assert!(p.contains("euro für vertragsarbeit"));
}

#[test]
fn test_a_script_without_latin_letters_still_yields_runs() {
    assert!(!phrases("الفاتورة من شركة كونتوسو بمبلغ ألف دولار").is_empty());
}

#[test]
fn test_an_unrelated_arabic_source_is_dropped() {
    assert!(!citation_used(
        &src("السيرة الذاتية: مهندس معماري رئيسي في شركة أكمي"),
        "الميزات الرئيسية لكاميرا زينيث: مستشعر بدقة 100 ميجابكسل",
    ));
}

#[test]
fn test_an_arabic_source_the_answer_used_is_kept() {
    assert!(citation_used(
        &src("الفاتورة من شركة كونتوسو بمبلغ ألف دولار مقابل أعمال تعاقدية"),
        "دفعت شركة كونتوسو بمبلغ ألف دولار مقابل أعمال تعاقدية.",
    ));
}

// Japanese, Chinese and Thai put no spaces between words, so splitting on
// whitespace yields one enormous token and no runs — the same hole, reopened.
// Their runs are counted in characters instead.

#[test]
fn test_a_space_less_script_yields_character_runs() {
    assert!(!phrases("发票金额为一千美元的合同工作服务").is_empty());
}

#[test]
fn test_an_unrelated_chinese_source_is_dropped() {
    assert!(!citation_used(
        &src("简历：阿克米公司首席架构师，负责设计生产级系统"),
        "泽尼斯相机的主要特点是一亿像素传感器",
    ));
}

#[test]
fn test_a_chinese_source_the_answer_used_is_kept() {
    assert!(citation_used(
        &src("简历：阿克米公司首席架构师，负责设计生产级系统"),
        "他是阿克米公司首席架构师，负责设计生产级系统。",
    ));
}

#[test]
fn test_a_japanese_source_the_answer_used_is_kept() {
    assert!(citation_used(
        &src("請求書：コントソ社への契約作業に対する千ドルの支払い"),
        "コントソ社への契約作業に対する千ドルが支払われました。",
    ));
}

// ---------------------------------------------------------------------------
// The embedding path.
//
// Word overlap is the wrong test for a citation: an answer is a rewrite of its
// source, not a quotation of it, so a paraphrase, a translation or a summary
// into a table keeps none of the source's phrasing while being entirely about
// it.  Verification therefore embeds both sides and compares them the way
// retrieval compares a chunk to the user's question — `encode_texts` for the
// vectors, `cosine_distances` for the comparison, the retrieval quality floor
// for the cutoff.  The lexical checks above remain as the fallback for
// installs with no embedding model, which is why they are still exercised.
//
// The stub embedder below is a bag of words over a fixed vocabulary.  It is
// not a language model and knows no synonyms, so these tests pin the plumbing
// -- batching, windowing, per-source scoring, thresholding, degradation -- and
// not how well bge-m3 happens to score any particular pair of texts.

const VOCAB: [&str; 6] = ["invoice", "contoso", "rivera", "camera", "sensor", "hasselblad"];

fn bag(text: &str) -> Vec<f32> {
    let lowered = text.to_lowercase();
    VOCAB.iter().map(|term| lowered.matches(term).count() as f32).collect()
}

type Calls = Arc<Mutex<Vec<Vec<String>>>>;

/// Install a deterministic bag-of-words embedder in place of the model.
fn stub_encoder(calls: Option<Calls>) -> EncoderGuard {
    embeddings::set_encode_texts_hook(Some(Arc::new(move |texts: &[String], _base_dir, _timeout| {
        if let Some(calls) = &calls {
            calls.lock().push(texts.to_vec());
        }
        // A text with none of the vocabulary still needs a non-zero vector:
        // the zero vector normalises to itself and is equidistant from
        // everything, which would make the threshold meaningless.
        let rows: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| {
                let r = bag(t);
                if r.iter().any(|v| *v != 0.0) { r } else { vec![1e-6; VOCAB.len()] }
            })
            .collect();
        let flat: Vec<f32> = rows.into_iter().flatten().collect();
        Some(Array2::from_shape_vec((texts.len(), VOCAB.len()), flat).unwrap())
    })));
    EncoderGuard
}

fn distance(source: &str, answer: &str) -> f64 {
    let _enc = stub_encoder(None);
    let mut c = Map::new();
    c.insert("_extract".into(), Value::String(source.into()));
    semantic_distances(&[c], answer, None).unwrap()[0].unwrap()
}

#[test]
#[serial(embeddings)]
fn test_a_source_the_answer_is_about_scores_close() {
    let d = distance(INVOICE, "The Contoso invoice is for Rivera.");
    assert!(d <= CITATION_MAX_DISTANCE);
}

#[test]
#[serial(embeddings)]
fn test_an_unrelated_source_scores_far() {
    let d = distance(INVOICE, "The camera sensor is a Hasselblad.");
    assert!(d > CITATION_MAX_DISTANCE);
}

#[test]
#[serial(embeddings)]
fn test_the_unrelated_source_is_dropped() {
    let _enc = stub_encoder(None);
    assert!(verified(&[cit()], "The camera sensor is a Hasselblad.").is_empty());
}

#[test]
#[serial(embeddings)]
fn test_the_used_source_survives_with_private_keys_stripped() {
    let _enc = stub_encoder(None);
    let out = verified(&[cit()], "The Contoso invoice is for Rivera.");
    assert_eq!(out.len(), 1);
    assert!(no_private_keys(&out[0]));
}

#[test]
#[serial(embeddings)]
fn test_every_source_is_embedded_in_one_batched_call() {
    // Verification runs after the answer has streamed, so its cost is latency
    // the user waits through.  One call for the answer and every source, not
    // one call per source.
    let calls: Calls = Arc::default();
    let _enc = stub_encoder(Some(calls.clone()));
    let cits: Vec<Cit> = (0..5)
        .map(|i| cit_with(Some(INVOICE), "d1", &format!("doc-{i}.pdf")))
        .collect();
    verified(&cits, "The Contoso invoice is for Rivera.");
    assert_eq!(calls.lock().len(), 1);
}

#[test]
#[serial(embeddings)]
fn test_duplicates_are_collapsed_before_embedding() {
    let calls: Calls = Arc::default();
    let _enc = stub_encoder(Some(calls.clone()));
    let out = verified(&[cit(), cit(), cit()], "The Contoso invoice is for Rivera.");
    assert_eq!(out.len(), 1);
    // One answer window plus one source window -- the same document is not
    // embedded three times.
    assert_eq!(calls.lock()[0].len(), 2);
}

#[test]
#[serial(embeddings)]
fn test_a_source_matching_only_in_its_tail_is_kept() {
    // A citation's extract is the whole document.  Scoring it as one blob
    // would average the matching passage away, and truncating to the first
    // window would never reach it -- so a document is scored on its closest
    // window, as retrieval scores one on its closest chunk.
    let _enc = stub_encoder(None);
    let buried = format!("{}{}", "filler text. ".repeat(400), INVOICE);
    let out = verified(&[cit_extract(Some(&buried))], "The Contoso invoice is for Rivera.");
    assert_eq!(out.len(), 1);
}

#[test]
#[serial(embeddings)]
fn test_a_long_document_is_capped_at_max_windows() {
    let calls: Calls = Arc::default();
    let _enc = stub_encoder(Some(calls.clone()));
    verified(&[cit_extract(Some(&"filler text. ".repeat(5000)))], "any answer");
    assert!(calls.lock()[0].len() - 1 <= MAX_WINDOWS);
}

#[test]
fn test_windows_sample_across_a_long_text_not_just_the_head() {
    let text: String = (0..200).map(|i| format!("[{i:04}]{}", "x".repeat(96))).collect();
    let w = windows(&text);
    assert!(w.len() <= MAX_WINDOWS);
    assert!(w.iter().all(|x| x.chars().count() <= WINDOW_CHARS));
    // The last window reaches material the first windows never see.
    let head: String = text.chars().take(WINDOW_CHARS * 2).collect();
    assert!(!head.contains(w.last().unwrap().as_str()));
}

#[test]
#[serial(embeddings)]
fn test_a_source_with_no_extract_cannot_be_judged_and_is_kept() {
    let _enc = stub_encoder(None);
    assert_eq!(verified(&[cit_extract(None)], "anything at all").len(), 1);
}

#[test]
#[serial(embeddings)]
fn test_verification_falls_back_to_the_lexical_check_without_an_embedder() {
    // No embedding model on this install: degrade to the weaker text check
    // rather than switching verification off and crediting everything.
    let _enc = no_embedder();
    assert!(verified(&[cit()], "The camera sensor is a Hasselblad.").is_empty());
    assert_eq!(verified(&[cit()], "Contoso paid Jordan Rivera $1,000.00").len(), 1);
}

#[test]
#[serial(embeddings)]
fn test_an_embedder_that_raises_never_costs_the_answer_its_sources() {
    embeddings::set_encode_texts_hook(Some(Arc::new(|_texts, _base_dir, _timeout| {
        panic!("llama context died")
    })));
    let _enc = EncoderGuard;
    let out = verified(&[cit(), cit()], "any answer");
    assert_eq!(out.len(), 1);
    assert!(no_private_keys(&out[0]));
}

// ---------------------------------------------------------------------------
// Which text a citation is judged on, and how many survive.
//
// `_extract` is the whole document: retrieval expands every match to the
// complete file before building the prompt, so a citation for a 40-page
// report carries all 40 pages.  Scoring that credits almost anything -- a
// long file covers enough ground that some page of it is related to any
// answer.  The chunks the document was retrieved *for* are the passages that
// made it a match, and each already carries the document reference the
// citation is built from, so those are what the answer is checked against.

const CAMERA_CHUNK: &str = "The Hasselblad camera uses a 100-megapixel sensor.";
const INVOICE_CHUNK: &str = "Invoice from Contoso Ltd. to Jordan Rivera.";

fn chunked(name: &str, chunks: &[&str], extract: &str) -> Cit {
    let mut c = Map::new();
    c.insert("document_id".into(), Value::String(name.into()));
    c.insert("document_name".into(), Value::String(name.into()));
    c.insert("_extract".into(), Value::String(extract.into()));
    c.insert(
        "_chunks".into(),
        Value::Array(chunks.iter().map(|s| Value::String(s.to_string())).collect()),
    );
    c
}

#[test]
fn test_a_citation_is_judged_on_its_chunks_not_the_whole_document() {
    let (vectors, texts) = citation_sources(&chunked("report.pdf", &[CAMERA_CHUNK], INVOICE));
    assert!(vectors.is_empty());
    assert_eq!(texts, vec![CAMERA_CHUNK.to_string()]);
}

#[test]
fn test_a_citation_without_chunks_falls_back_to_its_extract() {
    let (vectors, texts) = citation_sources(&chunked("report.pdf", &[], INVOICE));
    assert!(vectors.is_empty());
    assert_eq!(texts, windows(INVOICE));
}

#[test]
#[serial(embeddings)]
fn test_a_long_document_no_longer_rides_along_on_an_unrelated_passage() {
    // The document contains an invoice somewhere, but it was retrieved for its
    // camera passage.  Judged whole it looks like an invoice source; judged on
    // the chunk it matched on, it is not one.
    let _enc = stub_encoder(None);
    let answer = "The Contoso invoice is for Rivera.";
    let whole = format!("{CAMERA_CHUNK} {INVOICE}");
    let whole_doc = chunked("report.pdf", &[], &whole);
    assert_eq!(verified(&[whole_doc], answer).len(), 1);
    let by_chunk = chunked("report.pdf", &[CAMERA_CHUNK], &whole);
    assert!(verified(&[by_chunk], answer).is_empty());
}

// Three sources under the ceiling at spread-out distances, plus one further
// out.  This is the shape of a real run: a camera question answered from
// three web pages at 0.123 / 0.248 / 0.354, with two unrelated fund PDFs at
// 0.563 and 0.569 also retrieved.  All three pages were used.
const ANSWER: &str = "Invoice invoice invoice from Contoso to Rivera, invoice.";

fn near() -> Cit {
    chunked("near.pdf", &[INVOICE_CHUNK], INVOICE) // .18
}

fn mid() -> Cit {
    chunked("mid.pdf", &["Invoice for a camera sensor."], INVOICE) // .46
}

fn far() -> Cit {
    chunked("far.pdf", &["Invoice for a camera sensor hasselblad."], INVOICE)
}

#[test]
#[serial(embeddings)]
fn test_the_margin_is_wide_enough_for_sources_an_answer_spread_across() {
    // The failure this pins: at retrieval's own 0.15 the cutoff came out at
    // 0.273 on the run above and threw away the PCMag review of the exact
    // camera the question was about.  Sources an answer genuinely used spread
    // wider than chunks matching a query do.
    let _enc = stub_encoder(None);
    let distances = semantic_distances(&[near(), mid()], ANSWER, None).unwrap();
    assert!(
        distances[1].unwrap() - distances[0].unwrap() > 0.15,
        "fixture no longer spans the old margin"
    );
    let out = verified(&[near(), mid()], ANSWER);
    assert_eq!(names(&out), vec!["near.pdf", "mid.pdf"]);
}

#[test]
#[serial(embeddings)]
fn test_the_margin_still_drops_what_trails_the_closest_source() {
    // It is a real cutoff, not a formality: the ceiling alone is not a robust
    // separator — the unrelated PDFs in that run scored 0.563, only just
    // above it — so a source that trails the best by more than the margin
    // goes, even though it clears the ceiling on its own.
    let _enc = stub_encoder(None);
    let distances = semantic_distances(&[near(), far()], ANSWER, None).unwrap();
    assert!(distances.iter().all(|d| d.unwrap() <= CITATION_MAX_DISTANCE));
    assert!(distances[1].unwrap() - distances[0].unwrap() > CITATION_CUTOFF_MARGIN);

    let out = verified(&[near(), mid(), far()], ANSWER);
    assert_eq!(names(&out), vec!["near.pdf", "mid.pdf"]);
}

#[test]
#[serial(embeddings)]
fn test_sources_within_the_margin_of_the_best_are_all_kept() {
    let _enc = stub_encoder(None);
    let answer = "The Contoso invoice is for Rivera.";
    let a = chunked("a.pdf", &[INVOICE_CHUNK], INVOICE);
    let b = chunked("b.pdf", &["Contoso invoice, Rivera."], INVOICE);
    let out = verified(&[a, b], answer);
    assert_eq!(names(&out), vec!["a.pdf", "b.pdf"]);
}

#[test]
#[serial(embeddings)]
fn test_the_chunk_text_never_leaks_to_the_client() {
    let _enc = stub_encoder(None);
    let out = verified(&[chunked("report.pdf", &[INVOICE_CHUNK], INVOICE)], "The Contoso invoice is for Rivera.");
    assert!(!out.is_empty() && no_private_keys(&out[0]));
}

#[test]
#[serial(embeddings)]
fn test_the_lexical_fallback_also_judges_on_chunks() {
    let _enc = no_embedder();
    let answer = "Contoso paid Jordan Rivera $1,000.00 for contract work services.";
    let by_chunk = chunked("report.pdf", &[CAMERA_CHUNK], INVOICE);
    assert!(verified(&[by_chunk], answer).is_empty());
    let matching = chunked("report.pdf", &[INVOICE], CAMERA_CHUNK);
    assert_eq!(verified(&[matching], answer).len(), 1);
}

// ---------------------------------------------------------------------------
// What verification costs.
//
// Nothing on the source side.  A chunk's vector was computed when the
// document was indexed and read back out of the index to rank the search that
// produced this citation, so retrieval carries it along
// (`ChunkResult.embedding` -> `_vectors`).  Re-encoding that text here would be
// paying twice for the same numbers, after the answer has streamed, which is
// latency the user watches.  The answer is the only thing that has to be
// embedded.

/// A stored vector as retrieval carries it in `_vectors`: the BLOB bytes.
fn vec_blob(text: &str) -> Value {
    // Python: `_bag(text) or [1e-6] * 6` — a 6-element list is always truthy,
    // so this is the bag as-is.
    let v = Array1::from(bag(text));
    blob_to_value(&embedding_to_blob(v.view()))
}

/// A citation as retrieval builds it: chunk text and its stored vector.
fn indexed(name: &str, chunks: &[&str]) -> Cit {
    let mut c = chunked(name, chunks, INVOICE);
    c.insert("_vectors".into(), Value::Array(chunks.iter().map(|t| vec_blob(t)).collect()));
    c
}

#[test]
#[serial(embeddings)]
fn test_only_the_answer_is_embedded_when_retrieval_carried_the_vectors() {
    let calls: Calls = Arc::default();
    let _enc = stub_encoder(Some(calls.clone()));
    let answer = "The Contoso invoice is for Rivera.";
    let cits: Vec<Cit> = (0..5)
        .map(|i| indexed(&format!("doc-{i}.pdf"), &[INVOICE_CHUNK, CAMERA_CHUNK]))
        .collect();
    verified(&cits, answer);
    assert_eq!(*calls.lock(), vec![vec![answer.to_string()]]);
}

#[test]
#[serial(embeddings)]
fn test_a_carried_vector_scores_the_same_as_encoding_its_text() {
    let _enc = stub_encoder(None);
    let answer = "The Contoso invoice is for Rivera.";
    let carried = semantic_distances(&[indexed("report.pdf", &[CAMERA_CHUNK])], answer, None).unwrap();
    let encoded =
        semantic_distances(&[chunked("report.pdf", &[CAMERA_CHUNK], INVOICE)], answer, None).unwrap();
    let (a, b) = (carried[0].unwrap(), encoded[0].unwrap());
    // pytest.approx defaults: rel=1e-6, abs=1e-12.
    assert!((a - b).abs() <= (1e-6 * b.abs()).max(1e-12), "{a} != approx {b}");
}

#[test]
#[serial(embeddings)]
fn test_a_chunk_whose_vector_is_missing_is_encoded() {
    // The TF-IDF fallback path ranks without vectors and stores none, so the
    // text still has to be encoded — per chunk, not per citation.
    let calls: Calls = Arc::default();
    let _enc = stub_encoder(Some(calls.clone()));
    let mut c = indexed("report.pdf", &[INVOICE_CHUNK, CAMERA_CHUNK]);
    c.insert("_vectors".into(), Value::Array(vec![Value::Null, vec_blob(CAMERA_CHUNK)]));
    let answer = "The Contoso invoice is for Rivera.";
    verified(&[c], answer);
    assert_eq!(*calls.lock(), vec![vec![answer.to_string(), INVOICE_CHUNK.to_string()]]);
}

#[test]
#[serial(embeddings)]
fn test_a_vector_from_a_different_embedding_model_is_not_judged() {
    // Wrong dimensionality — the index was written by another model.  Decline
    // to score it rather than crash the response it belongs to; an
    // unjudgeable source is kept, since not measuring is not evidence of
    // misuse.
    let _enc = stub_encoder(None);
    let mut c = indexed("report.pdf", &[CAMERA_CHUNK]);
    let ones = Array1::<f32>::ones(3);
    c.insert("_vectors".into(), Value::Array(vec![blob_to_value(&embedding_to_blob(ones.view()))]));
    assert_eq!(
        semantic_distances(&[c.clone()], "The Contoso invoice is for Rivera.", None),
        Some(vec![None])
    );
    assert_eq!(verified(&[c], "The camera sensor is a Hasselblad.").len(), 1);
}

#[test]
#[serial(embeddings)]
fn test_carried_vectors_never_leak_to_the_client() {
    let _enc = stub_encoder(None);
    let out = verified(&[indexed("report.pdf", &[INVOICE_CHUNK])], "The Contoso invoice is for Rivera.");
    assert!(!out.is_empty());
    let keys: HashSet<String> = out[0].keys().cloned().collect();
    assert_eq!(keys, set(&["document_id", "document_name"]));
}
