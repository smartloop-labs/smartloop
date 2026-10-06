//! PDF → Markdown: hand-port of MarkItDown's `PdfConverter`.
//!
//! MarkItDown reads PDFs with `pdfplumber` (word positions, for form-style
//! pages with aligned columns) and `pdfminer` (plain text). Both are
//! replaced by MuPDF: words are rebuilt from MuPDF's positioned characters
//! with pdfplumber's clustering rules, and plain text follows pdfminer's
//! layout (text boxes followed by a blank line, a form feed per page).

use std::collections::{BTreeMap, BTreeSet};

use once_cell::sync::Lazy;
use regex::Regex;

use super::{ConvertOptions, DocumentConverter, DocumentConverterResult, StreamInfo};

/// Pattern for MasterFormat-style partial numbering (e.g., ".1", ".2", ".10").
static PARTIAL_NUMBERING_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\.\d+$").expect("valid regex"));

/// pdfplumber's ligature expansions.
const LIGATURES: &[(&str, &str)] = &[
    ("\u{fb00}", "ff"),
    ("\u{fb03}", "ffi"),
    ("\u{fb04}", "ffl"),
    ("\u{fb01}", "fi"),
    ("\u{fb02}", "fl"),
    ("\u{fb06}", "st"),
    ("\u{fb05}", "st"),
];

/// Post-process extracted text to merge MasterFormat-style partial numbering
/// with the following text line.
///
/// MasterFormat documents use partial numbering like ".1  The intent of this
/// Request for Proposal..."; some PDF extractors split these into separate
/// lines, which this function merges back together.
pub fn merge_partial_numbering_lines(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut result_lines: Vec<String> = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let stripped = line.trim();

        // Check if this line is ONLY a partial numbering
        if PARTIAL_NUMBERING_PATTERN.is_match(stripped) {
            // Look for the next non-empty line to merge with
            let mut j = i + 1;
            while j < lines.len() && lines[j].trim().is_empty() {
                j += 1;
            }

            if j < lines.len() {
                // Merge the partial numbering with the next line
                result_lines.push(format!("{stripped} {}", lines[j].trim()));
                i = j + 1; // Skip past the merged line
            } else {
                // No next line to merge with, keep as is
                result_lines.push(line.to_string());
                i += 1;
            }
        } else {
            result_lines.push(line.to_string());
            i += 1;
        }
    }

    result_lines.join("\n")
}

/// Converts PDFs to Markdown, extracting form-style tables into aligned
/// Markdown and falling back to plain text extraction.
pub struct PdfConverter;

impl DocumentConverter for PdfConverter {
    fn name(&self) -> &'static str {
        "PdfConverter"
    }

    fn accepts(&self, info: &StreamInfo) -> bool {
        info.matches(&[".pdf"], &["application/pdf", "application/x-pdf"])
    }

    fn convert(&self, data: &[u8], _info: &StreamInfo, _options: &ConvertOptions) -> anyhow::Result<DocumentConverterResult> {
        let mut markdown = match extract_with_word_positions(data) {
            // If no pages had form-style content, use plain extraction for
            // the whole document (better text spacing for prose).
            Ok(None) | Err(_) => extract_text(data)?,
            Ok(Some(markdown)) => markdown,
        };

        // Fallback if still empty
        if markdown.is_empty() {
            markdown = extract_text(data)?;
        }

        // Post-process to merge MasterFormat-style partial numbering with following text
        Ok(DocumentConverterResult::markdown(merge_partial_numbering_lines(&markdown)))
    }
}

/// A positioned character (pdfplumber's char dict).
#[derive(Debug, Clone)]
struct PdfChar {
    text: String,
    x0: f64,
    x1: f64,
    top: f64,
    bottom: f64,
}

/// A word (pdfplumber's word dict).
#[derive(Debug, Clone)]
pub struct PdfWord {
    pub text: String,
    pub x0: f64,
    pub x1: f64,
    pub top: f64,
    pub bottom: f64,
}

/// A page's characters and width.
struct PdfPage {
    chars: Vec<PdfChar>,
    width: f64,
}

fn load_pages(data: &[u8]) -> anyhow::Result<Vec<PdfPage>> {
    let doc = mupdf::Document::from_bytes(data, "application/pdf")?;
    let mut pages = Vec::new();
    for page in doc.pages()? {
        let page = page?;
        let bounds = page.bounds()?;
        // pdfplumber sees only the glyphs the PDF draws: no synthesized spaces.
        let text_page = page.to_text_page(
            mupdf::TextPageFlags::PRESERVE_LIGATURES
                | mupdf::TextPageFlags::PRESERVE_WHITESPACE
                | mupdf::TextPageFlags::INHIBIT_SPACES,
        )?;
        let mut chars = Vec::new();
        for block in text_page.blocks() {
            if block.r#type() != mupdf::text_page::TextBlockType::Text {
                continue;
            }
            for line in block.lines() {
                for ch in line.chars() {
                    let Some(c) = ch.char() else { continue };
                    let bbox = mupdf::Rect::from(ch.quad());
                    chars.push(PdfChar {
                        text: c.to_string(),
                        x0: (bbox.x0 - bounds.x0) as f64,
                        x1: (bbox.x1 - bounds.x0) as f64,
                        top: (bbox.y0 - bounds.y0) as f64,
                        bottom: (bbox.y1 - bounds.y0) as f64,
                    });
                }
            }
        }
        pages.push(PdfPage { chars, width: (bounds.x1 - bounds.x0) as f64 });
    }
    Ok(pages)
}

/// pdfplumber `cluster_list` + `cluster_objects`: group items whose key
/// values chain within `tolerance`, ordered by cluster value (stable).
fn cluster_objects<T: Clone>(items: &[T], key: impl Fn(&T) -> f64, tolerance: f64) -> Vec<Vec<T>> {
    let mut values: Vec<f64> = items.iter().map(&key).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup();

    let mut cluster_of: Vec<(f64, usize)> = Vec::new();
    let mut index = 0;
    let mut last: Option<f64> = None;
    for v in values {
        if let Some(l) = last
            && v > l + tolerance
        {
            index += 1;
        }
        cluster_of.push((v, index));
        last = Some(v);
    }
    let lookup = |v: f64| cluster_of.iter().find(|(value, _)| *value == v).map(|(_, i)| *i).unwrap_or(0);

    let mut tagged: Vec<(usize, T)> = items.iter().map(|item| (lookup(key(item)), item.clone())).collect();
    tagged.sort_by_key(|(cluster, _)| *cluster);

    let mut out: Vec<Vec<T>> = Vec::new();
    let mut current: Option<usize> = None;
    for (cluster, item) in tagged {
        if current != Some(cluster) {
            out.push(Vec::new());
            current = Some(cluster);
        }
        out.last_mut().expect("pushed").push(item);
    }
    out
}

/// pdfplumber `WordExtractor` (upright, left-to-right text).
fn extract_words(chars: &[PdfChar], keep_blank_chars: bool, x_tolerance: f64, y_tolerance: f64) -> Vec<PdfWord> {
    let mut words = Vec::new();
    if chars.is_empty() {
        return words;
    }
    let lines = cluster_objects(chars, |c| c.top, y_tolerance);
    for mut line in lines {
        line.sort_by(|a, b| (a.x0, a.x0).partial_cmp(&(b.x0, b.x0)).unwrap_or(std::cmp::Ordering::Equal));
        let mut current: Vec<PdfChar> = Vec::new();
        let flush = |current: &mut Vec<PdfChar>, words: &mut Vec<PdfWord>| {
            if !current.is_empty() {
                words.push(merge_chars(current));
                current.clear();
            }
        };
        for ch in line {
            let is_space = !ch.text.is_empty() && ch.text.chars().all(char::is_whitespace);
            if !keep_blank_chars && is_space {
                flush(&mut current, &mut words);
            } else if let Some(prev) = current.last() {
                // Intraline test, then interline test.
                let begins_new_word =
                    ch.x0 < prev.x0 || ch.x0 > prev.x1 + x_tolerance || (ch.top - prev.top).abs() > y_tolerance;
                if begins_new_word {
                    flush(&mut current, &mut words);
                }
                current.push(ch);
            } else {
                current.push(ch);
            }
        }
        flush(&mut current, &mut words);
    }
    words
}

fn merge_chars(chars: &[PdfChar]) -> PdfWord {
    let text: String = chars
        .iter()
        .map(|c| LIGATURES.iter().find(|(l, _)| *l == c.text).map(|(_, e)| e.to_string()).unwrap_or_else(|| c.text.clone()))
        .collect();
    PdfWord {
        text,
        x0: chars.iter().map(|c| c.x0).fold(f64::INFINITY, f64::min),
        x1: chars.iter().map(|c| c.x1).fold(f64::NEG_INFINITY, f64::max),
        top: chars.iter().map(|c| c.top).fold(f64::INFINITY, f64::min),
        bottom: chars.iter().map(|c| c.bottom).fold(f64::NEG_INFINITY, f64::max),
    }
}

/// pdfplumber `page.extract_text()` (non-layout): words grouped into lines by
/// `top`, single spaces between words and newlines between lines.
fn plumber_extract_text(page: &PdfPage) -> String {
    let words = extract_words(&page.chars, false, 3.0, 3.0);
    let mut sorted = words;
    sorted.sort_by(|a, b| a.top.partial_cmp(&b.top).unwrap_or(std::cmp::Ordering::Equal));
    let lines = cluster_objects(&sorted, |w| w.top, 3.0);
    lines
        .into_iter()
        .map(|mut line| {
            line.sort_by(|a, b| (a.x0, a.x0).partial_cmp(&(b.x0, b.x0)).unwrap_or(std::cmp::Ordering::Equal));
            line.iter().map(|w| w.text.clone()).collect::<Vec<_>>().join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Python `round()` (half to even).
fn py_round(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 { 2.0 * (x / 2.0).round() } else { r }
}

/// `str.ljust(width)` by characters.
fn ljust(s: &str, width: usize) -> String {
    let len = s.chars().count();
    if len >= width { s.to_string() } else { format!("{s}{}", " ".repeat(width - len)) }
}

struct RowInfo {
    words: Vec<PdfWord>,
    text: String,
    is_paragraph: bool,
    num_columns: usize,
    x_groups: Vec<f64>,
    has_partial_numbering: bool,
    is_table_row: bool,
}

/// Extract form-style content from a PDF page by analyzing word positions.
/// This handles borderless forms/tables where words are aligned in columns.
///
/// Returns markdown with proper table formatting (pipe-separated columns with
/// header separator rows; non-table content as plain text), or `None` if the
/// page doesn't appear to be a form-style document.
fn extract_form_content_from_words(page: &PdfPage) -> Option<String> {
    let words = extract_words(&page.chars, true, 3.0, 3.0);
    if words.is_empty() {
        return None;
    }

    // Group words by their Y position (rows)
    let y_tolerance = 5.0;
    let mut rows_by_y: BTreeMap<i64, Vec<PdfWord>> = BTreeMap::new();
    for word in words {
        let y_key = (py_round(word.top / y_tolerance) * y_tolerance) as i64;
        rows_by_y.entry(y_key).or_default().push(word);
    }

    let page_width = page.width;

    // First pass: analyze each row (sorted by Y position)
    let mut row_info: Vec<RowInfo> = Vec::new();
    for (_, mut row_words) in rows_by_y {
        row_words.sort_by(|a, b| a.x0.partial_cmp(&b.x0).unwrap_or(std::cmp::Ordering::Equal));
        if row_words.is_empty() {
            continue;
        }

        let first_x0 = row_words[0].x0;
        let last_x1 = row_words[row_words.len() - 1].x1;
        let line_width = last_x1 - first_x0;
        let combined_text = row_words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");

        // Count distinct x-position groups (columns)
        let mut x_positions: Vec<f64> = row_words.iter().map(|w| w.x0).collect();
        x_positions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut x_groups: Vec<f64> = Vec::new();
        for x in x_positions {
            if x_groups.last().is_none_or(|last| x - last > 50.0) {
                x_groups.push(x);
            }
        }

        // Determine row type
        let is_paragraph = line_width > page_width * 0.55 && combined_text.chars().count() > 60;

        // Check for MasterFormat-style partial numbering (e.g., ".1", ".2").
        // These should be treated as list items, not table rows.
        let has_partial_numbering = PARTIAL_NUMBERING_PATTERN.is_match(row_words[0].text.trim());

        row_info.push(RowInfo {
            num_columns: x_groups.len(),
            words: row_words,
            text: combined_text,
            is_paragraph,
            x_groups,
            has_partial_numbering,
            is_table_row: false,
        });
    }

    // Collect ALL x-positions from rows with 3+ columns (table-like rows).
    // This gives us the global column structure.
    let mut all_table_x_positions: Vec<f64> = Vec::new();
    for info in &row_info {
        if info.num_columns >= 3 && !info.is_paragraph {
            all_table_x_positions.extend(info.x_groups.iter().copied());
        }
    }

    if all_table_x_positions.is_empty() {
        return None;
    }

    // Compute adaptive column clustering tolerance based on gap analysis
    all_table_x_positions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Calculate gaps between consecutive x-positions
    let gaps: Vec<f64> = all_table_x_positions
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|gap| *gap > 5.0) // Only significant gaps
        .collect();

    // Determine optimal tolerance using statistical analysis
    let adaptive_tolerance = if gaps.len() >= 3 {
        // Use 70th percentile of gaps as threshold (balances precision/recall)
        let mut sorted_gaps = gaps.clone();
        sorted_gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let percentile_70_idx = (sorted_gaps.len() as f64 * 0.70) as usize;
        // Clamp tolerance to reasonable range [25, 50]
        sorted_gaps[percentile_70_idx].clamp(25.0, 50.0)
    } else {
        // Fallback to conservative value
        35.0
    };

    // Compute global column boundaries using adaptive tolerance
    let mut global_columns: Vec<f64> = Vec::new();
    for x in &all_table_x_positions {
        if global_columns.last().is_none_or(|last| x - last > adaptive_tolerance) {
            global_columns.push(*x);
        }
    }

    // Adaptive max column check based on page characteristics
    if global_columns.len() > 1 {
        let content_width = global_columns[global_columns.len() - 1] - global_columns[0];
        let avg_col_width = content_width / global_columns.len() as f64;

        // Forms with very narrow columns (< 30px) are likely dense text
        if avg_col_width < 30.0 {
            return None;
        }

        // Compute adaptive max based on columns per inch. Typical forms have
        // 3-8 columns per inch; if density is too high (> 10 cols/inch), it is
        // likely not a form.
        let columns_per_inch = global_columns.len() as f64 / (content_width / 72.0);
        if columns_per_inch > 10.0 {
            return None;
        }

        // Adaptive max: allow more columns for wider pages. Standard letter is
        // 612pt wide, so scale accordingly (at least 15).
        let adaptive_max_columns = ((20.0 * (page_width / 612.0)) as i64).max(15);
        if global_columns.len() as i64 > adaptive_max_columns {
            return None;
        }
    } else {
        // Single column, not a form
        return None;
    }

    // Now classify each row as table row or not. A row is a table row if it
    // has words that align with 2+ of the global columns.
    for info in row_info.iter_mut() {
        if info.is_paragraph || info.has_partial_numbering {
            // Rows with partial numbering (e.g., ".1", ".2") are list items, not table rows
            info.is_table_row = false;
            continue;
        }

        // Count how many global columns this row's words align with
        let mut aligned_columns: BTreeSet<usize> = BTreeSet::new();
        for word in &info.words {
            if let Some(col_idx) = global_columns.iter().position(|col_x| (word.x0 - col_x).abs() < 40.0) {
                aligned_columns.insert(col_idx);
            }
        }

        // If row uses 2+ of the established columns, it's a table row
        info.is_table_row = aligned_columns.len() >= 2;
    }

    // Find table regions (consecutive table rows)
    let mut table_regions: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < row_info.len() {
        if row_info[i].is_table_row {
            let start_idx = i;
            while i < row_info.len() && row_info[i].is_table_row {
                i += 1;
            }
            table_regions.push((start_idx, i));
        } else {
            i += 1;
        }
    }

    // Check if enough rows are table rows (at least 20%)
    let total_table_rows: usize = table_regions.iter().map(|(start, end)| end - start).sum();
    if !row_info.is_empty() && (total_table_rows as f64 / row_info.len() as f64) < 0.2 {
        return None;
    }

    // Build output - collect table data first, then format with proper column widths
    let mut result_lines: Vec<String> = Vec::new();
    let num_cols = global_columns.len();

    // Helper function to extract cells from a row
    let extract_cells = |info: &RowInfo| -> Vec<String> {
        let mut cells = vec![String::new(); num_cols];
        for word in &info.words {
            // Find the correct column using boundary ranges (default to last column)
            let assigned_col = (0..num_cols - 1).find(|&col_idx| word.x0 < global_columns[col_idx + 1] - 20.0).unwrap_or(num_cols - 1);
            if cells[assigned_col].is_empty() {
                cells[assigned_col] = word.text.clone();
            } else {
                cells[assigned_col].push(' ');
                cells[assigned_col].push_str(&word.text);
            }
        }
        cells
    };

    // Process rows, collecting table data for proper formatting
    let mut idx = 0;
    while idx < row_info.len() {
        // Check if this row starts a table region
        if let Some(&(start, end)) = table_regions.iter().find(|(start, _)| *start == idx) {
            // Collect all rows in this table
            let table_data: Vec<Vec<String>> = (start..end).map(|t| extract_cells(&row_info[t])).collect();

            // Calculate column widths for this table (minimum 3 for separator dashes)
            let col_widths: Vec<usize> = (0..num_cols)
                .map(|col| table_data.iter().map(|row| row[col].chars().count()).max().unwrap_or(0).max(3))
                .collect();

            let format_row = |row: &[String]| {
                format!(
                    "| {} |",
                    row.iter().enumerate().map(|(i, cell)| ljust(cell, col_widths[i])).collect::<Vec<_>>().join(" | ")
                )
            };

            // Format header row
            result_lines.push(format_row(&table_data[0]));
            // Format separator row
            result_lines.push(format!(
                "| {} |",
                (0..num_cols).map(|i| "-".repeat(col_widths[i])).collect::<Vec<_>>().join(" | ")
            ));
            // Format data rows
            for row in &table_data[1..] {
                result_lines.push(format_row(row));
            }

            idx = end; // Skip to end of table region
        } else {
            // Check if we're inside a table region (not at start)
            let in_table = table_regions.iter().any(|(start, end)| *start < idx && idx < *end);
            if !in_table {
                // Non-table content
                result_lines.push(row_info[idx].text.clone());
            }
            idx += 1;
        }
    }

    Some(result_lines.join("\n"))
}

/// The pdfplumber pass: `Ok(Some(markdown))` when at least one page has
/// form-style content, `Ok(None)` when none does.
fn extract_with_word_positions(data: &[u8]) -> anyhow::Result<Option<String>> {
    let pages = load_pages(data)?;
    let mut markdown_chunks: Vec<String> = Vec::new();
    let mut form_page_count = 0;

    for page in &pages {
        match extract_form_content_from_words(page) {
            Some(page_content) => {
                form_page_count += 1;
                if !page_content.trim().is_empty() {
                    markdown_chunks.push(page_content);
                }
            }
            None => {
                let text = plumber_extract_text(page);
                if !text.trim().is_empty() {
                    markdown_chunks.push(text.trim().to_string());
                }
            }
        }
    }

    if form_page_count == 0 {
        return Ok(None);
    }
    Ok(Some(markdown_chunks.join("\n\n").trim().to_string()))
}

/// `pdfminer.high_level.extract_text`: each text box's lines (newline
/// terminated) followed by a blank line, and a form feed after every page.
pub fn extract_text(data: &[u8]) -> anyhow::Result<String> {
    let doc = mupdf::Document::from_bytes(data, "application/pdf")?;
    let mut out = String::new();
    for page in doc.pages()? {
        let page = page?;
        let text_page = page.to_text_page(mupdf::TextPageFlags::empty())?;
        for block in text_page.blocks() {
            if block.r#type() != mupdf::text_page::TextBlockType::Text {
                continue;
            }
            for line in block.lines() {
                let text: String = line.chars().filter_map(|c| c.char()).collect();
                out.push_str(&text);
                out.push('\n');
            }
            out.push('\n');
        }
        out.push('\u{c}');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_partial_numbering() {
        assert_eq!(merge_partial_numbering_lines(".1\n\nThe intent\nnext"), ".1 The intent\nnext");
        assert_eq!(merge_partial_numbering_lines("a\n.2"), "a\n.2");
    }

    #[test]
    fn python_rounding() {
        assert_eq!(py_round(2.5), 2.0);
        assert_eq!(py_round(3.5), 4.0);
        assert_eq!(py_round(2.4), 2.0);
    }

    #[test]
    fn clusters_chain_within_tolerance() {
        let clusters = cluster_objects(&[1.0, 10.0, 2.5, 4.0], |v| *v, 1.5);
        assert_eq!(clusters, vec![vec![1.0, 2.5, 4.0], vec![10.0]]);
    }
}
