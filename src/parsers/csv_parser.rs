//! Port of `smartloop/parsers/csv_parser.py`.
//!
//! A parser for processing .csv files into a Markdown table. Python's
//! `csv.reader` (default `excel` dialect) is hand-ported in [`py_csv_reader`].

use std::path::Path;

use crate::parsers::base_parser::{read_text_universal, BaseParser, ParserError, SharedLLM};
use crate::paths::expand_home;

/// A parser for processing .csv files.
pub struct CsvParser {
    pub llm: Option<SharedLLM>,
    pub file_path: String,
}

impl CsvParser {
    pub fn new(llm: Option<SharedLLM>, file_path: impl Into<String>) -> Self {
        Self { llm, file_path: file_path.into() }
    }

    /// Expand `~` and fail with `File not found: {path}` when missing.
    fn existing_path(&self) -> anyhow::Result<String> {
        let filepath = expand_home(&self.file_path);
        if !Path::new(&filepath).exists() {
            return Err(ParserError::FileNotFound(filepath).into());
        }
        Ok(filepath)
    }

    /// Read every CSV row of the file.
    fn read_rows(&self) -> anyhow::Result<Vec<Vec<String>>> {
        let filepath = self.existing_path()?;
        let text = read_text_universal(&filepath)?;
        py_csv_reader(&text)
    }

    /// Validates the CSV file structure: not empty and every row has the same
    /// number of columns as the header.
    pub fn is_valid_csv(&self) -> anyhow::Result<bool> {
        let rows = self.read_rows()?;

        if rows.is_empty() {
            return Err(ParserError::Value("CSV file is empty.".to_string()).into());
        }

        let header_length = rows[0].len();
        for (i, row) in rows.iter().enumerate() {
            if row.len() != header_length {
                return Err(
                    ParserError::Value(format!("Row {} has an inconsistent number of columns.", i + 1)).into()
                );
            }
        }

        Ok(true)
    }

    /// Converts the CSV file content to a Markdown table.
    pub fn convert_to_markdown(&self) -> anyhow::Result<String> {
        let rows = self.read_rows()?;

        if rows.is_empty() {
            return Err(ParserError::Value("CSV file is empty.".to_string()).into());
        }

        let mut markdown_lines = Vec::with_capacity(rows.len() + 1);

        // Add header row
        let header = &rows[0];
        markdown_lines.push(format!("| {} |", header.join(" | ")));
        markdown_lines.push(format!("| {} |", vec!["---"; header.len()].join(" | ")));

        // Add data rows
        for row in &rows[1..] {
            markdown_lines.push(format!("| {} |", row.join(" | ")));
        }

        Ok(markdown_lines.join("\n"))
    }
}

impl BaseParser for CsvParser {
    fn llm(&self) -> Option<&SharedLLM> {
        self.llm.as_ref()
    }
    fn file_path(&self) -> &str {
        &self.file_path
    }
    fn type_name(&self) -> &'static str {
        "CsvParser"
    }
    fn parse(&self) -> anyhow::Result<String> {
        self.existing_path()?;
        self.convert_to_markdown()
    }
}

/// Parser states of CPython's `_csv.c` reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CsvState {
    StartRecord,
    StartField,
    InField,
    InQuotedField,
    QuoteInQuotedField,
    EatCrnl,
}

/// One input character, or the end-of-line marker `_csv.c` feeds after each line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CsvInput {
    Char(char),
    Eol,
}

/// Hand-port of Python's `list(csv.reader(text_lines))` with the default
/// `excel` dialect (`,` delimiter, `"` quote char, doubled quotes, no
/// escape char, non-strict). `text` is iterated line by line like a file
/// object (lines keep their `\n`), blank lines yield empty rows, and a quoted
/// field may span lines.
pub fn py_csv_reader(text: &str) -> anyhow::Result<Vec<Vec<String>>> {
    const DELIMITER: char = ',';
    const QUOTECHAR: char = '"';

    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut fields: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut field_len: usize = 0;
    let mut state = CsvState::StartRecord;

    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let mut line_idx = 0;

    loop {
        if line_idx >= lines.len() {
            // End of input: a pending field (or an unterminated quote) is
            // saved as the final record in non-strict mode.
            if field_len != 0 || state == CsvState::InQuotedField {
                fields.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut fields));
            }
            break;
        }

        // Process one line, then the EOL marker; repeat while a record spans lines.
        loop {
            let line = lines[line_idx];
            line_idx += 1;
            let inputs = line.chars().map(CsvInput::Char).chain(std::iter::once(CsvInput::Eol));
            for input in inputs {
                process_csv_char(
                    input,
                    &mut state,
                    &mut fields,
                    &mut field,
                    &mut field_len,
                    DELIMITER,
                    QUOTECHAR,
                )?;
            }
            if state == CsvState::StartRecord || line_idx >= lines.len() {
                break;
            }
        }

        if state == CsvState::StartRecord {
            rows.push(std::mem::take(&mut fields));
        }
    }

    Ok(rows)
}

fn process_csv_char(
    input: CsvInput,
    state: &mut CsvState,
    fields: &mut Vec<String>,
    field: &mut String,
    field_len: &mut usize,
    delimiter: char,
    quotechar: char,
) -> anyhow::Result<()> {
    let save_field = |fields: &mut Vec<String>, field: &mut String, field_len: &mut usize| {
        fields.push(std::mem::take(field));
        *field_len = 0;
    };
    let is_newline = |c: char| c == '\n' || c == '\r';

    match *state {
        CsvState::StartRecord => match input {
            // Empty line: return [].
            CsvInput::Eol => return Ok(()),
            CsvInput::Char(c) if is_newline(c) => {
                *state = CsvState::EatCrnl;
                return Ok(());
            }
            CsvInput::Char(_) => {
                // Normal character: handle as start of field.
                *state = CsvState::StartField;
                return process_csv_char(input, state, fields, field, field_len, delimiter, quotechar);
            }
        },
        CsvState::StartField => match input {
            CsvInput::Eol => {
                save_field(fields, field, field_len);
                *state = CsvState::StartRecord;
            }
            CsvInput::Char(c) if is_newline(c) => {
                save_field(fields, field, field_len);
                *state = CsvState::EatCrnl;
            }
            CsvInput::Char(c) if c == quotechar => {
                *state = CsvState::InQuotedField;
            }
            CsvInput::Char(c) if c == delimiter => {
                save_field(fields, field, field_len);
            }
            CsvInput::Char(c) => {
                field.push(c);
                *field_len += 1;
                *state = CsvState::InField;
            }
        },
        CsvState::InField => match input {
            CsvInput::Eol => {
                save_field(fields, field, field_len);
                *state = CsvState::StartRecord;
            }
            CsvInput::Char(c) if is_newline(c) => {
                save_field(fields, field, field_len);
                *state = CsvState::EatCrnl;
            }
            CsvInput::Char(c) if c == delimiter => {
                save_field(fields, field, field_len);
                *state = CsvState::StartField;
            }
            CsvInput::Char(c) => {
                field.push(c);
                *field_len += 1;
            }
        },
        CsvState::InQuotedField => match input {
            CsvInput::Eol => {}
            CsvInput::Char(c) if c == quotechar => {
                *state = CsvState::QuoteInQuotedField;
            }
            CsvInput::Char(c) => {
                field.push(c);
                *field_len += 1;
            }
        },
        CsvState::QuoteInQuotedField => match input {
            CsvInput::Char(c) if c == quotechar => {
                // Doubled quote: a literal quote char.
                field.push(c);
                *field_len += 1;
                *state = CsvState::InQuotedField;
            }
            CsvInput::Char(c) if c == delimiter => {
                save_field(fields, field, field_len);
                *state = CsvState::StartField;
            }
            CsvInput::Eol => {
                save_field(fields, field, field_len);
                *state = CsvState::StartRecord;
            }
            CsvInput::Char(c) if is_newline(c) => {
                save_field(fields, field, field_len);
                *state = CsvState::EatCrnl;
            }
            CsvInput::Char(c) => {
                // Non-strict: keep the character after the closing quote.
                field.push(c);
                *field_len += 1;
                *state = CsvState::InField;
            }
        },
        CsvState::EatCrnl => match input {
            CsvInput::Char(c) if is_newline(c) => {}
            CsvInput::Eol => *state = CsvState::StartRecord,
            CsvInput::Char(_) => {
                anyhow::bail!(
                    "new-line character seen in unquoted field - do you need to open the file with newline=''?"
                );
            }
        },
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::py_csv_reader;

    #[test]
    fn csv_reader_matches_python_semantics() {
        let rows = py_csv_reader("a,b\n1,\"x, y\"\n\n\"multi\nline\",\"he said \"\"hi\"\"\"\n").unwrap();
        assert_eq!(
            rows,
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["1".to_string(), "x, y".to_string()],
                vec![],
                vec!["multi\nline".to_string(), "he said \"hi\"".to_string()],
            ]
        );
        assert_eq!(py_csv_reader("x,y").unwrap(), vec![vec!["x".to_string(), "y".to_string()]]);
        assert_eq!(py_csv_reader("a,\n").unwrap(), vec![vec!["a".to_string(), String::new()]]);
        assert!(py_csv_reader("").unwrap().is_empty());
    }
}
