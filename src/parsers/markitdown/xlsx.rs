//! XLSX → Markdown: hand-port of MarkItDown's `XlsxConverter` /
//! `XlsConverter`.
//!
//! MarkItDown reads every sheet with `pandas.read_excel(..., engine="openpyxl")`
//! and renders it with `DataFrame.to_html(index=False)` before converting the
//! HTML to Markdown. Both pandas steps are emulated over `calamine`:
//!
//! * cells are converted like pandas' openpyxl reader (`_convert_cell`:
//!   integral numbers become ints, empty cells "", errors NaN, date-formatted
//!   numbers datetimes/times following `openpyxl.utils.datetime.from_excel`),
//!   trailing empty cells/rows are trimmed and rows padded to equal width;
//! * the first row becomes the header (`Unnamed: i` for blanks, duplicates
//!   mangled to `name.1`, ...), and each column's type is inferred like the
//!   Python parser (`_infer_types`: NA strings, numeric strings, booleans,
//!   object columns with the `sanitize_objects` memo);
//! * values are formatted like `to_html` (pandas' float formatter with six
//!   digits of precision and equal zero trimming, date-only datetimes, `NaN`,
//!   `NaT`, escaped control characters).

use std::io::Cursor;

use calamine::{Data, ExcelDateTime, ExcelDateTimeType, Reader, SheetType, Xlsx};
use chrono::{Duration, NaiveDate, NaiveDateTime, NaiveTime, Timelike};

use super::html::convert_html;
use super::{ConvertOptions, DocumentConverter, DocumentConverterResult, StreamInfo};

const MISSING_DEPENDENCY_MESSAGE: &str = "{converter} recognized the input as a potential {extension} file, but the dependencies needed to read {extension} files have not been installed. To resolve this error, include the optional dependency [{feature}] or [all] when installing MarkItDown. For example:

* pip install markitdown[{feature}]
* pip install markitdown[all]
* pip install markitdown[{feature}, ...]
* etc.";

/// Converts XLSX files to Markdown, with each sheet presented as a separate
/// Markdown table.
pub struct XlsxConverter;

impl DocumentConverter for XlsxConverter {
    fn name(&self) -> &'static str {
        "XlsxConverter"
    }

    fn accepts(&self, info: &StreamInfo) -> bool {
        info.matches(&[".xlsx"], &["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"])
    }

    fn convert(&self, data: &[u8], _info: &StreamInfo, _options: &ConvertOptions) -> anyhow::Result<DocumentConverterResult> {
        let mut workbook: Xlsx<_> = Xlsx::new(Cursor::new(data.to_vec()))?;
        let sheet_names: Vec<String> = workbook
            .sheets_metadata()
            .iter()
            .filter(|s| s.typ == SheetType::WorkSheet)
            .map(|s| s.name.clone())
            .collect();

        let mut md_content = String::new();
        for name in sheet_names {
            let range = workbook.worksheet_range(&name)?;
            md_content.push_str(&format!("## {name}\n"));
            let frame = DataFrame::from_rows(sheet_rows(&range));
            let html_content = frame.to_html();
            md_content.push_str(convert_html(&html_content).text_content.trim());
            md_content.push_str("\n\n");
        }

        Ok(DocumentConverterResult::markdown(md_content.trim()))
    }
}

/// Converts XLS files to Markdown. The Python installation ships without the
/// `xlrd` dependency (MarkItDown's `[xls]` extra), so the converter always
/// reports the missing dependency, as MarkItDown does.
pub struct XlsConverter;

impl DocumentConverter for XlsConverter {
    fn name(&self) -> &'static str {
        "XlsConverter"
    }

    fn accepts(&self, info: &StreamInfo) -> bool {
        info.matches(&[".xls"], &["application/vnd.ms-excel", "application/excel"])
    }

    fn convert(&self, _data: &[u8], _info: &StreamInfo, _options: &ConvertOptions) -> anyhow::Result<DocumentConverterResult> {
        anyhow::bail!(
            "{}",
            MISSING_DEPENDENCY_MESSAGE
                .replace("{converter}", "XlsConverter")
                .replace("{extension}", ".xls")
                .replace("{feature}", "xls")
        )
    }
}

// -- cell values ------------------------------------------------------------------

/// A Python value as pandas' openpyxl reader produces it.
#[derive(Debug, Clone, PartialEq)]
pub enum PyValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    DateTime(NaiveDateTime),
    Time(NaiveTime),
    TimeDelta(Duration),
    /// `np.nan` (error cells, missing values).
    NaN,
}

impl PyValue {
    /// The empty string an empty cell becomes.
    fn empty() -> Self {
        PyValue::Str(String::new())
    }

    fn is_empty_str(&self) -> bool {
        matches!(self, PyValue::Str(s) if s.is_empty())
    }

    /// Numeric value for Python's cross-type equality (`1 == 1.0 == True`).
    fn as_number(&self) -> Option<f64> {
        match self {
            PyValue::Int(i) => Some(*i as f64),
            PyValue::Float(f) => Some(*f),
            PyValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// Python `==` between two values.
    fn py_eq(&self, other: &PyValue) -> bool {
        match (self.as_number(), other.as_number()) {
            (Some(a), Some(b)) => a == b,
            (None, None) => match (self, other) {
                (PyValue::Str(a), PyValue::Str(b)) => a == b,
                (PyValue::DateTime(a), PyValue::DateTime(b)) => a == b,
                (PyValue::Time(a), PyValue::Time(b)) => a == b,
                (PyValue::TimeDelta(a), PyValue::TimeDelta(b)) => a == b,
                _ => false,
            },
            _ => false,
        }
    }

    /// Python `str()` of the value.
    fn py_str(&self) -> String {
        match self {
            PyValue::Int(i) => i.to_string(),
            PyValue::Float(f) => crate::parsers::json_parser::py_float_repr(*f),
            PyValue::Bool(b) => if *b { "True" } else { "False" }.to_string(),
            PyValue::Str(s) => s.clone(),
            PyValue::DateTime(dt) => py_datetime_str(dt),
            PyValue::Time(t) => py_time_str(t),
            PyValue::TimeDelta(d) => py_timedelta_str(d),
            PyValue::NaN => "nan".to_string(),
        }
    }
}

fn py_time_str(t: &NaiveTime) -> String {
    let micros = t.nanosecond() / 1000;
    if micros == 0 { t.format("%H:%M:%S").to_string() } else { format!("{}.{:06}", t.format("%H:%M:%S"), micros) }
}

fn py_datetime_str(dt: &NaiveDateTime) -> String {
    format!("{} {}", dt.format("%Y-%m-%d"), py_time_str(&dt.time()))
}

/// pandas' rendering of a timedelta (`0 days 01:30:00`).
fn py_timedelta_str(d: &Duration) -> String {
    let total_micros = d.num_microseconds().unwrap_or(0);
    let days = total_micros.div_euclid(86_400_000_000);
    let rem = total_micros.rem_euclid(86_400_000_000);
    let secs = rem / 1_000_000;
    let micros = rem % 1_000_000;
    let base = format!("{} days {:02}:{:02}:{:02}", days, secs / 3600, (secs / 60) % 60, secs % 60);
    if micros == 0 { base } else { format!("{base}.{micros:06}") }
}

/// `openpyxl.utils.datetime.from_excel` (Windows or Mac epoch).
fn from_excel(value: f64, is_1904: bool) -> PyValue {
    let epoch = if is_1904 {
        NaiveDate::from_ymd_opt(1904, 1, 1).expect("valid date")
    } else {
        NaiveDate::from_ymd_opt(1899, 12, 30).expect("valid date")
    }
    .and_time(NaiveTime::MIN);

    let mut day = value.div_euclid(1.0);
    let fraction = value.rem_euclid(1.0);
    let diff_ms = (fraction * 86_400.0 * 1000.0).round() as i64;
    let diff = Duration::milliseconds(diff_ms);
    if (0.0..1.0).contains(&value) && diff.num_days() == 0 {
        return PyValue::Time(NaiveTime::MIN + diff);
    }
    if 0.0 < value && value < 60.0 && !is_1904 {
        day += 1.0;
    }
    match epoch.checked_add_signed(Duration::days(day as i64)).and_then(|d| d.checked_add_signed(diff)) {
        Some(dt) => PyValue::DateTime(dt),
        None => PyValue::Float(value),
    }
}

fn excel_datetime_value(dt: &ExcelDateTime) -> PyValue {
    let value = dt.as_f64();
    if dt.is_duration() {
        // Round to millisecond precision.
        let ms = (value * 86_400_000.0).round() as i64;
        return PyValue::TimeDelta(Duration::milliseconds(ms));
    }
    // calamine keeps the workbook epoch private; a 1904-epoch value converts
    // differently from the same serial in the 1900 epoch.
    let is_1904 = dt.to_ymd_hms_milli() != ExcelDateTime::new(value, ExcelDateTimeType::DateTime, false).to_ymd_hms_milli();
    from_excel(value, is_1904)
}

/// pandas' `OpenpyxlReader._convert_cell`.
fn convert_cell(cell: &Data) -> PyValue {
    match cell {
        Data::Empty => PyValue::empty(),
        Data::Error(_) => PyValue::NaN,
        Data::Int(i) => PyValue::Int(*i),
        Data::Float(f) => {
            if f.is_finite() && f.fract() == 0.0 && f.abs() < 9.2e18 {
                PyValue::Int(*f as i64)
            } else {
                PyValue::Float(*f)
            }
        }
        Data::Bool(b) => PyValue::Bool(*b),
        Data::String(s) => PyValue::Str(s.clone()),
        Data::DateTime(dt) => excel_datetime_value(dt),
        Data::DateTimeIso(s) => NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f")
            .map(PyValue::DateTime)
            .or_else(|_| NaiveDate::parse_from_str(s, "%Y-%m-%d").map(|d| PyValue::DateTime(d.and_time(NaiveTime::MIN))))
            .or_else(|_| NaiveTime::parse_from_str(s, "%H:%M:%S%.f").map(PyValue::Time))
            .unwrap_or_else(|_| PyValue::Str(s.clone())),
        Data::DurationIso(s) => PyValue::Str(s.clone()),
    }
}

/// `OpenpyxlReader.get_sheet_data`: rows from row 1 / column 1, trailing
/// empty cells and rows trimmed, rows padded to the widest row.
fn sheet_rows(range: &calamine::Range<Data>) -> Vec<Vec<PyValue>> {
    let Some((start_row, start_col)) = range.start() else {
        return Vec::new();
    };
    let mut data: Vec<Vec<PyValue>> = Vec::new();
    for _ in 0..start_row {
        data.push(Vec::new());
    }
    for row in range.rows() {
        let mut converted: Vec<PyValue> = (0..start_col).map(|_| PyValue::empty()).collect();
        converted.extend(row.iter().map(convert_cell));
        // trim trailing empty elements
        while converted.last().is_some_and(PyValue::is_empty_str) {
            converted.pop();
        }
        data.push(converted);
    }

    // Trim trailing empty rows
    while data.last().is_some_and(|r| r.is_empty()) {
        data.pop();
    }

    // extend rows to max width
    if let Some(max_width) = data.iter().map(Vec::len).max() {
        for row in data.iter_mut() {
            while row.len() < max_width {
                row.push(PyValue::empty());
            }
        }
    }
    data
}

// -- DataFrame emulation -------------------------------------------------------------

/// The default pandas NA strings.
const NA_VALUES: &[&str] = &[
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA",
    "NULL", "NaN", "None", "n/a", "nan", "null",
];

const TRUE_VALUES: &[&str] = &["True", "TRUE", "true"];
const FALSE_VALUES: &[&str] = &["False", "FALSE", "false"];

/// A typed DataFrame column.
#[derive(Debug, Clone, PartialEq)]
enum Column {
    Int(Vec<i64>),
    Float(Vec<Option<f64>>),
    Bool(Vec<bool>),
    DateTime(Vec<Option<NaiveDateTime>>),
    /// Python objects; `None` is NaN.
    Object(Vec<Option<PyValue>>),
}

impl Column {
    fn len(&self) -> usize {
        match self {
            Column::Int(v) => v.len(),
            Column::Float(v) => v.len(),
            Column::Bool(v) => v.len(),
            Column::DateTime(v) => v.len(),
            Column::Object(v) => v.len(),
        }
    }
}

struct DataFrame {
    columns: Vec<String>,
    data: Vec<Column>,
}

fn is_na(value: &PyValue) -> bool {
    match value {
        PyValue::NaN => true,
        PyValue::Float(f) => f.is_nan(),
        PyValue::Str(s) => NA_VALUES.contains(&s.as_str()),
        _ => false,
    }
}

/// Parse a numeric string like `maybe_convert_numeric` (surrounding
/// whitespace allowed; ints stay ints).
fn parse_numeric_str(s: &str) -> Option<PyValue> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    if let Ok(i) = t.parse::<i64>() {
        return Some(PyValue::Int(i));
    }
    let lower = t.to_ascii_lowercase();
    let valid_float = match lower.trim_start_matches(['+', '-']) {
        "inf" | "infinity" => true,
        "nan" => true,
        body => body.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | '+' | '-')),
    };
    if !valid_float {
        return None;
    }
    t.parse::<f64>().ok().map(PyValue::Float)
}

/// Infer a column's type like pandas' Python parser (`_infer_types`) and the
/// DataFrame constructor.
fn infer_column(values: Vec<PyValue>) -> Column {
    // Datetime columns: every non-NA value is a datetime.
    let non_na: Vec<&PyValue> = values.iter().filter(|v| !is_na(v)).collect();
    if !non_na.is_empty() && non_na.iter().all(|v| matches!(v, PyValue::DateTime(_))) {
        return Column::DateTime(
            values
                .iter()
                .map(|v| match v {
                    PyValue::DateTime(dt) => Some(*dt),
                    _ => None,
                })
                .collect(),
        );
    }

    // `maybe_convert_numeric`.
    let mut numeric: Vec<Option<PyValue>> = Vec::with_capacity(values.len());
    let mut numeric_ok = true;
    let (mut seen_null, mut seen_float, mut seen_int, mut seen_bool) = (false, false, false, false);
    for value in &values {
        if is_na(value) {
            seen_null = true;
            numeric.push(None);
            continue;
        }
        let converted = match value {
            PyValue::Int(_) => Some(value.clone()),
            PyValue::Float(_) => Some(value.clone()),
            PyValue::Bool(_) => Some(value.clone()),
            PyValue::Str(s) => parse_numeric_str(s),
            _ => None,
        };
        match converted {
            Some(PyValue::Int(i)) => {
                seen_int = true;
                numeric.push(Some(PyValue::Int(i)));
            }
            Some(PyValue::Float(f)) => {
                seen_float = true;
                numeric.push(Some(PyValue::Float(f)));
            }
            Some(PyValue::Bool(b)) => {
                seen_bool = true;
                numeric.push(Some(PyValue::Bool(b)));
            }
            _ => {
                numeric_ok = false;
                break;
            }
        }
    }

    if numeric_ok {
        if seen_null || seen_float {
            return Column::Float(numeric.iter().map(|v| v.as_ref().and_then(PyValue::as_number)).collect());
        }
        if seen_bool && !seen_int {
            return Column::Bool(numeric.iter().map(|v| matches!(v, Some(PyValue::Bool(true)))).collect());
        }
        return Column::Int(
            numeric.iter().map(|v| v.as_ref().and_then(PyValue::as_number).unwrap_or(0.0) as i64).collect(),
        );
    }

    // Object column: `sanitize_objects` replaces NA values and makes equal
    // values share the first-seen object.
    let mut memo: Vec<PyValue> = Vec::new();
    let mut objects: Vec<Option<PyValue>> = Vec::with_capacity(values.len());
    for value in values {
        if is_na(&value) {
            objects.push(None);
        } else if let Some(seen) = memo.iter().find(|m| m.py_eq(&value)) {
            objects.push(Some(seen.clone()));
        } else {
            memo.push(value.clone());
            objects.push(Some(value));
        }
    }

    // `maybe_convert_bool`, unless the first value is an int (or bool).
    let first_is_int = matches!(objects.first(), Some(Some(PyValue::Int(_))) | Some(Some(PyValue::Bool(_))));
    if objects.is_empty() || !first_is_int {
        let mut bools: Vec<Option<bool>> = Vec::with_capacity(objects.len());
        let mut convertible = true;
        for value in &objects {
            match value {
                None => bools.push(None),
                Some(PyValue::Bool(b)) => bools.push(Some(*b)),
                Some(PyValue::Str(s)) if TRUE_VALUES.contains(&s.as_str()) => bools.push(Some(true)),
                Some(PyValue::Str(s)) if FALSE_VALUES.contains(&s.as_str()) => bools.push(Some(false)),
                _ => {
                    convertible = false;
                    break;
                }
            }
        }
        if convertible && !objects.is_empty() {
            if bools.iter().all(Option::is_some) {
                return Column::Bool(bools.into_iter().map(|b| b.unwrap_or(false)).collect());
            }
            return Column::Object(bools.into_iter().map(|b| b.map(PyValue::Bool)).collect());
        }
    }

    Column::Object(objects)
}

impl DataFrame {
    /// `TextParser(data, header=0)`: the first row is the header.
    fn from_rows(rows: Vec<Vec<PyValue>>) -> Self {
        let Some((header, body)) = rows.split_first() else {
            return DataFrame { columns: Vec::new(), data: Vec::new() };
        };

        // Name blank header cells "Unnamed: i".
        let mut names: Vec<PyValue> = Vec::with_capacity(header.len());
        let mut unnamed: Vec<usize> = Vec::new();
        for (i, c) in header.iter().enumerate() {
            if c.is_empty_str() {
                names.push(PyValue::Str(format!("Unnamed: {i}")));
                unnamed.push(i);
            } else {
                names.push(c.clone());
            }
        }

        // Mangle duplicates, regular columns before unnamed ones.
        let mut counts: Vec<(PyValue, usize)> = Vec::new();
        let get_count = |counts: &Vec<(PyValue, usize)>, key: &PyValue| {
            counts.iter().find(|(k, _)| k.py_eq(key)).map(|(_, c)| *c).unwrap_or(0)
        };
        let set_count = |counts: &mut Vec<(PyValue, usize)>, key: &PyValue, value: usize| {
            if let Some(entry) = counts.iter_mut().find(|(k, _)| k.py_eq(key)) {
                entry.1 = value;
            } else {
                counts.push((key.clone(), value));
            }
        };
        let loop_order: Vec<usize> =
            (0..names.len()).filter(|i| !unnamed.contains(i)).chain(unnamed.iter().copied()).collect();
        for i in loop_order {
            let old_col = names[i].clone();
            let mut col = old_col.clone();
            let mut cur_count = get_count(&counts, &col);
            if cur_count > 0 {
                while cur_count > 0 {
                    set_count(&mut counts, &old_col, cur_count + 1);
                    col = PyValue::Str(format!("{}.{}", old_col.py_str(), cur_count));
                    if names.iter().any(|n| n.py_eq(&col)) {
                        cur_count += 1;
                    } else {
                        cur_count = get_count(&counts, &col);
                    }
                }
            }
            names[i] = col.clone();
            set_count(&mut counts, &col, cur_count + 1);
        }

        let columns: Vec<String> = names.iter().map(PyValue::py_str).collect();
        let data = (0..columns.len())
            .map(|c| infer_column(body.iter().map(|row| row.get(c).cloned().unwrap_or_else(PyValue::empty)).collect()))
            .collect();
        DataFrame { columns, data }
    }

    /// `DataFrame.to_html(index=False)`.
    fn to_html(&self) -> String {
        let mut html = String::from("<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n");
        for name in &self.columns {
            html.push_str(&format!("      <th>{}</th>\n", html_cell(name)));
        }
        html.push_str("    </tr>\n  </thead>\n  <tbody>\n");

        let formatted: Vec<Vec<String>> = self.data.iter().map(format_column).collect();
        let n_rows = self.data.first().map(Column::len).unwrap_or(0);
        for r in 0..n_rows {
            html.push_str("    <tr>\n");
            for column in &formatted {
                html.push_str(&format!("      <td>{}</td>\n", html_cell(&column[r])));
            }
            html.push_str("    </tr>\n");
        }
        html.push_str("  </tbody>\n</table>");
        html
    }
}

/// `pprint_thing(s, escape_chars=...)` + HTML escaping + `.strip()`.
fn html_cell(s: &str) -> String {
    let escaped = s
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    escaped.trim().to_string()
}

/// `_trim_zeros_single_float`.
fn trim_zeros_single_float(s: &str) -> String {
    let trimmed = s.trim_end_matches('0');
    if trimmed.ends_with('.') { format!("{trimmed}0") } else { trimmed.to_string() }
}

fn is_number_with_decimal(s: &str) -> bool {
    let t = s.trim_start();
    let t = t.strip_prefix(['+', '-']).unwrap_or(t);
    match t.split_once('.') {
        Some((int, frac)) => !int.is_empty() && int.chars().all(|c| c.is_ascii_digit()) && frac.chars().all(|c| c.is_ascii_digit()),
        None => false,
    }
}

/// `_trim_zeros_float`: trim trailing zeros equally from all decimal numbers.
fn trim_zeros_float(values: Vec<String>) -> Vec<String> {
    let mut trimmed = values;
    loop {
        let numbers: Vec<&String> = trimmed.iter().filter(|x| is_number_with_decimal(x)).collect();
        if numbers.is_empty() || !numbers.iter().all(|x| x.ends_with('0')) {
            break;
        }
        trimmed = trimmed
            .into_iter()
            .map(|x| if is_number_with_decimal(&x) { x[..x.len() - 1].to_string() } else { x })
            .collect();
    }
    trimmed.into_iter().map(|x| if is_number_with_decimal(&x) && x.ends_with('.') { format!("{x}0") } else { x }).collect()
}

/// Python `f"{value: .6f}"`.
fn fixed6(v: f64) -> String {
    if v.is_infinite() {
        return if v > 0.0 { " inf".to_string() } else { "-inf".to_string() };
    }
    let s = format!("{v:.6}");
    if s.starts_with('-') { s } else { format!(" {s}") }
}

/// Python `f"{value: .6e}"`.
fn sci6(v: f64) -> String {
    if v.is_infinite() {
        return if v > 0.0 { " inf".to_string() } else { "-inf".to_string() };
    }
    let s = format!("{v:.6e}");
    let (mantissa, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let body = format!("{mantissa}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    if body.starts_with('-') { body } else { format!(" {body}") }
}

/// pandas' `FloatArrayFormatter` (precision 6, fixed width).
fn format_floats(values: &[Option<f64>]) -> Vec<String> {
    let format_with = |f: &dyn Fn(f64) -> String| -> Vec<String> {
        trim_zeros_float(values.iter().map(|v| v.map(f).unwrap_or_else(|| "NaN".to_string())).collect())
    };
    let formatted = format_with(&fixed6);
    let maxlen = formatted.iter().map(|s| s.chars().count()).max().unwrap_or(0);
    let too_long = maxlen > 6 + 6;
    let abs_vals: Vec<f64> = values.iter().flatten().map(|v| v.abs()).collect();
    let has_large_values = abs_vals.iter().any(|v| *v > 1e6);
    let has_small_values = abs_vals.iter().any(|v| *v < 1e-6 && *v > 0.0);
    if has_small_values || (too_long && has_large_values) {
        return format_with(&sci6);
    }
    formatted
}

/// pandas' datetime column formatting: date only when every value is at
/// midnight, otherwise with the column-wide sub-second precision.
fn format_datetimes(values: &[Option<NaiveDateTime>]) -> Vec<String> {
    let present: Vec<&NaiveDateTime> = values.iter().flatten().collect();
    let dates_only = present.iter().all(|dt| dt.time() == NaiveTime::MIN);
    let show_us = present.iter().any(|dt| (dt.nanosecond() / 1000) % 1000 != 0);
    let show_ms = present.iter().any(|dt| dt.nanosecond() / 1000 != 0);
    values
        .iter()
        .map(|v| match v {
            None => "NaT".to_string(),
            Some(dt) if dates_only => dt.format("%Y-%m-%d").to_string(),
            Some(dt) => {
                let base = dt.format("%Y-%m-%d %H:%M:%S").to_string();
                let micros = dt.nanosecond() / 1000;
                if show_us {
                    format!("{base}.{micros:06}")
                } else if show_ms {
                    format!("{base}.{:03}", micros / 1000)
                } else {
                    base
                }
            }
        })
        .collect()
}

fn format_column(column: &Column) -> Vec<String> {
    match column {
        Column::Int(values) => values.iter().map(i64::to_string).collect(),
        Column::Float(values) => format_floats(values),
        Column::Bool(values) => values.iter().map(|b| if *b { "True" } else { "False" }.to_string()).collect(),
        Column::DateTime(values) => format_datetimes(values),
        Column::Object(values) => values
            .iter()
            .map(|v| match v {
                None => "NaN".to_string(),
                Some(PyValue::Float(f)) if f.is_nan() => "NaN".to_string(),
                Some(PyValue::Float(f)) => trim_zeros_single_float(fixed6(*f).trim()),
                Some(other) => other.py_str(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> PyValue {
        PyValue::Str(v.to_string())
    }

    #[test]
    fn float_formatting_matches_pandas() {
        assert_eq!(format_floats(&[Some(1.0), None, Some(3.0)]), vec![" 1.0", "NaN", " 3.0"]);
        assert_eq!(format_floats(&[None, Some(2.25), Some(4.0)]), vec!["NaN", " 2.25", " 4.00"]);
        assert_eq!(format_floats(&[Some(0.1), Some(1e-7)]), vec![" 1.000000e-01", " 1.000000e-07"]);
        assert_eq!(format_floats(&[Some(1234567890123.25), Some(1.0)]), vec![" 1.234568e+12", " 1.000000e+00"]);
        assert_eq!(format_floats(&[Some(999999.5), Some(1000000.5)]), vec![" 999999.5", " 1000000.5"]);
        assert_eq!(format_floats(&[Some(1.5e-5), Some(2.0)]), vec![" 0.000015", " 2.000000"]);
    }

    #[test]
    fn column_inference_matches_pandas() {
        assert_eq!(infer_column(vec![s("123"), s("4.5")]), Column::Float(vec![Some(123.0), Some(4.5)]));
        assert_eq!(infer_column(vec![PyValue::Int(1), PyValue::Bool(true)]), Column::Int(vec![1, 1]));
        assert_eq!(infer_column(vec![s("TRUE"), s("False")]), Column::Bool(vec![true, false]));
        assert_eq!(
            infer_column(vec![PyValue::Int(1), s("a"), PyValue::Bool(true)]),
            Column::Object(vec![Some(PyValue::Int(1)), Some(s("a")), Some(PyValue::Int(1))])
        );
        assert_eq!(
            infer_column(vec![s("True"), s(""), s("false")]),
            Column::Object(vec![Some(PyValue::Bool(true)), None, Some(PyValue::Bool(false))])
        );
    }

    #[test]
    fn header_mangling_matches_pandas() {
        let frame = DataFrame::from_rows(vec![
            vec![s("a"), s("a"), s("a"), s("a.1"), s("")],
            vec![PyValue::Int(1), PyValue::Int(2), PyValue::Int(3), PyValue::Int(4), PyValue::Int(5)],
        ]);
        assert_eq!(frame.columns, vec!["a", "a.2", "a.3", "a.1", "Unnamed: 4"]);
    }
}
