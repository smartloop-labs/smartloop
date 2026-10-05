//! Python formatting compatibility helpers.
//!
//! Several utilities render JSON values exactly the way the Python app did
//! (`str(value)`, `repr(value)`, `json.dumps(...)`), and those strings end up
//! in prompts, metadata columns and files that must stay byte-compatible.
//! `serde_json`'s own formatting differs in a few visible ways (float repr,
//! separators, `ensure_ascii`), so the Python behaviour is reproduced here.

use serde_json::Value;

/// Python `repr(float)` / `str(float)`: shortest round-tripping digits,
/// positional for exponents in `[-4, 16)`, scientific (`1e+16`, `1.5e-05`)
/// otherwise, and always a `.0` on integral positional values.
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".to_string() } else { "0.0".to_string() };
    }
    // `{:e}` yields the shortest round-tripping digits, e.g. "1.2345e-5".
    let sci = format!("{:e}", f);
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp.parse().expect("valid exponent");
    let negative = mantissa.starts_with('-');
    let mantissa = mantissa.trim_start_matches('-');
    let sign = if negative { "-" } else { "" };

    if !(-4..16).contains(&exp) {
        let exp_sign = if exp < 0 { '-' } else { '+' };
        return format!("{sign}{mantissa}e{exp_sign}{:02}", exp.abs());
    }

    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let n = digits.len() as i32;
    if exp >= 0 {
        let int_len = exp + 1;
        if n <= int_len {
            let zeros = "0".repeat((int_len - n) as usize);
            format!("{sign}{digits}{zeros}.0")
        } else {
            let (int_part, frac_part) = digits.split_at(int_len as usize);
            format!("{sign}{int_part}.{frac_part}")
        }
    } else {
        let zeros = "0".repeat((-exp - 1) as usize);
        format!("{sign}0.{zeros}{digits}")
    }
}

fn number_str(n: &serde_json::Number) -> String {
    if n.is_f64() {
        py_float_repr(n.as_f64().unwrap_or(0.0))
    } else {
        n.to_string()
    }
}

/// Python `str(value)` for a JSON-decoded value (`None`, `True`, `3.0`, raw
/// strings, and `repr`-style lists / dicts).
pub fn py_str(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Number(n) => number_str(n),
        Value::String(s) => s.clone(),
        Value::Array(_) | Value::Object(_) => py_repr(value),
    }
}

/// Python `repr(str)`: single quotes unless the text contains a single quote
/// and no double quote.
pub fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `repr(value)` for a JSON-decoded value.
pub fn py_repr(value: &Value) -> String {
    match value {
        Value::String(s) => py_repr_str(s),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> =
                map.iter().map(|(k, v)| format!("{}: {}", py_repr_str(k), py_repr(v))).collect();
            format!("{{{}}}", inner.join(", "))
        }
        other => py_str(other),
    }
}

fn push_json_string(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && !(' '..='~').contains(&c) => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Options mirroring `json.dumps` keyword arguments.
#[derive(Debug, Clone, Copy)]
pub struct DumpsOptions<'a> {
    pub indent: Option<usize>,
    pub item_separator: &'a str,
    pub key_separator: &'a str,
    pub ensure_ascii: bool,
}

impl Default for DumpsOptions<'_> {
    /// `json.dumps(value)` defaults: `", "` / `": "`, `ensure_ascii=True`.
    fn default() -> Self {
        Self { indent: None, item_separator: ", ", key_separator: ": ", ensure_ascii: true }
    }
}

fn dumps_into(out: &mut String, value: &Value, opts: &DumpsOptions<'_>, level: usize) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&number_str(n)),
        Value::String(s) => push_json_string(out, s, opts.ensure_ascii),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(opts.item_separator);
                }
                if let Some(indent) = opts.indent {
                    out.push('\n');
                    out.push_str(&" ".repeat(indent * (level + 1)));
                }
                dumps_into(out, item, opts, level + 1);
            }
            if let Some(indent) = opts.indent {
                out.push('\n');
                out.push_str(&" ".repeat(indent * level));
            }
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(opts.item_separator);
                }
                if let Some(indent) = opts.indent {
                    out.push('\n');
                    out.push_str(&" ".repeat(indent * (level + 1)));
                }
                push_json_string(out, k, opts.ensure_ascii);
                out.push_str(opts.key_separator);
                dumps_into(out, v, opts, level + 1);
            }
            if let Some(indent) = opts.indent {
                out.push('\n');
                out.push_str(&" ".repeat(indent * level));
            }
            out.push('}');
        }
    }
}

/// `json.dumps(value, ...)` with explicit options.
pub fn py_json_dumps_with(value: &Value, opts: DumpsOptions<'_>) -> String {
    let mut out = String::new();
    dumps_into(&mut out, value, &opts, 0);
    out
}

/// `json.dumps(value)` with Python's defaults.
pub fn py_json_dumps(value: &Value) -> String {
    py_json_dumps_with(value, DumpsOptions::default())
}

/// `json.dump(value, f, indent=indent)` formatting (`","` / `": "`,
/// `ensure_ascii=True`).
pub fn py_json_dumps_indent(value: &Value, indent: usize) -> String {
    py_json_dumps_with(
        value,
        DumpsOptions { indent: Some(indent), item_separator: ",", key_separator: ": ", ensure_ascii: true },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_repr_matches_python() {
        assert_eq!(py_float_repr(3.0), "3.0");
        assert_eq!(py_float_repr(290.64), "290.64");
        assert_eq!(py_float_repr(0.0001), "0.0001");
        assert_eq!(py_float_repr(0.00001), "1e-05");
        assert_eq!(py_float_repr(1.5e-5), "1.5e-05");
        assert_eq!(py_float_repr(1e16), "1e+16");
        assert_eq!(py_float_repr(1234567890123456.0), "1234567890123456.0");
        assert_eq!(py_float_repr(-2.5), "-2.5");
        assert_eq!(py_float_repr(0.1), "0.1");
    }

    #[test]
    fn dumps_matches_python_defaults() {
        let v = json!({"a": [1, 2.0, "é"], "b": null, "c": {}});
        let expected = String::from(r#"{"a": [1, 2.0, ""#) + "\\" + r#"u00e9"], "b": null, "c": {}}"#;
        assert_eq!(py_json_dumps(&v), expected);
        assert_eq!(
            py_json_dumps_indent(&json!({"k": {"x": 1}}), 2),
            "{\n  \"k\": {\n    \"x\": 1\n  }\n}"
        );
    }

    #[test]
    fn repr_matches_python() {
        let v = json!({"a": "it's", "b": true, "c": null, "d": [1, "x"]});
        assert_eq!(py_repr(&v), r#"{'a': "it's", 'b': True, 'c': None, 'd': [1, 'x']}"#);
    }
}
