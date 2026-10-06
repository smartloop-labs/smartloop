//! Port of `tests/unit/test_doc_splitter.py`.

use serde_json::json;
use smartloop::utils::doc_splitter::{is_markdown_table, json_to_table};

mod test_json_to_table {
    use super::*;

    #[test]
    fn test_converts_array_of_objects() {
        let data = json!([
            {"name": "Drake Maye", "team": "NE", "td": 31},
            {"name": "Jordan Love", "team": "GB", "td": 23},
        ]);
        let table = json_to_table(&data.to_string());
        assert!(table.is_some());
        let table = table.unwrap();
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines[0], "| name | team | td |");
        assert_eq!(lines[1], "|---|---|---|");
        assert!(lines.contains(&"| Drake Maye | NE | 31 |"));
        assert!(lines.contains(&"| Jordan Love | GB | 23 |"));
    }

    #[test]
    fn test_accepts_rows_key() {
        let table = json_to_table(&json!({"category": "EPA/play", "rows": [{"name": "A", "v": 1}]}).to_string());
        assert!(table.is_some());
        let table = table.unwrap();
        assert!(table.contains("| name | v |"));
        assert!(table.contains("| A | 1 |"));
    }

    #[test]
    fn test_flattens_nested_objects_with_dot_keys() {
        let table = json_to_table(
            &json!({
                "rows": [{"name": "A", "values": {"epa": 0.296484, "td": 31}}]
            })
            .to_string(),
        )
        .unwrap();
        assert!(table.contains("| name | values.epa | values.td |"));
        assert!(table.contains("| A | 0.2965 | 31 |"));
    }

    #[test]
    fn test_rounds_floats_and_drops_integral_decimal() {
        let table = json_to_table(&json!([{"a": 0.1234567, "b": 3.0}]).to_string()).unwrap();
        assert!(table.contains("| 0.1235 | 3 |"));
    }

    #[test]
    fn test_returns_none_for_non_tabular_json() {
        assert!(json_to_table(r#"{"hello": "world"}"#).is_none());
        assert!(json_to_table("[1, 2, 3]").is_none());
        assert!(json_to_table("[[1, 2], [3, 4]]").is_none());
    }

    #[test]
    fn test_returns_none_for_non_json() {
        assert!(json_to_table("plain prose that is not json").is_none());
        assert!(json_to_table("").is_none());
    }

    #[test]
    fn test_caps_rows() {
        let rows: Vec<_> = (0..501).map(|i| json!({"i": i})).collect();
        let table = json_to_table(&serde_json::Value::Array(rows).to_string());
        assert!(table.is_some());
        assert!(table.unwrap().contains("(500 of 501 rows shown)"));
    }
}

mod test_is_markdown_table {
    use super::*;

    #[test]
    fn test_detects_separator_row() {
        assert!(is_markdown_table("| a | b |\n|---|---|\n| 1 | 2 |"));
    }

    #[test]
    fn test_false_for_prose() {
        assert!(!is_markdown_table("the quick brown fox jumps over the lazy dog"));
    }

    #[test]
    fn test_false_for_data_row_without_separator() {
        assert!(!is_markdown_table("| a | b |\n| 1 | 2 |"));
    }

    #[test]
    fn test_false_for_empty() {
        assert!(!is_markdown_table(""));
    }
}
