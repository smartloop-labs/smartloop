//! A turn on a project bound to the user's folder waits for the folder's
//! index before it answers, and says what it is reading meanwhile; a project
//! with no folder answers straight away.

use std::time::Duration;

use serial_test::serial;

use crate::test_stream_opening_buffer::{request, run, service, yields};

/// The `chat.status` frames of step `index`, as `(status, message)`.
fn index_statuses(raw: &[String]) -> Vec<(String, String)> {
    raw.iter()
        .flat_map(|frame| frame.lines())
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
        .filter(|event| event["object"] == "chat.status" && event["step"] == "index")
        .map(|event| (event["status"].as_str().unwrap().to_string(), event["message"].as_str().unwrap().to_string()))
        .collect()
}

#[tokio::test]
#[serial]
async fn test_a_bound_project_reads_its_folder_before_answering() {
    let tmp = tempfile::tempdir().unwrap();
    let svc = service(|_, _| yields(&["Class A shares vote."]));
    *svc.work_dir.lock() = Some(tmp.path().to_string_lossy().into_owned());
    *svc.index_activity.lock() =
        vec!["Reading Fund supplement.pdf".to_string(), "Indexing Fund supplement.pdf".to_string()];
    *svc.index_hold.lock() = Duration::from_millis(2400);

    let raw = run(request(), &svc).await;

    assert_eq!(*svc.index_calls.lock(), 1);
    let statuses = index_statuses(&raw);
    assert!(statuses[0].1.starts_with("Checking documents in"));
    let lines: Vec<&str> = statuses.iter().map(|(_, message)| message.as_str()).collect();
    let reading = lines.iter().position(|m| *m == "Reading Fund supplement.pdf").expect("a Reading line");
    let indexing = lines.iter().position(|m| *m == "Indexing Fund supplement.pdf").expect("an Indexing line");
    assert!(reading < indexing);
    assert_eq!(statuses.last().unwrap(), &("completed".to_string(), "Documents indexed".to_string()));
}

#[tokio::test]
#[serial]
async fn test_a_project_without_a_folder_does_not_wait() {
    let svc = service(|_, _| yields(&["Hello."]));

    let raw = run(request(), &svc).await;

    assert_eq!(*svc.index_calls.lock(), 0);
    assert!(index_statuses(&raw).is_empty());
}
