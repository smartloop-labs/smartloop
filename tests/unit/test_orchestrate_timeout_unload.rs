//! Port of `tests/unit/test_orchestrate_timeout_unload.py`.
//!
//! The orchestration timeout must not free models still being decoded on.
//!
//! When the 180s budget expires the router signals `cancelled` and stops
//! waiting, but it deliberately does not await the worker — a stuck native
//! llama.cpp call would block forever.  So the orchestration thread is still
//! running, very likely inside that native call.  Closing its llama.cpp
//! context there is a use-after-free that segfaults the whole server, which is
//! what `make run-debug` hit right after "[mem] Evicting prior-step capability
//! model qwen3.5-2b".  The unload has to wait for the thread to unwind.

use std::sync::Arc;
use std::time::Duration;

use smartloop::api::routers::chat::{unload_when_idle, ChatService};

use crate::shared::api_chat::{Event, FakeChatService};

#[tokio::test]
async fn test_models_are_not_freed_while_the_thread_is_running() {
    let service = Arc::new(FakeChatService::default());
    let still_working = Event::new();

    // A future that never resolves while the sender is held.
    let (_pending, future) = tokio::sync::oneshot::channel::<()>();
    unload_when_idle(future, service.clone() as Arc<dyn ChatService>);

    // The orchestration thread has not unwound yet.
    tokio::task::yield_now().await;
    assert_eq!(service.unload_call_count(), 0);
    assert!(!still_working.is_set());
}

#[tokio::test]
async fn test_models_are_freed_once_the_thread_unwinds() {
    let service = Arc::new(FakeChatService::default());

    let (done, future) = tokio::sync::oneshot::channel::<(&str, Option<()>, Option<()>, Vec<()>, Option<()>, bool)>();
    unload_when_idle(future, service.clone() as Arc<dyn ChatService>);

    done.send(("context", None, None, Vec::new(), None, false)).unwrap();
    // One hop for the completion, one for the worker it hands off to.
    for _ in 0..10 {
        tokio::time::sleep(Duration::from_millis(10)).await;
        if service.unload_call_count() > 0 {
            break;
        }
    }

    assert_eq!(service.unload_call_count(), 1);
}

#[tokio::test]
async fn test_an_unload_failure_is_swallowed() {
    // Cleanup runs after the client already got its timeout response; a
    // failure here must not surface anywhere.
    let service = Arc::new(FakeChatService::default());
    *service.device.unload_failure.lock() = Some("GPU busy".to_string());

    let (done, future) = tokio::sync::oneshot::channel::<Option<()>>();
    unload_when_idle(future, service.clone() as Arc<dyn ChatService>);
    done.send(None).unwrap();

    for _ in 0..10 {
        tokio::time::sleep(Duration::from_millis(10)).await;
        if service.unload_call_count() > 0 {
            break;
        }
    }
    assert!(service.unload_call_count() > 0);
}
