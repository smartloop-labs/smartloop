//! Resume state for chunked downloads: the interval set of written bytes.

use smartloop::model::downloader::resume::{IntervalSet, ResumeState, state_path};

/// `IntervalSet`
mod test_interval_set {
    use super::*;

    #[test]
    fn test_touching_and_overlapping_spans_merge() {
        let mut set = IntervalSet::new();
        set.insert(0, 10);
        set.insert(10, 20);
        set.insert(30, 40);
        set.insert(35, 50);
        assert_eq!(set.spans(), vec![(0, 20), (30, 50)]);
        set.insert(15, 32);
        assert_eq!(set.spans(), vec![(0, 50)]);
    }

    #[test]
    fn test_a_span_inside_another_changes_nothing() {
        let mut set = IntervalSet::from_spans([(0, 100)]);
        set.insert(10, 20);
        assert_eq!(set.spans(), vec![(0, 100)]);
    }

    #[test]
    fn test_a_span_covering_several_swallows_them() {
        let mut set = IntervalSet::from_spans([(10, 20), (30, 40), (50, 60)]);
        set.insert(5, 55);
        assert_eq!(set.spans(), vec![(5, 60)]);
    }

    #[test]
    fn test_empty_spans_are_ignored() {
        let mut set = IntervalSet::new();
        set.insert(5, 5);
        set.insert(9, 3);
        assert!(set.is_empty());
    }

    #[test]
    fn test_missing_lists_the_gaps_up_to_the_total() {
        let set = IntervalSet::from_spans([(10, 20), (30, 40)]);
        assert_eq!(set.missing(50), vec![(0, 10), (20, 30), (40, 50)]);
        assert_eq!(set.missing(35), vec![(0, 10), (20, 30)]);
        assert_eq!(IntervalSet::new().missing(8), vec![(0, 8)]);
    }

    #[test]
    fn test_covers_and_length() {
        let set = IntervalSet::from_spans([(0, 64), (64, 100)]);
        assert!(set.covers(100));
        assert!(!set.covers(101));
        assert_eq!(set.len_bytes(), 100);
        assert!(set.missing(100).is_empty());
    }
}

/// `ResumeState`, the `.part.state` sidecar.
mod test_resume_state {
    use super::*;

    const ETAG: &str = "\"0123456789abcdef0123456789abcdef-2213\"";

    #[test]
    fn test_it_round_trips_next_to_the_part() {
        let tmp = tempfile::tempdir().unwrap();
        let part = tmp.path().join("w.gguf.part");
        let mut state = ResumeState::new(100, Some(ETAG.to_string()), None);
        state.set_done(&IntervalSet::from_spans([(0, 10), (20, 30)]));
        state.save(&part).unwrap();

        assert_eq!(state_path(&part), tmp.path().join("w.gguf.part.state"));
        let loaded = ResumeState::load(&part).unwrap();
        assert_eq!(loaded, state);
        assert_eq!(loaded.done_set().spans(), vec![(0, 10), (20, 30)]);
        assert!(!tmp.path().join("w.gguf.part.state.tmp").exists());
    }

    #[test]
    fn test_a_corrupt_or_foreign_sidecar_is_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        let part = tmp.path().join("w.gguf.part");
        std::fs::write(state_path(&part), b"{not json").unwrap();
        assert!(ResumeState::load(&part).is_none());

        let mut state = ResumeState::new(100, Some(ETAG.to_string()), None);
        state.version = 99;
        std::fs::write(state_path(&part), serde_json::to_vec(&state).unwrap()).unwrap();
        assert!(ResumeState::load(&part).is_none());
    }

    #[test]
    fn test_remove_tolerates_a_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        ResumeState::remove(&tmp.path().join("w.gguf.part")).unwrap();
    }

    #[test]
    fn test_the_etag_decides_whether_it_is_the_same_object() {
        let state = ResumeState::new(100, Some(ETAG.to_string()), Some("Mon".to_string()));
        assert!(state.same_object(100, Some(ETAG), Some("Tue")));
        assert!(!state.same_object(100, Some("\"other\""), Some("Mon")));
        assert!(!state.same_object(101, Some(ETAG), None));
        assert!(!state.same_object(100, None, Some("Mon")));
    }

    #[test]
    fn test_last_modified_stands_in_without_etags() {
        let state = ResumeState::new(100, None, Some("Mon".to_string()));
        assert!(state.same_object(100, None, Some("Mon")));
        assert!(!state.same_object(100, None, Some("Tue")));
        assert!(!state.same_object(100, None, None));
    }

    #[test]
    fn test_an_object_without_a_strong_validator_is_not_resumable() {
        assert!(!ResumeState::new(100, None, None).resumable());
        assert!(!ResumeState::new(100, Some("W/\"x\"".to_string()), None).resumable());
        assert!(!ResumeState::new(0, Some(ETAG.to_string()), None).resumable());
        let weak = ResumeState::new(100, Some("W/\"x\"".to_string()), None);
        assert!(!weak.same_object(100, Some("W/\"x\""), None));
    }
}
