//! Resume state for chunked downloads: the interval set of written bytes.

use smartloop::model::downloader::resume::IntervalSet;

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
