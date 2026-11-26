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

/// A chunked download keeps what it wrote and picks up from there.
mod test_chunked_resume {
    use std::path::Path;
    use std::sync::Arc;

    use md5::{Digest, Md5};
    use smartloop::model::downloader::artifact_downloader::{RemoteFileInfo, SLPArtifactDownloader};
    use smartloop::model::downloader::digest::RemoteDigest;

    use super::*;
    use crate::shared::model_fakes::{RangeServer, test_logger};

    fn data(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn downloader(server: &Arc<RangeServer>, workers: usize) -> SLPArtifactDownloader {
        let dl = SLPArtifactDownloader::for_tests(None, test_logger(), None).with_http(Arc::clone(server) as _);
        dl.cancel_token().set_instant_wait(true);
        dl.set_connections(Some(workers));
        dl
    }

    fn fetch(server: &Arc<RangeServer>, part: &Path) -> Result<(), smartloop::model::downloader::artifact_downloader::DownloadError> {
        downloader(server, 1).download_file_chunked("w.gguf", "https://x/w", part, &server.info(), None)
    }

    #[test]
    fn test_a_failed_download_resumes_from_what_arrived() {
        let tmp = tempfile::tempdir().unwrap();
        let part = tmp.path().join("w.gguf.part");
        let server = RangeServer::new(data(40));
        *server.cut_at.lock() = Some(24);

        assert!(fetch(&server, &part).is_err());
        let saved = ResumeState::load(&part).expect("progress was saved");
        assert_eq!(saved.done, vec![[0, 24]]);

        // A new downloader, as after a restart.
        *server.cut_at.lock() = None;
        server.requests.lock().clear();
        fetch(&server, &part).unwrap();

        assert_eq!(std::fs::read(&part).unwrap(), server.data);
        assert_eq!(server.ranges(), vec!["bytes=24-39".to_string()]);
    }

    #[test]
    fn test_a_changed_object_starts_over() {
        let tmp = tempfile::tempdir().unwrap();
        let part = tmp.path().join("w.gguf.part");
        let server = RangeServer::new(data(40));
        *server.cut_at.lock() = Some(24);
        assert!(fetch(&server, &part).is_err());

        *server.cut_at.lock() = None;
        *server.etag.lock() = "\"v2\"".to_string();
        server.requests.lock().clear();
        fetch(&server, &part).unwrap();

        assert_eq!(server.ranges(), vec!["bytes=0-39".to_string()]);
        assert_eq!(std::fs::read(&part).unwrap(), server.data);
    }

    #[test]
    fn test_a_part_of_the_wrong_length_starts_over() {
        let tmp = tempfile::tempdir().unwrap();
        let part = tmp.path().join("w.gguf.part");
        let server = RangeServer::new(data(40));
        *server.cut_at.lock() = Some(24);
        assert!(fetch(&server, &part).is_err());

        std::fs::OpenOptions::new().write(true).open(&part).unwrap().set_len(30).unwrap();
        *server.cut_at.lock() = None;
        server.requests.lock().clear();
        fetch(&server, &part).unwrap();
        assert_eq!(server.ranges(), vec!["bytes=0-39".to_string()]);
    }

    #[test]
    fn test_an_object_without_validators_keeps_no_journal() {
        let tmp = tempfile::tempdir().unwrap();
        let part = tmp.path().join("w.gguf.part");
        let server = RangeServer::new(data(40));
        *server.cut_at.lock() = Some(24);
        let info = RemoteFileInfo { etag: None, ..server.info() };
        assert!(downloader(&server, 1).download_file_chunked("w.gguf", "https://x/w", &part, &info, None).is_err());
        assert!(!state_path(&part).exists());
    }

    #[test]
    fn test_parallel_ranges_resume_only_the_gaps() {
        let tmp = tempfile::tempdir().unwrap();
        let part = tmp.path().join("w.gguf.part");
        let server = RangeServer::new(data(64));
        let mut state = ResumeState::new(64, Some("\"v1\"".to_string()), None);
        state.set_done(&IntervalSet::from_spans([(0, 16), (32, 48)]));
        std::fs::write(&part, {
            let mut bytes = vec![0u8; 64];
            bytes[0..16].copy_from_slice(&server.data[0..16]);
            bytes[32..48].copy_from_slice(&server.data[32..48]);
            bytes
        })
        .unwrap();
        state.save(&part).unwrap();

        downloader(&server, 4).download_file_chunked("w.gguf", "https://x/w", &part, &server.info(), None).unwrap();

        assert_eq!(std::fs::read(&part).unwrap(), server.data);
        let mut fetched: Vec<(u64, u64)> = server
            .ranges()
            .iter()
            .map(|r| {
                let (a, b) = r.strip_prefix("bytes=").unwrap().split_once('-').unwrap();
                (a.parse().unwrap(), b.parse().unwrap())
            })
            .collect();
        fetched.sort();
        let covered = IntervalSet::from_spans(fetched.iter().map(|(a, b)| (*a, b + 1)));
        assert_eq!(covered.spans(), vec![(16, 32), (48, 64)]);
    }

    fn publish(server: &Arc<RangeServer>, dir: &Path, md5: Option<String>) -> Result<String, smartloop::model::downloader::artifact_downloader::DownloadError> {
        let dl = downloader(server, 2);
        dl.set_chunked_threshold(Some(1));
        let info = RemoteFileInfo { digest: md5.map(RemoteDigest::Md5), ..server.info() };
        dl.download_file("w.gguf", "https://x/w", dir, &info, None)
    }

    #[test]
    fn test_a_published_file_leaves_no_journal() {
        let tmp = tempfile::tempdir().unwrap();
        let server = RangeServer::new(data(40));
        *server.cut_at.lock() = Some(24);
        assert!(publish(&server, tmp.path(), None).is_err());
        assert!(state_path(&tmp.path().join("w.gguf.part")).exists());

        *server.cut_at.lock() = None;
        publish(&server, tmp.path(), Some(hex::encode(Md5::digest(&server.data)))).unwrap();
        assert_eq!(std::fs::read(tmp.path().join("w.gguf")).unwrap(), server.data);
        assert!(!tmp.path().join("w.gguf.part").exists());
        assert!(!state_path(&tmp.path().join("w.gguf.part")).exists());
    }

    #[test]
    fn test_a_resumed_file_that_fails_verification_restarts_clean() {
        let tmp = tempfile::tempdir().unwrap();
        let part = tmp.path().join("w.gguf.part");
        let server = RangeServer::new(data(40));
        // A journal claiming bytes the .part doesn't actually hold.
        let mut state = ResumeState::new(40, Some("\"v1\"".to_string()), None);
        state.set_done(&IntervalSet::from_spans([(0, 24)]));
        std::fs::write(&part, vec![0xEE; 40]).unwrap();
        state.save(&part).unwrap();

        publish(&server, tmp.path(), Some(hex::encode(Md5::digest(&server.data)))).unwrap();

        assert_eq!(std::fs::read(tmp.path().join("w.gguf")).unwrap(), server.data);
        let ranges = server.ranges();
        assert!(ranges.iter().all(|r| r.as_str() != "bytes=0-23"), "first pass fetched only the gap");
        assert!(ranges.iter().any(|r| r.starts_with("bytes=0-")), "second pass started over");
    }

    #[test]
    fn test_every_range_carries_if_range() {
        let tmp = tempfile::tempdir().unwrap();
        let server = RangeServer::new(data(40));
        downloader(&server, 2)
            .download_file_chunked("w.gguf", "https://x/w", &tmp.path().join("w.gguf.part"), &server.info(), None)
            .unwrap();
        let requests = server.requests.lock();
        assert!(!requests.is_empty());
        assert!(requests.iter().all(|(_, if_range)| if_range.as_deref() == Some("\"v1\"")));
    }

    #[test]
    fn test_a_file_changed_mid_download_is_fetched_again_from_the_start() {
        let tmp = tempfile::tempdir().unwrap();
        let server = RangeServer::new(data(40));
        *server.cut_at.lock() = Some(24);
        assert!(publish(&server, tmp.path(), None).is_err());

        // Same size, new version; the caller still holds the old probe.
        let stale = server.info();
        *server.cut_at.lock() = None;
        *server.etag.lock() = "\"v2\"".to_string();
        server.requests.lock().clear();
        let dl = downloader(&server, 1);
        dl.set_chunked_threshold(Some(1));
        dl.download_file("w.gguf", "https://x/w", tmp.path(), &stale, None).unwrap();

        assert_eq!(std::fs::read(tmp.path().join("w.gguf")).unwrap(), server.data);
        let requests = server.requests.lock().clone();
        assert_eq!(requests[0], ("bytes=24-39".to_string(), Some("\"v1\"".to_string())));
        assert_eq!(requests.last().unwrap(), &("bytes=0-39".to_string(), Some("\"v2\"".to_string())));
    }

    #[test]
    fn test_resumable_bytes_count_only_a_matching_journal() {
        use smartloop::model::downloader::artifact_downloader::resumable_bytes;
        let tmp = tempfile::tempdir().unwrap();
        let part = tmp.path().join("w.gguf.part");
        let server = RangeServer::new(data(40));
        *server.cut_at.lock() = Some(24);
        assert!(fetch(&server, &part).is_err());

        assert_eq!(resumable_bytes(&part, &server.info()), 24);
        let other = RemoteFileInfo { etag: Some("\"v2\"".to_string()), ..server.info() };
        assert_eq!(resumable_bytes(&part, &other), 0);
        assert_eq!(resumable_bytes(&tmp.path().join("none.part"), &server.info()), 0);
    }
}

/// Free space is checked before the first byte is fetched.
mod test_disk_space {
    use std::sync::Arc;

    use smartloop::model::downloader::artifact_downloader::{SLPArtifactDownloader, is_disk_full};
    use smartloop::model::downloader::artifact_source::ArtifactMap;

    use crate::shared::model_fakes::{RangeServer, test_logger};

    fn download(server: &Arc<RangeServer>, dir: &std::path::Path, free: u64) -> Result<std::path::PathBuf, smartloop::model::downloader::artifact_downloader::DownloadError> {
        let dl = SLPArtifactDownloader::for_tests(None, test_logger(), None).with_http(Arc::clone(server) as _);
        dl.cancel_token().set_instant_wait(true);
        dl.set_connections(Some(2));
        dl.set_chunked_threshold(Some(1));
        dl.set_free_space(Some(free));
        let mut artifacts = ArtifactMap::new();
        artifacts.insert("w.gguf".to_string(), "https://x/w".to_string());
        dl.download_artifacts(&artifacts, dir, None, "Model", dir, "w")
    }

    #[test]
    fn test_too_little_space_fails_before_any_request() {
        let tmp = tempfile::tempdir().unwrap();
        let server = RangeServer::new(vec![1u8; 4000]);
        let err = download(&server, tmp.path(), 3000).unwrap_err();
        assert!(is_disk_full(&err));
        assert!(err.to_string().contains("Not enough disk space"), "{err}");
        assert_eq!(smartloop::api::routers::_errors::status_for(&err), 507);
        assert!(server.ranges().is_empty());
        assert!(!tmp.path().join("w.gguf.part").exists());
    }

    #[test]
    fn test_bytes_a_resume_already_has_are_not_counted() {
        let tmp = tempfile::tempdir().unwrap();
        let server = RangeServer::new((0..4000u32).map(|i| i as u8).collect());
        *server.cut_at.lock() = Some(3000);
        assert!(download(&server, tmp.path(), 1 << 20).is_err());

        *server.cut_at.lock() = None;
        download(&server, tmp.path(), 1100).unwrap();
        assert_eq!(std::fs::read(tmp.path().join("w.gguf")).unwrap(), server.data);
    }

    /// Space a running download still has to write is not free for the
    /// next one; what it has received since starting is already on disk.
    #[test]
    fn test_space_owed_to_a_running_download_is_counted() {
        use smartloop::model::downloader::artifact_downloader::DiskReservations;
        use smartloop::model::downloader::ranges::{TransferStats, system_clock};

        let reservations = DiskReservations::new();
        let other = TransferStats::new(system_clock());
        let held = reservations.hold(3000, &other);
        let download_beside = |free: u64| {
            let tmp = tempfile::tempdir().unwrap();
            let server = RangeServer::new(vec![1u8; 4000]);
            let dl = SLPArtifactDownloader::for_tests(None, test_logger(), None)
                .with_http(Arc::clone(&server) as _)
                .with_reservations(Arc::clone(&reservations));
            dl.cancel_token().set_instant_wait(true);
            dl.set_connections(Some(2));
            dl.set_chunked_threshold(Some(1));
            dl.set_free_space(Some(free));
            let mut artifacts = ArtifactMap::new();
            artifacts.insert("w.gguf".to_string(), "https://x/w".to_string());
            (dl.download_artifacts(&artifacts, tmp.path(), None, "Model", tmp.path(), "w"), server)
        };

        // 6000 free, 3000 of it owed: 3000 left for a 4000-byte download.
        let (result, server) = download_beside(6000);
        let err = result.unwrap_err();
        assert!(is_disk_full(&err));
        assert!(err.to_string().contains("held for downloads in progress"), "{err}");
        assert!(server.ranges().is_empty());

        // The other download has since written 2000 of its 3000.
        other.open().add(2000);
        let (result, _) = download_beside(6000);
        result.unwrap();
        assert_eq!(reservations.owed(), 1000, "a finished download releases its claim");

        drop(held);
        assert_eq!(reservations.owed(), 0);
    }

    #[test]
    fn test_enough_space_downloads() {
        let tmp = tempfile::tempdir().unwrap();
        let server = RangeServer::new(vec![1u8; 4000]);
        download(&server, tmp.path(), 4100).unwrap();
        assert!(tmp.path().join("w.gguf").exists());
    }
}
