//! How many connections a download opens and how its files share them.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use smartloop::model::downloader::artifact_downloader::{RemoteFileInfo, SEGMENT_SIZE};
use smartloop::model::downloader::ranges::{ConnectionPermits, MAX_CONNECTIONS, starting_connections};

use crate::shared::model_fakes::downloader;

const MB: u64 = 1024 * 1024;
const GB: u64 = 1024 * MB;

/// `starting_connections`
mod test_starting_count {
    use super::*;

    #[test]
    fn test_it_grows_with_the_bytes_to_fetch() {
        assert_eq!(starting_connections(0), 1);
        assert_eq!(starting_connections(99 * MB), 1);
        assert_eq!(starting_connections(100 * MB), 4);
        assert_eq!(starting_connections(2 * GB - 1), 4);
        assert_eq!(starting_connections(2 * GB), 8);
        assert_eq!(starting_connections(40 * GB), 8);
        assert!(starting_connections(u64::MAX) <= MAX_CONNECTIONS);
    }
}

/// `ConnectionPermits`
mod test_permits {
    use super::*;

    #[test]
    fn test_a_permit_is_released_on_drop() {
        let permits = ConnectionPermits::new(1);
        let first = permits.acquire(|| false).unwrap();
        assert_eq!(permits.in_use(), 1);
        assert!(permits.acquire(|| true).is_none(), "gives up instead of waiting forever");
        drop(first);
        assert_eq!(permits.in_use(), 0);
        assert!(permits.acquire(|| false).is_some());
    }

    #[test]
    fn test_raising_the_limit_wakes_a_waiter() {
        let permits = ConnectionPermits::new(1);
        let _held = permits.acquire(|| false).unwrap();
        let waiter = {
            let permits = Arc::clone(&permits);
            std::thread::spawn(move || permits.acquire(|| false).is_some())
        };
        std::thread::sleep(Duration::from_millis(20));
        permits.set_limit(2);
        assert!(waiter.join().unwrap());
    }

    #[test]
    fn test_lowering_the_limit_holds_back_new_connections() {
        let permits = ConnectionPermits::new(3);
        let mut held: Vec<_> = (0..3).map(|_| permits.acquire(|| false).unwrap()).collect();
        permits.set_limit(1);
        held.pop();
        assert_eq!(permits.in_use(), 2);
        let gave_up = permits.acquire({
            let tries = AtomicUsize::new(0);
            move || tries.fetch_add(1, Ordering::SeqCst) > 2
        });
        assert!(gave_up.is_none(), "2 open is still over a limit of 1");
    }
}

/// The files of one download draw on one pool.
mod test_shared_pool {
    use super::*;

    /// Run a chunked download whose ranges just record how many run at once.
    fn peak_while(dl: &smartloop::model::downloader::artifact_downloader::SLPArtifactDownloader, files: usize) -> usize {
        let now = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        {
            let (now, peak) = (Arc::clone(&now), Arc::clone(&peak));
            dl.set_download_chunk_hook(Some(Arc::new(move |_start, _end| {
                let open = now.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(open, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(5));
                now.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            })));
        }
        let tmp = tempfile::tempdir().unwrap();
        std::thread::scope(|scope| {
            for i in 0..files {
                let path = tmp.path().join(format!("f{i}.part"));
                scope.spawn(move || {
                    dl.download_file_chunked("f", "https://x/f", &path, &RemoteFileInfo::sized(12 * SEGMENT_SIZE), None)
                        .unwrap()
                });
            }
        });
        peak.load(Ordering::SeqCst)
    }

    #[test]
    fn test_two_files_never_exceed_the_pool_together() {
        let dl = downloader(None, None);
        dl.set_connections(Some(3));
        let peak = peak_while(&dl, 2);
        assert!(peak <= 3, "peak {peak}");
        assert!(peak >= 2, "peak {peak}: the pool was used");
    }
}

/// A clock tests move by hand.
pub fn manual_clock() -> (smartloop::model::downloader::ranges::Clock, Arc<parking_lot::Mutex<Duration>>) {
    let now = Arc::new(parking_lot::Mutex::new(Duration::ZERO));
    let read = Arc::clone(&now);
    (Arc::new(move || *read.lock()), now)
}

/// `TransferStats` and `ConnectionMeter`
mod test_throughput {
    use super::*;
    use smartloop::model::downloader::ranges::{RATE_WINDOW, Slowness, TransferStats};

    const KB: u64 = 1024;

    /// Feed `meter` `rate` bytes per second for `secs` seconds.
    fn run(meter: &mut smartloop::model::downloader::ranges::ConnectionMeter, now: &parking_lot::Mutex<Duration>, rate: u64, secs: u64) {
        for _ in 0..secs {
            *now.lock() += Duration::from_secs(1);
            meter.add(rate);
        }
    }

    #[test]
    fn test_a_new_connection_is_not_judged() {
        let (clock, now) = manual_clock();
        let stats = TransferStats::new(clock);
        let mut meter = stats.open();
        run(&mut meter, &now, 1, RATE_WINDOW.as_secs() - 1);
        assert_eq!(meter.slowness(), None);
    }

    #[test]
    fn test_a_trickle_is_stalled() {
        let (clock, now) = manual_clock();
        let stats = TransferStats::new(clock);
        let mut meter = stats.open();
        run(&mut meter, &now, 8 * KB, RATE_WINDOW.as_secs());
        assert!(matches!(meter.slowness(), Some(Slowness::Stalled { .. })));
    }

    #[test]
    fn test_a_connection_far_behind_its_peers_is_slow() {
        let (clock, now) = manual_clock();
        let stats = TransferStats::new(clock);
        let mut fast: Vec<_> = (0..3).map(|_| stats.open()).collect();
        let mut slow = stats.open();
        for _ in 0..RATE_WINDOW.as_secs() {
            *now.lock() += Duration::from_secs(1);
            for meter in &mut fast {
                meter.add(4 * MB);
            }
            slow.add(512 * KB);
        }
        assert!(matches!(slow.slowness(), Some(Slowness::BehindPeers { .. })));
        assert_eq!(fast[0].slowness(), None);
        assert!(stats.total_rate() > 12.0 * MB as f64);
    }

    #[test]
    fn test_a_link_that_is_slow_for_everyone_is_left_alone() {
        let (clock, now) = manual_clock();
        let stats = TransferStats::new(clock);
        let mut meters: Vec<_> = (0..4).map(|_| stats.open()).collect();
        for _ in 0..RATE_WINDOW.as_secs() {
            *now.lock() += Duration::from_secs(1);
            for meter in &mut meters {
                meter.add(40 * KB);
            }
        }
        assert!(meters.iter_mut().all(|m| m.slowness().is_none()));
    }

    #[test]
    fn test_one_peer_is_not_enough_to_compare_against() {
        let (clock, now) = manual_clock();
        let stats = TransferStats::new(clock);
        let mut fast = stats.open();
        let mut slow = stats.open();
        for _ in 0..RATE_WINDOW.as_secs() {
            *now.lock() += Duration::from_secs(1);
            fast.add(4 * MB);
            slow.add(64 * KB);
        }
        assert_eq!(slow.slowness(), None);
    }

    #[test]
    fn test_a_closed_connection_stops_counting_as_a_peer() {
        let (clock, now) = manual_clock();
        let stats = TransferStats::new(clock);
        let mut fast: Vec<_> = (0..2).map(|_| stats.open()).collect();
        let mut slow = stats.open();
        for _ in 0..RATE_WINDOW.as_secs() {
            *now.lock() += Duration::from_secs(1);
            for meter in &mut fast {
                meter.add(4 * MB);
            }
            slow.add(512 * KB);
        }
        fast.pop();
        assert_eq!(slow.slowness(), None);
    }
}

/// A connection the meter calls slow is dropped and asked again.
mod test_low_speed_abort {
    use super::*;
    use crate::shared::model_fakes::{FakeResp, ScriptedHttp, range_header};
    use parking_lot::Mutex;
    use smartloop::model::downloader::artifact_downloader::{DownloadError, HttpError, TransportErrorKind, is_retryable_error};

    #[test]
    fn test_a_stalled_connection_is_reissued_from_the_next_byte() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("w.part");
        std::fs::write(&dest, [0u8; 32]).unwrap();
        let (clock, now) = manual_clock();
        // 4 bytes every 10s: stalled once the 30s window is up, after CCCC.
        let trickle = FakeResp::new(206, &[b"AAAA", b"BBBB", b"CCCC", b"DDDD", b"EEEE", b"FFFF"]).with_on_chunk({
            let now = Arc::clone(&now);
            move |_| *now.lock() += Duration::from_secs(10)
        });
        let rest = FakeResp::new(206, &[b"DDDDEEEEFFFFGGGGHHHH"]);
        let http = ScriptedHttp::new(vec![trickle, rest]);
        let dl = downloader(None, Some(Arc::clone(&http)));
        dl.set_clock(Some(clock));

        let counter = Mutex::new(0u64);
        dl.download_chunk("https://x/w", &dest, 0, 31, "w.gguf", 32, &counter, None).unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), b"AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH");
        let calls = http.header_calls.lock();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1], range_header("bytes=12-31"));
    }

    #[test]
    fn test_a_low_speed_abort_is_retryable() {
        let err: DownloadError = HttpError::transport(TransportErrorKind::LowSpeed, "slow").into();
        assert!(is_retryable_error(&err));
    }
}

/// An idle connection takes half of the largest range still in flight.
mod test_tail_splitting {
    use super::*;
    use crate::shared::model_fakes::{RangeServer, test_logger};
    use smartloop::model::downloader::artifact_downloader::SLPArtifactDownloader;
    use smartloop::model::downloader::ranges::{InFlight, RangeCursor};

    #[test]
    fn test_a_split_takes_the_upper_half_beyond_the_claim() {
        let in_flight = InFlight::new(4);
        let cursor = RangeCursor::new(0, 31);
        in_flight.start(&cursor);
        assert_eq!(cursor.claim(8), Some((0, 8)));

        let taken = in_flight.split().unwrap();
        assert_eq!(taken.position(), (20, 31));
        assert_eq!(cursor.position(), (8, 19));
        // The owner can no longer claim past its new end.
        assert_eq!(cursor.claim(100), Some((8, 12)));
        assert_eq!(cursor.claim(1), None);
    }

    #[test]
    fn test_a_range_too_small_to_halve_is_left_alone() {
        let in_flight = InFlight::new(16);
        let cursor = RangeCursor::new(0, 30);
        in_flight.start(&cursor);
        assert!(!in_flight.splittable());
        assert!(in_flight.split().is_none());
    }

    #[test]
    fn test_a_failed_write_gives_its_claim_back() {
        let cursor = RangeCursor::new(10, 19);
        let (offset, _) = cursor.claim(4).unwrap();
        cursor.unclaim(offset);
        assert_eq!(cursor.position(), (10, 19));
    }

    #[test]
    fn test_a_slow_last_range_is_shared() {
        let tmp = tempfile::tempdir().unwrap();
        let data: Vec<u8> = (0..64u8).collect();
        let server = RangeServer::new(data.clone());
        *server.slow_from.lock() = Some((0, Duration::from_millis(30)));
        let dl = SLPArtifactDownloader::for_tests(None, test_logger(), None).with_http(Arc::clone(&server) as _);
        dl.set_connections(Some(2));
        dl.set_min_split(Some(4));

        let part = tmp.path().join("w.part");
        dl.download_file_chunked("w", "https://x/w", &part, &server.info(), None).unwrap();

        assert_eq!(std::fs::read(&part).unwrap(), data);
        let ranges = server.ranges();
        assert!(
            ranges.iter().any(|r| {
                let (a, b) = r.strip_prefix("bytes=").unwrap().split_once('-').unwrap();
                let (a, b): (u64, u64) = (a.parse().unwrap(), b.parse().unwrap());
                a > 0 && a < 32 && b == 31
            }),
            "the slow first range was split: {ranges:?}"
        );
    }
}
