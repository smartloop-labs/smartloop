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
