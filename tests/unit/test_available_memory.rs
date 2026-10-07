//! `available_memory_gb` counts memory macOS can hand back, not just the
//! free pages `sysinfo` reports.

use smartloop::utils::device_utils::available_memory_gb;

#[test]
fn test_available_is_never_less_than_the_free_pages() {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let free_gb = sys.available_memory() as f64 / 1024f64.powi(3);

    let (available_gb, total_gb) = available_memory_gb();

    // Memory moves between the two readings; allow a little slack.
    assert!(available_gb + 0.5 >= free_gb, "{available_gb:.1}GB < {free_gb:.1}GB free");
    assert!(available_gb <= total_gb, "{available_gb:.1}GB > {total_gb:.1}GB total");
}
