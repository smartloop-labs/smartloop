//! Checking downloaded files against single-part and multipart ETags.

use md5::{Digest, Md5};
use smartloop::model::downloader::cancellation::CancelToken;
use smartloop::model::downloader::digest::{RemoteDigest, Verdict, part_sizes, verify};

const MIB: u64 = 1024 * 1024;

/// The multipart ETag S3 and R2 compute for `data` uploaded in `part`-byte parts.
fn multipart_etag(data: &[u8], part: usize) -> RemoteDigest {
    let mut outer = Md5::new();
    let mut parts = 0;
    for chunk in data.chunks(part) {
        outer.update(Md5::digest(chunk));
        parts += 1;
    }
    RemoteDigest::Multipart { md5: hex::encode(outer.finalize()), parts }
}

fn file_with(data: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("w.gguf");
    std::fs::write(&path, data).unwrap();
    (tmp, path)
}

#[test]
fn test_etags_are_read_as_digests() {
    let md5 = "0123456789ABCDEF0123456789abcdef";
    assert_eq!(RemoteDigest::from_etag(&format!("\"{md5}\"")), Some(RemoteDigest::Md5(md5.to_lowercase())));
    assert_eq!(
        RemoteDigest::from_etag(&format!("\"{md5}-2213\"")),
        Some(RemoteDigest::Multipart { md5: md5.to_lowercase(), parts: 2213 })
    );
    assert_eq!(RemoteDigest::from_etag(&format!("W/\"{md5}\"")), None);
    assert_eq!(RemoteDigest::from_etag("\"v1\""), None);
    assert_eq!(RemoteDigest::from_etag(&format!("\"{md5}-0\"")), None);
}

#[test]
fn test_the_part_size_is_worked_out_from_the_count() {
    // qwen3-coder-30b-a3b on R2.
    assert_eq!(part_sizes(18_556_688_288, 2213), vec![8 * MIB]);
    assert_eq!(part_sizes(100, 7), Vec::<u64>::new());
}

#[test]
fn test_a_matching_multipart_file_verifies() {
    let data: Vec<u8> = (0..(10 * MIB as usize + 1234)).map(|i| (i % 253) as u8).collect();
    let (_tmp, path) = file_with(&data);
    let digest = multipart_etag(&data, 5 * MIB as usize);
    assert_eq!(verify(&path, &digest, None).unwrap(), Verdict::Match);
}

#[test]
fn test_a_corrupt_multipart_file_does_not() {
    let mut data: Vec<u8> = (0..(10 * MIB as usize + 1234)).map(|i| (i % 253) as u8).collect();
    let digest = multipart_etag(&data, 5 * MIB as usize);
    data[7 * MIB as usize] ^= 0xFF;
    let (_tmp, path) = file_with(&data);
    assert!(matches!(verify(&path, &digest, None).unwrap(), Verdict::Mismatch { .. }));
}

#[test]
fn test_a_part_count_no_size_explains_is_unverifiable() {
    let (_tmp, path) = file_with(b"hello");
    let digest = RemoteDigest::Multipart { md5: "0".repeat(32), parts: 7 };
    assert_eq!(verify(&path, &digest, None).unwrap(), Verdict::Unverifiable);
}

#[test]
fn test_a_single_part_md5_still_verifies() {
    let (_tmp, path) = file_with(b"hello");
    let digest = RemoteDigest::Md5(hex::encode(Md5::digest(b"hello")));
    assert_eq!(verify(&path, &digest, None).unwrap(), Verdict::Match);
    let wrong = RemoteDigest::Md5(hex::encode(Md5::digest(b"world")));
    assert!(matches!(verify(&path, &wrong, None).unwrap(), Verdict::Mismatch { .. }));
}

#[test]
fn test_verifying_honours_a_cancel() {
    let (_tmp, path) = file_with(&vec![0u8; 3 * 8 * MIB as usize]);
    let token = CancelToken::new();
    token.cancel();
    let digest = RemoteDigest::Md5("0".repeat(32));
    assert!(verify(&path, &digest, Some(&token)).unwrap_err().is_cancelled());
}

#[test]
fn test_the_aggregate_bar_passes_verifying_through() {
    use std::sync::Arc;

    use parking_lot::Mutex;
    use smartloop::model::downloader::slp_downloader::{AggregateProgress, DownloadProgress};

    let seen = Arc::new(Mutex::new(Vec::new()));
    let record = Arc::clone(&seen);
    let aggregate = AggregateProgress::new(10, Some(Arc::new(move |p: DownloadProgress| record.lock().push(p.status))), "M");
    aggregate.call(DownloadProgress::new("a", 10, 10, "downloading"));
    aggregate.call(DownloadProgress::new("a", 10, 10, "verifying"));
    assert_eq!(*seen.lock(), vec!["downloading".to_string(), "verifying".to_string()]);
}
