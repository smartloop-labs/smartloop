//! Checking a downloaded file against the digest its server published.
//!
//! Single-part objects carry their MD5 as the ETag.  Objects uploaded in
//! parts (S3, R2, GCS XML multipart) carry `"<md5>-<n>"` instead: the MD5 of
//! the n part digests concatenated.  Reproducing that needs the part size,
//! which is not published, so it is worked out from the file size and part
//! count.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use md5::{Digest, Md5};

use crate::model::downloader::artifact_downloader::{CHUNK_SIZE, DownloadResult};
use crate::model::downloader::cancellation::CancelToken;

const MIB: u64 = 1024 * 1024;

/// Part sizes uploaders commonly use, tried when matching a multipart ETag.
pub const PART_SIZE_CANDIDATES: &[u64] =
    &[5 * MIB, 8 * MIB, 10 * MIB, 15 * MIB, 16 * MIB, 32 * MIB, 64 * MIB, 100 * MIB, 128 * MIB, 256 * MIB, 512 * MIB];

/// The digest a server published for an object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteDigest {
    /// Hex MD5 of the whole object.
    Md5(String),
    /// Hex MD5 of the concatenated part MD5s, and the part count.
    Multipart { md5: String, parts: u64 },
}

impl RemoteDigest {
    /// Read a digest from a strong ETag (quoted or not); `None` for a weak
    /// one or anything else.
    pub fn from_etag(etag: &str) -> Option<Self> {
        let etag = etag.trim();
        if etag.starts_with("W/") {
            return None;
        }
        let etag = etag.trim_matches('"');
        let is_md5 = |s: &str| s.len() == 32 && s.chars().all(|c| c.is_ascii_hexdigit());
        if is_md5(etag) {
            return Some(Self::Md5(etag.to_lowercase()));
        }
        let (md5, parts) = etag.split_once('-')?;
        let parts: u64 = parts.parse().ok()?;
        (is_md5(md5) && parts > 0).then(|| Self::Multipart { md5: md5.to_lowercase(), parts })
    }
}

impl std::fmt::Display for RemoteDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Md5(md5) => write!(f, "{md5}"),
            Self::Multipart { md5, parts } => write!(f, "{md5}-{parts}"),
        }
    }
}

/// The part sizes that cut `size` bytes into exactly `parts` parts: the
/// common ones, plus the even split rounded up to a whole MiB.
pub fn part_sizes(size: u64, parts: u64) -> Vec<u64> {
    if size == 0 || parts == 0 {
        return Vec::new();
    }
    let even = size.div_ceil(parts).div_ceil(MIB) * MIB;
    let mut sizes: Vec<u64> =
        PART_SIZE_CANDIDATES.iter().copied().chain([even]).filter(|ps| size.div_ceil(*ps) == parts).collect();
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

/// What checking a file against a [`RemoteDigest`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Match,
    /// The digest computed locally (for a multipart ETag, under the first
    /// part size tried).
    Mismatch { got: String },
    /// No part size reproduces the ETag's part count, so it can't be checked.
    Unverifiable,
}

/// The multipart ETag of a stream, for one part size.
struct MultipartHasher {
    part_size: u64,
    part: Md5,
    in_part: u64,
    outer: Md5,
    parts: u64,
}

impl MultipartHasher {
    fn new(part_size: u64) -> Self {
        Self { part_size, part: Md5::new(), in_part: 0, outer: Md5::new(), parts: 0 }
    }

    fn update(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            let take = ((self.part_size - self.in_part) as usize).min(data.len());
            self.part.update(&data[..take]);
            self.in_part += take as u64;
            data = &data[take..];
            if self.in_part == self.part_size {
                self.close_part();
            }
        }
    }

    fn close_part(&mut self) {
        let digest = std::mem::take(&mut self.part).finalize();
        self.outer.update(digest);
        self.parts += 1;
        self.in_part = 0;
    }

    fn finish(mut self) -> String {
        if self.in_part > 0 {
            self.close_part();
        }
        format!("{}-{}", hex::encode(self.outer.finalize()), self.parts)
    }
}

/// Check `path` against `digest` in one read of the file, honouring
/// `cancel` between reads.
pub fn verify(path: &Path, digest: &RemoteDigest, cancel: Option<&CancelToken>) -> DownloadResult<Verdict> {
    let size = std::fs::metadata(path)?.len();
    let mut whole = Md5::new();
    let mut multipart: Vec<MultipartHasher> = match digest {
        RemoteDigest::Md5(_) => Vec::new(),
        RemoteDigest::Multipart { parts, .. } => part_sizes(size, *parts).into_iter().map(MultipartHasher::new).collect(),
    };
    if matches!(digest, RemoteDigest::Multipart { .. }) && multipart.is_empty() {
        return Ok(Verdict::Unverifiable);
    }

    let mut file = File::open(path)?;
    let mut buf = vec![0u8; CHUNK_SIZE];
    loop {
        let n = match file.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        };
        if let Some(cancel) = cancel {
            cancel.check("verification")?;
        }
        match digest {
            RemoteDigest::Md5(_) => whole.update(&buf[..n]),
            RemoteDigest::Multipart { .. } => multipart.iter_mut().for_each(|h| h.update(&buf[..n])),
        }
    }

    let expected = digest.to_string();
    let computed: Vec<String> = match digest {
        RemoteDigest::Md5(_) => vec![hex::encode(whole.finalize())],
        RemoteDigest::Multipart { .. } => multipart.into_iter().map(MultipartHasher::finish).collect(),
    };
    if computed.iter().any(|got| *got == expected) {
        return Ok(Verdict::Match);
    }
    Ok(Verdict::Mismatch { got: computed.into_iter().next().unwrap_or_default() })
}
