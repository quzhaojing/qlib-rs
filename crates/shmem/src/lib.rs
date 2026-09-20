//! Safe observation-frame API over an audited `memmap2` boundary.
//!
//! A control transport must provide synchronization: the writer completes
//! [`MappedObservationRegion::write_frame`] before sending its control reply, and the reader calls
//! [`MappedObservationRegion::read_frame`] only after receiving that reply. The mapping itself is
//! the data plane and does not provide concurrent-reader synchronization.

use std::{
    fs::{File, OpenOptions},
    io,
    path::{Path, PathBuf},
};

use crc32fast::hash;
use memmap2::{MmapMut, MmapOptions};
use thiserror::Error;

pub const MAPPED_OBSERVATION_VERSION: u16 = 1;
const MAGIC: &[u8; 8] = b"QLIBSHM1";
const HEADER_LEN: usize = 32;
const VERSION_OFFSET: usize = 8;
const REQUEST_ID_OFFSET: usize = 12;
const PAYLOAD_LEN_OFFSET: usize = 20;
const CHECKSUM_OFFSET: usize = 28;

type OpenFile = fn(&Path) -> io::Result<File>;
type ResizeFile = fn(&File, u64) -> io::Result<()>;
type MapFile = fn(&File) -> io::Result<MmapMut>;
type ReadMetadata = fn(&File) -> io::Result<std::fs::Metadata>;

#[derive(Debug, Error)]
pub enum MappedObservationError {
    #[error("mapped observation capacity must be greater than zero")]
    ZeroCapacity,
    #[error("mapped observation size overflows the platform usize")]
    SizeOverflow,
    #[error("mapped observation I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("mapped observation region has {actual} bytes; expected {expected}")]
    RegionLength { expected: u64, actual: u64 },
    #[error("mapped observation region has not been initialized")]
    Uninitialized,
    #[error("mapped observation magic is invalid")]
    InvalidMagic,
    #[error("mapped observation version {actual} is unsupported; expected {expected}")]
    UnsupportedVersion { expected: u16, actual: u16 },
    #[error("mapped observation payload has {actual} bytes; capacity is {capacity}")]
    PayloadTooLarge { capacity: usize, actual: usize },
    #[error("mapped observation payload length {actual} exceeds capacity {capacity}")]
    InvalidPayloadLength { capacity: usize, actual: u64 },
    #[error("mapped observation checksum mismatch: expected {expected:#010x}, got {actual:#010x}")]
    ChecksumMismatch { expected: u32, actual: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MappedObservationFrame {
    pub request_id: u64,
    pub payload: Vec<u8>,
}

/// A fixed-capacity, file-backed shared mapping used by exactly one observation producer and one
/// observation consumer. Cross-process ordering is supplied by the separate control transport.
pub struct MappedObservationRegion {
    path: PathBuf,
    capacity: usize,
    _file: File,
    mapping: MmapMut,
}

impl MappedObservationRegion {
    /// Creates a new zero-initialized region without overwriting an existing path.
    ///
    /// # Errors
    /// Returns capacity, overflow, create, resize, or mapping failures.
    pub fn create(path: impl AsRef<Path>, capacity: usize) -> Result<Self, MappedObservationError> {
        Self::create_with(
            path.as_ref(),
            capacity,
            |path| {
                OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(path)
            },
            File::set_len,
            map_file,
        )
    }

    fn create_with(
        path: &Path,
        capacity: usize,
        open: OpenFile,
        resize: ResizeFile,
        map: MapFile,
    ) -> Result<Self, MappedObservationError> {
        let mapping_len = mapping_len(capacity)?;
        let file = open(path)?;
        resize(&file, mapping_len as u64)?;
        let mut region = Self::from_file(path, capacity, file, map)?;
        region.mapping.fill(0);
        Ok(region)
    }

    /// Opens an existing region and verifies its exact configured length.
    ///
    /// # Errors
    /// Returns capacity, overflow, open, metadata, length, or mapping failures.
    pub fn open(path: impl AsRef<Path>, capacity: usize) -> Result<Self, MappedObservationError> {
        Self::open_with(
            path.as_ref(),
            capacity,
            |path| OpenOptions::new().read(true).write(true).open(path),
            File::metadata,
            map_file,
        )
    }

    fn open_with(
        path: &Path,
        capacity: usize,
        open: OpenFile,
        metadata: ReadMetadata,
        map: MapFile,
    ) -> Result<Self, MappedObservationError> {
        let expected = mapping_len(capacity)?;
        let file = open(path)?;
        let actual = metadata(&file)?.len();
        if actual != expected as u64 {
            return Err(MappedObservationError::RegionLength {
                expected: expected as u64,
                actual,
            });
        }
        Self::from_file(path, capacity, file, map)
    }

    fn from_file(
        path: &Path,
        capacity: usize,
        file: File,
        map: MapFile,
    ) -> Result<Self, MappedObservationError> {
        let mapping = map(&file)?;
        Ok(Self {
            path: path.to_path_buf(),
            capacity,
            _file: file,
            mapping,
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Replaces the current frame. Callers must send the matching control response only after this
    /// method returns.
    ///
    /// # Errors
    /// Returns an error when the payload exceeds the fixed region capacity.
    pub fn write_frame(
        &mut self,
        request_id: u64,
        payload: &[u8],
    ) -> Result<(), MappedObservationError> {
        if payload.len() > self.capacity {
            return Err(MappedObservationError::PayloadTooLarge {
                capacity: self.capacity,
                actual: payload.len(),
            });
        }
        self.mapping[HEADER_LEN..HEADER_LEN + payload.len()].copy_from_slice(payload);
        let mut header = [0_u8; HEADER_LEN];
        header[..MAGIC.len()].copy_from_slice(MAGIC);
        header[VERSION_OFFSET..VERSION_OFFSET + 2]
            .copy_from_slice(&MAPPED_OBSERVATION_VERSION.to_le_bytes());
        header[REQUEST_ID_OFFSET..REQUEST_ID_OFFSET + 8].copy_from_slice(&request_id.to_le_bytes());
        header[PAYLOAD_LEN_OFFSET..PAYLOAD_LEN_OFFSET + 8]
            .copy_from_slice(&(payload.len() as u64).to_le_bytes());
        header[CHECKSUM_OFFSET..CHECKSUM_OFFSET + 4].copy_from_slice(&hash(payload).to_le_bytes());
        self.mapping[..HEADER_LEN].copy_from_slice(&header);
        Ok(())
    }

    /// Copies and validates the latest complete frame.
    ///
    /// # Errors
    /// Returns initialization, layout, version, length, or checksum failures.
    ///
    /// # Panics
    /// Panics only on a target whose `usize` cannot represent a payload already proven no larger
    /// than this region's `usize` capacity.
    pub fn read_frame(&self) -> Result<MappedObservationFrame, MappedObservationError> {
        if self.mapping[..MAGIC.len()].iter().all(|byte| *byte == 0) {
            return Err(MappedObservationError::Uninitialized);
        }
        if &self.mapping[..MAGIC.len()] != MAGIC {
            return Err(MappedObservationError::InvalidMagic);
        }
        let version = read_u16(&self.mapping, VERSION_OFFSET);
        if version != MAPPED_OBSERVATION_VERSION {
            return Err(MappedObservationError::UnsupportedVersion {
                expected: MAPPED_OBSERVATION_VERSION,
                actual: version,
            });
        }
        let request_id = read_u64(&self.mapping, REQUEST_ID_OFFSET);
        let payload_len = read_u64(&self.mapping, PAYLOAD_LEN_OFFSET);
        if payload_len > self.capacity as u64 {
            return Err(MappedObservationError::InvalidPayloadLength {
                capacity: self.capacity,
                actual: payload_len,
            });
        }
        let payload_len_usize =
            usize::try_from(payload_len).expect("payload length was bounded by a usize capacity");
        let payload = self.mapping[HEADER_LEN..HEADER_LEN + payload_len_usize].to_vec();
        let expected = read_u32(&self.mapping, CHECKSUM_OFFSET);
        let actual = hash(&payload);
        if actual != expected {
            return Err(MappedObservationError::ChecksumMismatch { expected, actual });
        }
        Ok(MappedObservationFrame {
            request_id,
            payload,
        })
    }
}

fn map_file(file: &File) -> io::Result<MmapMut> {
    // SAFETY: the caller retains `file` for the mapping lifetime, its length was set or verified,
    // and the safe region API exposes no references into the mapping. The control channel
    // serializes cross-process reads and writes.
    unsafe { MmapOptions::new().map_mut(file) }
}

fn mapping_len(capacity: usize) -> Result<usize, MappedObservationError> {
    if capacity == 0 {
        return Err(MappedObservationError::ZeroCapacity);
    }
    HEADER_LEN
        .checked_add(capacity)
        .ok_or(MappedObservationError::SizeOverflow)
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("fixed header field"),
    )
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("fixed header field"),
    )
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("fixed header field"),
    )
}

#[cfg(test)]
#[path = "../tests/support/unit.rs"]
mod tests;
