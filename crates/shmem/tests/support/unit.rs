use std::{
    fs,
    io::Write,
    sync::atomic::{AtomicU64, Ordering},
};

use super::*;

static NEXT_PATH_ID: AtomicU64 = AtomicU64::new(0);

fn unique_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "shmem-{name}-{}-{}",
        std::process::id(),
        NEXT_PATH_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

fn remove(path: &Path) {
    let _ = fs::remove_file(path);
}

#[test]
fn create_open_write_and_read_round_trip() {
    let path = unique_path("round-trip");
    remove(&path);
    let mut writer = MappedObservationRegion::create(&path, 16).unwrap();
    assert_eq!(writer.path(), path);
    assert_eq!(writer.capacity(), 16);
    assert!(matches!(
        writer.read_frame(),
        Err(MappedObservationError::Uninitialized)
    ));
    let reader = MappedObservationRegion::open(&path, 16).unwrap();
    writer.write_frame(7, b"observation").unwrap();
    assert_eq!(
        reader.read_frame().unwrap(),
        MappedObservationFrame {
            request_id: 7,
            payload: b"observation".to_vec()
        }
    );
    writer.write_frame(u64::MAX, b"").unwrap();
    assert_eq!(reader.read_frame().unwrap().request_id, u64::MAX);
    drop(reader);
    drop(writer);
    remove(&path);
}

#[test]
fn validates_create_open_and_payload_sizes() {
    let path = unique_path("sizes");
    remove(&path);
    assert!(matches!(
        MappedObservationRegion::create(&path, 0),
        Err(MappedObservationError::ZeroCapacity)
    ));
    assert!(matches!(
        MappedObservationRegion::create(&path, usize::MAX),
        Err(MappedObservationError::SizeOverflow)
    ));
    assert!(matches!(
        MappedObservationRegion::open(&path, 0),
        Err(MappedObservationError::ZeroCapacity)
    ));
    assert!(matches!(
        MappedObservationRegion::open(&path, usize::MAX),
        Err(MappedObservationError::SizeOverflow)
    ));
    assert!(matches!(
        MappedObservationRegion::create_with(
            &path,
            1,
            |path| File::create(path),
            |_, _| Err(io::Error::other("resize failed")),
            map_file,
        ),
        Err(MappedObservationError::Io(_))
    ));
    remove(&path);
    assert!(matches!(
        MappedObservationRegion::create_with(
            &path,
            1,
            |path| File::create(path),
            File::set_len,
            |_| Err(io::Error::other("map failed")),
        ),
        Err(MappedObservationError::Io(_))
    ));
    remove(&path);
    assert!(matches!(
        MappedObservationRegion::open(&path, 1),
        Err(MappedObservationError::Io(_))
    ));
    let mut region = MappedObservationRegion::create(&path, 2).unwrap();
    assert!(matches!(
        MappedObservationRegion::create(&path, 2),
        Err(MappedObservationError::Io(_))
    ));
    assert!(matches!(
        MappedObservationRegion::open(&path, 3),
        Err(MappedObservationError::RegionLength { .. })
    ));
    assert!(matches!(
        region.write_frame(0, b"abc"),
        Err(MappedObservationError::PayloadTooLarge {
            capacity: 2,
            actual: 3
        })
    ));
    drop(region);
    remove(&path);
}

#[test]
fn rejects_corrupt_headers_payloads_and_versions() {
    let path = unique_path("corrupt");
    remove(&path);
    let mut region = MappedObservationRegion::create(&path, 8).unwrap();

    region.mapping[0] = b'X';
    assert!(matches!(
        region.read_frame(),
        Err(MappedObservationError::InvalidMagic)
    ));

    region.write_frame(1, b"ok").unwrap();
    region.mapping[VERSION_OFFSET..VERSION_OFFSET + 2].copy_from_slice(&2_u16.to_le_bytes());
    assert!(matches!(
        region.read_frame(),
        Err(MappedObservationError::UnsupportedVersion {
            expected: 1,
            actual: 2
        })
    ));

    region.write_frame(1, b"ok").unwrap();
    region.mapping[PAYLOAD_LEN_OFFSET..PAYLOAD_LEN_OFFSET + 8]
        .copy_from_slice(&9_u64.to_le_bytes());
    assert!(matches!(
        region.read_frame(),
        Err(MappedObservationError::InvalidPayloadLength {
            capacity: 8,
            actual: 9
        })
    ));

    region.write_frame(1, b"ok").unwrap();
    region.mapping[HEADER_LEN] ^= 1;
    assert!(matches!(
        region.read_frame(),
        Err(MappedObservationError::ChecksumMismatch { .. })
    ));
    drop(region);
    remove(&path);
}

#[test]
fn reports_metadata_failures() {
    let path = unique_path("metadata");
    remove(&path);
    let mut file = File::create(&path).unwrap();
    file.write_all(b"short").unwrap();
    assert!(matches!(
        MappedObservationRegion::open(&path, 8),
        Err(MappedObservationError::RegionLength {
            expected: 40,
            actual: 5
        })
    ));
    assert!(matches!(
        MappedObservationRegion::open_with(
            &path,
            8,
            |_| Err(io::Error::other("open failed")),
            File::metadata,
            map_file,
        ),
        Err(MappedObservationError::Io(_))
    ));
    file.set_len(40).unwrap();
    assert!(matches!(
        MappedObservationRegion::open_with(
            &path,
            8,
            |path| OpenOptions::new().read(true).write(true).open(path),
            |_| Err(io::Error::other("metadata failed")),
            map_file,
        ),
        Err(MappedObservationError::Io(_))
    ));
    assert!(matches!(
        MappedObservationRegion::open_with(
            &path,
            8,
            |path| OpenOptions::new().read(true).write(true).open(path),
            File::metadata,
            |_| Err(io::Error::other("map failed")),
        ),
        Err(MappedObservationError::Io(_))
    ));
    drop(file);
    remove(&path);
}
