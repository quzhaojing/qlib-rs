use super::*;
use std::{io, time::Duration};

#[path = "rl_checkpoint_file_integration.rs"]
mod integration;

fn storage() -> FileRlCheckpointStorage<BincodeRlCheckpointCodec> {
    FileRlCheckpointStorage::new(BincodeRlCheckpointCodec)
}

fn files(
    storage: &mut FileRlCheckpointStorage<BincodeRlCheckpointCodec>,
) -> &mut dyn RlCheckpointStorage<Vec<i64>> {
    storage
}

#[test]
fn file_codec_preserves_values_copy_and_errors() {
    let directory = tempfile::tempdir().unwrap();
    let target = directory.path().join("new");
    let mut store = storage();
    assert!(!target.exists());
    files(&mut store).create_directory(&target).unwrap();
    let saved = target.join("state.bin");
    files(&mut store).save(&vec![1, 2, 3], &saved).unwrap();
    assert_eq!(store.load::<Vec<i64>>(&saved).unwrap(), vec![1, 2, 3]);
    let latest = target.join("latest.bin");
    assert!(!files(&mut store).exists(&latest).unwrap());
    assert!(!files(&mut store).is_link(&latest).unwrap());
    files(&mut store).copy(&saved, &latest).unwrap();
    assert_eq!(fs::read(&saved).unwrap(), fs::read(&latest).unwrap());
    files(&mut store).save(&vec![], &saved).unwrap();
    assert!(store.load::<Vec<i64>>(&saved).unwrap().is_empty());
    assert_eq!(store.load::<Vec<i64>>(&latest).unwrap(), vec![1, 2, 3]);
    assert!(files(&mut store).create_directory(&saved).is_err());
    assert!(files(&mut store).save(&vec![1], &target).is_err());
    assert!(
        files(&mut store)
            .save(&vec![1], &target.join("missing/child"))
            .is_err()
    );
    assert!(store.load::<Vec<i64>>(&target.join("absent")).is_err());
    assert!(files(&mut store).copy(&saved, &target).is_err());
    assert!(files(&mut store).remove(&target).is_err());
    files(&mut store).remove(&latest).unwrap();
    assert!(files(&mut store).remove(&latest).is_err());
    fs::write(&saved, [1, 2]).unwrap();
    assert!(store.load::<Vec<i64>>(&saved).is_err());
}

#[test]
fn file_links_preserve_target_text_and_dangling_detection() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.bin");
    let link = directory.path().join("latest.bin");
    let mut store = storage();
    files(&mut store).save(&vec![42], &source).unwrap();
    // On Windows this requires Developer Mode or symlink privilege. Do not silently copy.
    files(&mut store).link(&source, &link).unwrap();
    assert_eq!(fs::read_link(&link).unwrap(), source);
    assert!(files(&mut store).exists(&link).unwrap());
    assert!(files(&mut store).is_link(&link).unwrap());
    assert_eq!(store.load::<Vec<i64>>(&link).unwrap(), vec![42]);
    assert!(files(&mut store).link(&source, &link).is_err());
    files(&mut store).remove(&source).unwrap();
    assert!(!files(&mut store).exists(&link).unwrap());
    assert!(files(&mut store).is_link(&link).unwrap());
    files(&mut store).remove(&link).unwrap();
    let relative = Path::new("nested/source.bin");
    files(&mut store).link(relative, &link).unwrap();
    assert_eq!(fs::read_link(&link).unwrap(), relative);
    assert!(!files(&mut store).exists(&link).unwrap());
}

struct BrokenIo;
impl Read for BrokenIo {
    fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::other("read"))
    }
}
impl Write for BrokenIo {
    fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("write"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn streaming_codec_and_wall_clock_boundaries_are_explicit() {
    let mut codec = BincodeRlCheckpointCodec;
    assert!(
        codec
            .encode(&vec![42_i64], &mut BrokenIo)
            .unwrap_err()
            .contains("write")
    );
    let result: Result<Vec<i64>, _> = codec.decode(&mut BrokenIo);
    assert!(result.unwrap_err().contains("read"));
    assert_eq!(
        timestamp(UNIX_EPOCH + Duration::from_millis(1250)).to_bits(),
        1.25_f64.to_bits()
    );
    assert_eq!(
        timestamp(UNIX_EPOCH - Duration::from_millis(1250)).to_bits(),
        (-1.25_f64).to_bits()
    );
    let mut clock = SystemRlCheckpointClock;
    let before = timestamp(SystemTime::now());
    let now = clock.timestamp().unwrap();
    let after = timestamp(SystemTime::now());
    assert!(now >= before && now <= after);
    let local_before = chrono::Local::now().format("%Y%m%d%H%M%S").to_string();
    let local = clock.local_time().unwrap();
    let local_after = chrono::Local::now().format("%Y%m%d%H%M%S").to_string();
    assert!(local == local_before || local == local_after);
}
