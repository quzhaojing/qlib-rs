use super::*;
use std::os::windows::ffi::OsStrExt;

fn packet(tag: u32, name: &[u16], offset: u16) -> Vec<u8> {
    let base = if tag == IO_REPARSE_TAG_SYMLINK { 12 } else { 8 };
    let mut data = vec![0; 8 + base + usize::from(offset) + name.len() * 2];
    LittleEndian::write_u32(&mut data, tag);
    let body_length = u16::try_from(data.len() - 8).unwrap();
    LittleEndian::write_u16(&mut data[4..], body_length);
    LittleEndian::write_u16(&mut data[8..], offset);
    LittleEndian::write_u16(&mut data[10..], u16::try_from(name.len() * 2).unwrap());
    for (chunk, unit) in data[8 + base + usize::from(offset)..]
        .chunks_exact_mut(2)
        .zip(name)
    {
        LittleEndian::write_u16(chunk, *unit);
    }
    data
}

#[test]
fn substitute_name_offsets_and_prefix_policy() {
    for tag in [IO_REPARSE_TAG_SYMLINK, IO_REPARSE_TAG_MOUNT_POINT] {
        for offset in [0, 4, 12] {
            for (input, expected) in [
                (vec![], vec![]),
                (vec![92, 63, 63, 92], vec![92, 63, 63, 92]),
                (vec![92, 63, 63, 92, 67], vec![92, 92, 63, 92, 67]),
                (
                    vec![97, 92, 0xd800, 0, 0xdc00],
                    vec![97, 92, 0xd800, 0, 0xdc00],
                ),
            ] {
                let parsed = target(&packet(tag, &input, offset)).unwrap();
                assert_eq!(
                    parsed.as_os_str().encode_wide().collect::<Vec<_>>(),
                    expected
                );
            }
        }
    }
    assert_eq!(
        PathQueryError::NotSymbolicLink.to_string(),
        "not a symbolic link"
    );
}

#[test]
fn malformed_payloads_fail_without_out_of_bounds_reads() {
    assert_eq!(target(&[0; 8]), Err(PathQueryError::NotSymbolicLink));
    let valid = packet(IO_REPARSE_TAG_SYMLINK, &[65, 66], 0);
    for size in 0..valid.len() {
        assert_eq!(target(&valid[..size]), Err(INVALID_DATA), "size={size}");
    }
    let mut short = valid.clone();
    LittleEndian::write_u16(&mut short[4..], 2);
    assert_eq!(target(&short), Err(INVALID_DATA));
    for (offset, length) in [(1, 2), (0, 1), (u16::MAX - 1, 2), (0, 100)] {
        let mut invalid = valid.clone();
        LittleEndian::write_u16(&mut invalid[8..], offset);
        LittleEndian::write_u16(&mut invalid[10..], length);
        assert_eq!(target(&invalid), Err(INVALID_DATA));
    }
}
