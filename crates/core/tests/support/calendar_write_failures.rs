use super::*;

struct Sink {
    bytes: Vec<u8>,
    fail_at: Option<usize>,
    flush_error: bool,
    flushed: bool,
}

impl Write for Sink {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.fail_at == Some(self.bytes.len()) {
            return Err(std::io::Error::other("write failed"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.flushed = true;
        if self.flush_error {
            Err(std::io::Error::other("flush failed"))
        } else {
            Ok(())
        }
    }
}

#[test]
fn writer_errors_retain_prefix_and_flush_overrides_prior_failure() {
    let array = CalendarTextArray {
        shape: vec![1, 2],
        values: vec!["a".into(), "b".into()],
    };
    for fail_at in [None, Some(0), Some(1), Some(2), Some(3)] {
        for flush_error in [false, true] {
            let mut sink = Sink {
                bytes: vec![],
                fail_at,
                flush_error,
                flushed: false,
            };
            let result = write_and_flush(&array, &mut sink);
            let error = if flush_error {
                Some("flush failed")
            } else {
                fail_at.map(|_| "write failed")
            };
            assert_eq!(
                result,
                error.map_or(Ok(()), |text| Err(CalendarLoadError::Other(text.into())))
            );
            assert_eq!(sink.bytes, b"a b\n"[..fail_at.unwrap_or(4)]);
            assert!(sink.flushed);
        }
    }
    for shape in [vec![], vec![1, 1, 1], vec![2]] {
        let array = CalendarTextArray {
            shape,
            values: vec!["x".into()],
        };
        let mut sink = Sink {
            bytes: vec![],
            fail_at: None,
            flush_error: false,
            flushed: false,
        };
        assert!(matches!(
            write_and_flush(&array, &mut sink),
            Err(CalendarLoadError::Value(_))
        ));
        assert!(sink.bytes.is_empty());
        assert!(sink.flushed);
    }
}
