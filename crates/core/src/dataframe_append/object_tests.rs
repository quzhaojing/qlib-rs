use super::*;

struct FailObjects {
    fail_cast: bool,
    calls: Vec<&'static str>,
}
impl ObjectOperations for FailObjects {
    fn cast_numeric(&mut self, array: &ArrayRef, dtype: &DataType) -> Result<ArrayRef, ArrowError> {
        self.calls.push("cast");
        if self.fail_cast {
            Err(ArrowError::ComputeError("injected object cast".into()))
        } else {
            ArrowObjects.cast_numeric(array, dtype)
        }
    }
    fn build(
        &mut self,
        _: UnionFields,
        _: Vec<i8>,
        _: Vec<ArrayRef>,
    ) -> Result<ArrayRef, ArrowError> {
        self.calls.push("build");
        Err(ArrowError::ComputeError(
            "injected object construction".into(),
        ))
    }
}

#[test]
fn object_construction_errors_propagate_without_partial_results() {
    let array = Arc::new(arrow_array::Int8Array::from(vec![-1, 1])) as ArrayRef;
    for fail_cast in [true, false] {
        let mut operations = FailObjects {
            fail_cast,
            calls: vec![],
        };
        let error = box_with(&array, &mut operations).unwrap_err();
        if fail_cast {
            assert!(error.to_string().ends_with("injected object cast"));
            assert_eq!(operations.calls, ["cast"]);
        } else {
            assert!(error.to_string().ends_with("injected object construction"));
            assert_eq!(operations.calls, ["cast", "build"]);
        }
        assert_eq!(
            array
                .as_any()
                .downcast_ref::<arrow_array::Int8Array>()
                .unwrap()
                .values()
                .as_ref(),
            &[-1, 1]
        );
    }
    let boxed = numeric_object_array(&array).unwrap();
    let mut operations = FailObjects {
        fail_cast: true,
        calls: vec![],
    };
    assert!(Arc::ptr_eq(
        &box_with(&boxed, &mut operations).unwrap(),
        &boxed
    ));
    assert!(operations.calls.is_empty());
}
