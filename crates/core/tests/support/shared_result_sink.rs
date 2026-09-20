use domain_core::nested_executor::SharedNestedResult;
use domain_core::{NestedExecutorReturnSink, NestedExecutorReturnSinkError};
use std::sync::{Arc, Mutex};

pub struct ClearingResultSink {
    pub retained: Arc<Mutex<Option<SharedNestedResult>>>,
    pub fail: bool,
}

impl NestedExecutorReturnSink for ClearingResultSink {
    fn store_execute_result(
        &mut self,
        executions: &SharedNestedResult,
    ) -> Result<(), NestedExecutorReturnSinkError> {
        {
            let mut rows = executions
                .try_lock()
                .expect("sink called without framework guard");
            assert!(!rows.is_empty());
            rows.clear();
        }
        *self.retained.lock().unwrap() = Some(Arc::clone(executions));
        if self.fail {
            Err(NestedExecutorReturnSinkError {
                message: "sink after clear".into(),
            })
        } else {
            Ok(())
        }
    }
}
