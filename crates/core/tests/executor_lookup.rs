use std::{
    cell::{Cell, RefCell},
    path::PathBuf,
    process::Command,
};

use domain_core::executor_lookup::{ExecutorLookup, SimulatorLookupError, get_simulator_executor};
use serde_json::{Value, json};

struct Node<'a> {
    child: Option<&'a Node<'a>>,
    simulator: Option<&'a Cell<i32>>,
    failure: Option<&'a str>,
    name: String,
    events: &'a RefCell<Vec<String>>,
}

impl ExecutorLookup<Cell<i32>, &'static str> for Node<'_> {
    fn inner_executor(
        &self,
    ) -> Result<Option<&dyn ExecutorLookup<Cell<i32>, &'static str>>, &'static str> {
        if self.failure.is_some() {
            self.events.borrow_mut().push(self.name.clone());
            return Err("child unavailable");
        }
        Ok(self.child.map(|child| {
            self.events.borrow_mut().push(self.name.clone());
            child as &dyn ExecutorLookup<Cell<i32>, &'static str>
        }))
    }

    fn as_simulator(&self) -> Option<&Cell<i32>> {
        self.simulator
    }
}

fn native_case(depth: usize, kind: &str) -> Value {
    let events = RefCell::new(Vec::new());
    let simulator = Cell::new(7);
    let leaf = Node {
        child: None,
        simulator: Some(&simulator),
        failure: None,
        name: String::new(),
        events: &events,
    };
    let terminal = Node {
        child: (kind == "both").then_some(&leaf),
        simulator: (kind != "invalid").then_some(&simulator),
        failure: (kind == "failure").then_some("child unavailable"),
        name: kind.to_owned(),
        events: &events,
    };
    // Arena slots keep references stable without a recursively owned chain.
    let slots: Vec<std::cell::OnceCell<Node<'_>>> =
        (0..depth).map(|_| std::cell::OnceCell::new()).collect();
    let mut root = &terminal;
    for (i, slot) in slots.iter().enumerate() {
        root = slot.get_or_init(|| Node {
            child: Some(root),
            simulator: None,
            failure: None,
            name: i.to_string(),
            events: &events,
        });
    }
    let (status, message) = match get_simulator_executor(root) {
        Ok(result) => {
            assert!(std::ptr::eq(result, &raw const simulator));
            result.set(19);
            assert_eq!(simulator.get(), 19);
            ("same", String::new())
        }
        Err(error @ SimulatorLookupError::Access(_)) => {
            assert_eq!(error, SimulatorLookupError::Access("child unavailable"));
            assert!(format!("{error:?}").contains("Access"));
            ("access", error.to_string())
        }
        Err(error @ SimulatorLookupError::NotSimulator) => {
            assert_eq!(error, SimulatorLookupError::NotSimulator);
            assert!(format!("{error:?}").contains("NotSimulator"));
            ("AssertionError", error.to_string())
        }
    };
    json!({"depth": depth, "kind": kind, "events": events.into_inner(), "status": status, "message": message})
}

#[test]
fn lookup_matches_live_python_identity_order_subclasses_and_failures() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = std::env::var_os("QLIB_PYTHON_RL_EXECUTION_UTILS").map_or_else(
        || root.join("../../../qlib/qlib/rl/order_execution/utils.py"),
        PathBuf::from,
    );
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg(root.join("tests/fixtures/simulator_executor_lookup.py"))
        .arg(source)
        .output()
        .expect("Python starts");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    let actual: Vec<Value> = [0, 1, 7, 1024]
        .into_iter()
        .flat_map(|depth| {
            ["simulator", "derived", "invalid", "failure", "both"]
                .map(|kind| native_case(depth, kind))
        })
        .collect();
    assert_eq!(actual, expected);
}
