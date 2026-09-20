use std::{mem::size_of, path::PathBuf, process::Command};

use domain_core::SingleOrderStrategy;
use serde::Deserialize;

#[test]
fn crate_root_exposes_single_order_strategy() {
    assert_ne!(size_of::<SingleOrderStrategy>(), 0);
}

#[derive(Debug, Deserialize, PartialEq)]
struct PythonSnapshot {
    source_sha256: String,
    module_doc: Option<String>,
    public_names: Vec<String>,
    body_kinds: Vec<String>,
    import_module: String,
    import_level: usize,
    import_names: Vec<String>,
    exported_names: Vec<String>,
    identity_name: String,
    identity_module: String,
    facts: Vec<String>,
}

#[test]
fn live_python_strategy_initializer_freezes_reexport_identity_and_order() {
    assert_eq!(live_python_snapshot(), expected_python_snapshot());
}

fn live_python_snapshot() -> PythonSnapshot {
    let source = std::env::var_os("QLIB_PYTHON_RL_STRATEGY_INIT").map_or_else(
        || {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../qlib/qlib/rl/strategy/__init__.py")
        },
        PathBuf::from,
    );
    assert!(
        source.is_file(),
        "Python source not found: {}",
        source.display()
    );
    let output = Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
        .arg("-c")
        .arg(PYTHON_SNAPSHOT)
        .arg(&source)
        .output()
        .expect("Python interpreter starts");
    assert!(
        output.status.success(),
        "RL strategy package snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("Python returns a JSON snapshot")
}

fn expected_python_snapshot() -> PythonSnapshot {
    PythonSnapshot {
        source_sha256: "a1cbc00cfe589a4150a3e803ada2a0188d996c3e971ede3e078e42cf4d961252"
            .to_owned(),
        module_doc: None,
        public_names: vec!["SingleOrderStrategy".to_owned()],
        body_kinds: ["ImportFrom", "Assign"].map(str::to_owned).to_vec(),
        import_module: "single_order".to_owned(),
        import_level: 1,
        import_names: vec!["SingleOrderStrategy".to_owned()],
        exported_names: vec!["SingleOrderStrategy".to_owned()],
        identity_name: "SingleOrderStrategy".to_owned(),
        identity_module: "qlib.rl.strategy.single_order".to_owned(),
        facts: [
            "all-is-independent-list",
            "all-matches-public-order",
            "import-preserves-object-identity",
            "no-additional-public-name",
        ]
        .map(str::to_owned)
        .to_vec(),
    }
}

const PYTHON_SNAPSHOT: &str = r#"
import ast
import hashlib
import json
import sys
import types

path = sys.argv[1]
raw = open(path, "rb").read()
tree = ast.parse(raw, filename=path)
package = types.ModuleType("qlib.rl.strategy")
package.__file__ = path
package.__package__ = "qlib.rl.strategy"
package.__path__ = []
dependency = types.ModuleType("qlib.rl.strategy.single_order")
sentinel = type("SingleOrderStrategy", (), {"__module__": dependency.__name__})
dependency.SingleOrderStrategy = sentinel
sys.modules[package.__name__] = package
sys.modules[dependency.__name__] = dependency
exec(compile(raw, path, "exec"), package.__dict__)

public_names = [name for name in vars(package) if not name.startswith("_")]
facts = []
if package.__all__ is not public_names:
    facts.append("all-is-independent-list")
if package.__all__ == public_names:
    facts.append("all-matches-public-order")
if package.SingleOrderStrategy is sentinel:
    facts.append("import-preserves-object-identity")
if public_names == ["SingleOrderStrategy"]:
    facts.append("no-additional-public-name")

import_node = next(node for node in tree.body if isinstance(node, ast.ImportFrom))
print(json.dumps({
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "module_doc": package.__doc__,
    "public_names": public_names,
    "body_kinds": [type(node).__name__ for node in tree.body],
    "import_module": import_node.module,
    "import_level": import_node.level,
    "import_names": [alias.name for alias in import_node.names],
    "exported_names": package.__all__,
    "identity_name": package.SingleOrderStrategy.__name__,
    "identity_module": package.SingleOrderStrategy.__module__,
    "facts": facts,
}, ensure_ascii=False))
"#;
