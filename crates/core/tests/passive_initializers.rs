use std::{path::PathBuf, process::Command};

use serde::Deserialize;

#[derive(Debug, Deserialize, PartialEq)]
struct Snapshot {
    path: String,
    sha256: String,
    doc: Option<String>,
    body: Vec<String>,
    code_names: Vec<String>,
    public_names: Vec<String>,
    has_all: bool,
}

struct Expected {
    path: &'static str,
    sha256: &'static str,
    doc: Option<&'static str>,
}

const EXPECTED: &[Expected] = &[
    Expected {
        path: "qlib/cli/__init__.py",
        sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        doc: None,
    },
    Expected {
        path: "qlib/contrib/__init__.py",
        sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        doc: None,
    },
    Expected {
        path: "qlib/contrib/data/__init__.py",
        sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        doc: None,
    },
    Expected {
        path: "qlib/contrib/data/utils/__init__.py",
        sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        doc: None,
    },
    Expected {
        path: "qlib/contrib/eva/__init__.py",
        sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        doc: None,
    },
    Expected {
        path: "qlib/contrib/online/__init__.py",
        sha256: "b11f28b0f1acfdaf07b16d8867d41142e3546de0a05cd3e3b1b448100ae6c5c5",
        doc: Some(
            "\nTODO:\n\n- Online needs that the model have such method\n    def get_data_with_date(self, date, **kwargs):\n        \"\"\"\n        Will be called in online module\n        need to return the data that used to predict the label (score) of stocks at date.\n\n        :param\n            date: pd.Timestamp\n                predict date\n        :return:\n            data: the input data that used to predict the label (score) of stocks at predict date.\n        \"\"\"\n        raise NotImplementedError(\"get_data_with_date for this model is not implemented.\")\n\n",
        ),
    },
    Expected {
        path: "qlib/contrib/ops/__init__.py",
        sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        doc: None,
    },
    Expected {
        path: "qlib/contrib/report/data/__init__.py",
        sha256: "76d04ad58f72a2af9fa69655f11e7d066179dd419314af6042b403735984cf52",
        doc: Some("\nThis module is designed to analysis data\n\n"),
    },
    Expected {
        path: "qlib/contrib/rolling/__init__.py",
        sha256: "8465d0f61d09c1b9903883e1ac1d37b136ac8b80914fc319653cb2dd42e3ea6a",
        doc: Some(
            "\nThe difference between me and the scripts in examples/benchmarks/benchmarks_dynamic\n- This module only focus provide a general rolling implementation.\n  Anything specific that benchmark is placed in examples/benchmarks/benchmarks_dynamic\n",
        ),
    },
    Expected {
        path: "qlib/contrib/tuner/__init__.py",
        sha256: "9b3a58b524a2277140e3ed6aa3b1201766c83e2537dd38b3d05ca4d2f2bddce3",
        doc: None,
    },
    Expected {
        path: "qlib/data/_libs/__init__.py",
        sha256: "6823abeac12c429bfdcf8709816140e2a3ace3126a725a98627c4e55cc622b64",
        doc: None,
    },
    Expected {
        path: "qlib/model/ens/__init__.py",
        sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        doc: None,
    },
    Expected {
        path: "qlib/model/interpret/__init__.py",
        sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        doc: None,
    },
    Expected {
        path: "qlib/rl/contrib/__init__.py",
        sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        doc: None,
    },
    Expected {
        path: "qlib/rl/data/__init__.py",
        sha256: "d7f9c5a9f94fb87897efa816406e892f9531968c75bc0c60d33e1a0d02abd460",
        doc: Some(
            "Common utilities to handle ad-hoc-styled data.\n\nMost of these snippets comes from research project (paper code).\nPlease take caution when using them in production.\n",
        ),
    },
    Expected {
        path: "qlib/strategy/__init__.py",
        sha256: "6823abeac12c429bfdcf8709816140e2a3ace3126a725a98627c4e55cc622b64",
        doc: None,
    },
    Expected {
        path: "qlib/workflow/online/__init__.py",
        sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        doc: None,
    },
    Expected {
        path: "qlib/workflow/task/__init__.py",
        sha256: "ec39e452db5bf74426ae5e824f652c69de5251dfaf89c1742880c38f9fdf57a6",
        doc: Some(
            "\nTask related workflow is implemented in this folder\n\nA typical task workflow\n\n| Step                  | Description                                    |\n|-----------------------+------------------------------------------------|\n| TaskGen               | Generating tasks.                              |\n| TaskManager(optional) | Manage generated tasks                         |\n| run task              | retrieve  tasks from TaskManager and run tasks. |\n",
        ),
    },
];

#[test]
fn passive_package_initializers_have_no_runtime_surface_or_side_effects() {
    let root = std::env::var_os("QLIB_PYTHON_ROOT").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../qlib"),
        PathBuf::from,
    );
    assert!(
        root.join("qlib").is_dir(),
        "Qlib source not found: {}",
        root.display()
    );

    let script = r#"
import ast
import hashlib
import importlib.util
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
result = []
for relative in sys.argv[2:]:
    path = root / relative
    raw = path.read_bytes()
    tree = ast.parse(raw, filename=str(path))
    code = compile(tree, str(path), "exec")
    spec = importlib.util.spec_from_file_location("passive_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result.append({
        "path": relative,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "doc": module.__doc__,
        "body": [type(node).__name__ for node in tree.body],
        "code_names": list(code.co_names),
        "public_names": [name for name in vars(module) if not name.startswith("_")],
        "has_all": hasattr(module, "__all__"),
    })
print(json.dumps(result, ensure_ascii=False))
"#;
    let python = std::env::var_os("PYTHON").unwrap_or_else(|| "python".into());
    let output = Command::new(python)
        .arg("-c")
        .arg(script)
        .arg(&root)
        .args(EXPECTED.iter().map(|item| item.path))
        .output()
        .expect("Python interpreter starts");
    assert!(
        output.status.success(),
        "passive-module snapshot failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let actual: Vec<Snapshot> =
        serde_json::from_slice(&output.stdout).expect("Python returns a JSON snapshot");
    assert_eq!(actual.len(), EXPECTED.len());
    for (snapshot, expected) in actual.iter().zip(EXPECTED) {
        assert_eq!(snapshot.path, expected.path);
        assert_eq!(snapshot.sha256, expected.sha256);
        assert_eq!(snapshot.doc.as_deref(), expected.doc);
        assert!(snapshot.public_names.is_empty());
        assert!(!snapshot.has_all);
        if expected.doc.is_some() {
            assert_eq!(snapshot.body, ["Expr"]);
            assert_eq!(snapshot.code_names, ["__doc__"]);
        } else {
            assert!(snapshot.body.is_empty());
            assert!(snapshot.code_names.is_empty());
        }
    }
}
