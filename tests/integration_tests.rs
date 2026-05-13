//! Integration tests for the `esm` CLI.
//!
//! Ported from `test/integration_tests.jl`. Each test stages the relevant
//! fixtures into a tempdir and invokes the compiled binary, asserting on the
//! files produced. Hash-based assertions from the Julia suite are replaced
//! with structural checks (parsed JSON / file presence) so the tests aren't
//! sensitive to incidental formatting differences between the two ports.

use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

fn esm_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_esm"))
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn fixture(name: &str) -> PathBuf {
    repo_root().join("test").join("inputs").join(name)
}

fn run_esm(args: &[&str], cwd: Option<&Path>) -> std::process::Output {
    let mut cmd = Command::new(esm_bin());
    cmd.args(args);
    // example.xlsx references "$GITHUB_WORKSPACE/test/inputs/pr_folder"; make
    // sure the variable resolves to the repo root.
    cmd.env("GITHUB_WORKSPACE", repo_root());
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    let output = cmd.output().expect("failed to spawn esm");
    if !output.status.success() {
        panic!(
            "esm {:?} failed (status={:?}):\n--- stdout ---\n{}\n--- stderr ---\n{}",
            args,
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
    output
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir_recursive(&from, &to);
        } else {
            std::fs::copy(&from, &to).unwrap();
        }
    }
}

fn staged(dir: &TempDir, name: &str) -> PathBuf {
    let dest = dir.path().join(name);
    std::fs::copy(fixture(name), &dest).unwrap();
    dest
}

// ---------------------------------------------------------------------------
// Template
// ---------------------------------------------------------------------------

#[test]
fn template_integration() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();

    let out1 = d.join("tmp.xlsx");
    run_esm(
        &["template", "--output-path", out1.to_str().unwrap()],
        None,
    );
    assert!(out1.is_file(), "expected {:?}", out1);

    let out2 = d.join("t2.xlsx");
    run_esm(&["template", "-o", out2.to_str().unwrap()], None);
    assert!(out2.is_file(), "expected {:?}", out2);

    // Default output is `ESM.xlsx` relative to cwd.
    let cwd = tempfile::tempdir().unwrap();
    run_esm(&["template"], Some(cwd.path()));
    assert!(cwd.path().join("ESM.xlsx").is_file());
}

// ---------------------------------------------------------------------------
// Translate
// ---------------------------------------------------------------------------

#[test]
fn translate_integration() {
    let dir = tempfile::tempdir().unwrap();
    let out_esm = dir.path().join("tmp.esm");
    run_esm(
        &[
            "translate",
            fixture("example.xlsx").to_str().unwrap(),
            out_esm.to_str().unwrap(),
        ],
        None,
    );
    assert!(out_esm.is_file());

    let contents = std::fs::read_to_string(&out_esm).unwrap();
    let json: serde_json::Value = serde_json::from_str(&contents).unwrap();

    // Structural assertions matching the contents of example.xlsx
    let samples = json["samples"].as_object().expect("samples object");
    assert!(!samples.is_empty(), "no samples written");

    let groups = json["groups"].as_object().expect("groups object");
    assert!(groups.contains_key("plate_01"));
    assert!(groups.contains_key("first_group"));
    assert!(groups.contains_key("second_group"));
    assert!(groups.contains_key("third_group"));

    let transformations =
        json["transformations"].as_object().expect("transformations object");
    assert!(transformations.contains_key("flow_sub"));
    assert!(transformations.contains_key("od_sub"));

    let views = json["views"].as_object().expect("views object");
    for name in ["flowsub", "group1", "group2", "group3", "mega", "odsub", "sample"] {
        assert!(views.contains_key(name), "missing view: {}", name);
    }
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

const EXAMPLE_VIEWS: &[&str] = &[
    "flowsub.csv",
    "group1.csv",
    "group2.csv",
    "group3.csv",
    "mega.csv",
    "odsub.csv",
    "sample.csv",
];

fn translate_example(out_dir: &Path) -> PathBuf {
    let out_esm = out_dir.join("tmp.esm");
    run_esm(
        &[
            "translate",
            fixture("example.xlsx").to_str().unwrap(),
            out_esm.to_str().unwrap(),
        ],
        None,
    );
    out_esm
}

#[test]
fn views_all_views() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let esm = translate_example(d);
    run_esm(
        &[
            "views",
            esm.to_str().unwrap(),
            "--output-dir",
            d.to_str().unwrap(),
        ],
        None,
    );
    for name in EXAMPLE_VIEWS {
        assert!(d.join(name).is_file(), "missing view: {}", name);
    }

    // Cleanup and rerun with the short flag form.
    for name in EXAMPLE_VIEWS {
        std::fs::remove_file(d.join(name)).unwrap();
    }
    run_esm(
        &["views", esm.to_str().unwrap(), "-o", d.to_str().unwrap()],
        None,
    );
    for name in EXAMPLE_VIEWS {
        assert!(d.join(name).is_file(), "missing view (short flag): {}", name);
    }
}

#[test]
fn views_specific_view() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let esm = translate_example(d);

    let out_dir = tempfile::tempdir().unwrap();
    let od = out_dir.path();
    run_esm(
        &[
            "views",
            esm.to_str().unwrap(),
            "--view",
            "mega",
            "--output-dir",
            od.to_str().unwrap(),
        ],
        None,
    );
    assert!(od.join("mega.csv").is_file());
    for name in EXAMPLE_VIEWS {
        if *name == "mega.csv" {
            continue;
        }
        assert!(!od.join(name).exists(), "should not have produced: {}", name);
    }

    // Same again using short flag forms.
    let out_dir2 = tempfile::tempdir().unwrap();
    let od2 = out_dir2.path();
    run_esm(
        &[
            "views",
            esm.to_str().unwrap(),
            "-v",
            "mega",
            "-o",
            od2.to_str().unwrap(),
        ],
        None,
    );
    assert!(od2.join("mega.csv").is_file());

    // The produced mega.csv should be deterministic across runs.
    let first = std::fs::read(od.join("mega.csv")).unwrap();
    let second = std::fs::read(od2.join("mega.csv")).unwrap();
    assert_eq!(first, second, "mega.csv differs between flag forms");
}

// ---------------------------------------------------------------------------
// Summarise
// ---------------------------------------------------------------------------

#[test]
fn summarise_esm() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let path = staged(&dir, "summarise.esm");

    run_esm(&["summarise", path.to_str().unwrap(), "--plot"], None);
    let pdf = d.join("summarise.esm.pdf");
    assert!(pdf.is_file());
    std::fs::remove_file(&pdf).unwrap();

    run_esm(&["summarise", path.to_str().unwrap(), "-p"], None);
    assert!(pdf.is_file());
}

#[test]
fn summarise_fcs() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let path = staged(&dir, "small.fcs");

    run_esm(&["summarise", path.to_str().unwrap()], None);
    assert!(!d.join("small.fcs.pdf").exists());
    assert!(!d.join("small.fcs.csv").exists());

    run_esm(
        &["summarise", path.to_str().unwrap(), "--plot", "--csv"],
        None,
    );
    assert!(d.join("small.fcs.pdf").is_file());
    assert!(d.join("small.fcs.csv").is_file());
}

#[test]
fn summarise_spectramax() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let path = staged(&dir, "spectramax-summarise.txt");

    run_esm(
        &["summarise", path.to_str().unwrap(), "--type", "spectramax"],
        None,
    );
    assert!(!d.join("spectramax-summarise.txt.pdf").exists());

    run_esm(
        &[
            "summarise",
            path.to_str().unwrap(),
            "-t",
            "spectramax",
            "--plot",
            "--csv",
        ],
        None,
    );
    assert!(d.join("spectramax-summarise.txt.pdf").is_file());
    for channel in ["600", "700", "535_485"] {
        let f = d.join(format!("spectramax-summarise.txt_{}.csv", channel));
        assert!(f.is_file(), "missing {:?}", f);
    }
}

#[test]
fn summarise_biotek() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let path = staged(&dir, "biotek-summarise.csv");

    run_esm(
        &["summarise", path.to_str().unwrap(), "--type", "biotek"],
        None,
    );
    assert!(!d.join("biotek-summarise.csv.pdf").exists());

    run_esm(
        &[
            "summarise",
            path.to_str().unwrap(),
            "-t",
            "biotek",
            "-p",
            "-c",
        ],
        None,
    );
    assert!(d.join("biotek-summarise.csv.pdf").is_file());
    for channel in ["OD_600", "OD_700", "GFP_485_530"] {
        let f = d.join(format!("biotek-summarise.csv_{}.csv", channel));
        assert!(f.is_file(), "missing {:?}", f);
    }
}

#[test]
fn summarise_tecan() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let path = staged(&dir, "tecan-summarise.xlsx");

    run_esm(
        &["summarise", path.to_str().unwrap(), "--type", "tecan"],
        None,
    );
    assert!(!d.join("tecan-summarise.xlsx.pdf").exists());

    run_esm(
        &[
            "summarise",
            path.to_str().unwrap(),
            "-t",
            "tecan",
            "-p",
            "--csv",
        ],
        None,
    );
    assert!(d.join("tecan-summarise.xlsx.pdf").is_file());
    for channel in ["OD_600", "OD_700", "GFP"] {
        let f = d.join(format!("tecan-summarise.xlsx_{}.csv", channel));
        assert!(f.is_file(), "missing {:?}", f);
    }
}

#[test]
fn summarise_bmg() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let path = staged(&dir, "bmg-summarise.csv");

    run_esm(
        &["summarise", path.to_str().unwrap(), "--type", "bmg"],
        None,
    );
    assert!(!d.join("bmg-summarise.csv.pdf").exists());

    run_esm(
        &[
            "summarise",
            path.to_str().unwrap(),
            "-t",
            "bmg",
            "-p",
            "--csv",
        ],
        None,
    );
    assert!(d.join("bmg-summarise.csv.pdf").is_file());
    for channel in ["ABS_600_0_nm", "ABS_700_0_nm", "FI_YFP_pAN1717"] {
        let f = d.join(format!("bmg-summarise.csv_{}.csv", channel));
        assert!(f.is_file(), "missing {:?}", f);
    }
}

#[test]
fn summarise_pr_folder() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let dest = d.join("pr_folder");
    copy_dir_recursive(&fixture("pr_folder"), &dest);

    run_esm(&["summarise", dest.to_str().unwrap()], None);
    assert!(!d.join("pr_folder.pdf").exists());

    run_esm(&["summarise", dest.to_str().unwrap(), "-p", "-c"], None);
    assert!(d.join("pr_folder.pdf").is_file());
    assert!(d.join("pr_folder_OD.csv").is_file());
    assert!(d.join("pr_folder_flo.csv").is_file());
}
