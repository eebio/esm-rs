//! Port of `test/write_tests.jl`.

use std::path::{Path, PathBuf};

use esm::dataframe::Value;
use esm::esm_files::{read_esm, sample_names, get_sample_values};
use esm::read_data::{read_data, write_esm_data};
use esm::utils::{expand_group, expand_groups};
use serde_json::Value as JsonValue;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn fixture(name: &str) -> PathBuf {
    repo_root().join("test").join("inputs").join(name)
}

fn with_workspace<F: FnOnce() -> R, R>(f: F) -> R {
    unsafe { std::env::set_var("GITHUB_WORKSPACE", repo_root()); }
    f()
}

#[test]
fn read_data_example() {
    let es = with_workspace(|| read_data(fixture("example.xlsx").to_str().unwrap()).unwrap());

    let samples = es["samples"].as_object().unwrap();
    let p02 = &samples["plate_02_a1"];

    let fsc_h: Vec<f64> = p02["values"]["FSC_H"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(fsc_h, vec![628.0, 1023.0, 373.0, 1023.0]);

    let fl4_h: Vec<f64> = p02["values"]["FL4_H"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let expected = [28.133175, 310.590027, 3.819718, 2414.418213];
    for (g, e) in fl4_h.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-5, "got {} expected {}", g, e);
    }

    assert_eq!(p02["type"], JsonValue::String("population".into()));
    assert_eq!(p02["metadata"]["FL1_H"]["range"], JsonValue::String("1024".into()));
    assert_eq!(p02["metadata"]["FL1_H"]["amp_type"], JsonValue::String("0,0".into()));

    let groups = es["groups"].as_object().unwrap();
    let first: Vec<String> = groups["first_group"]["sample_IDs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(first, vec!["plate_01_A1", "plate_01_A5", "plate_01_A9"]);

    let second: Vec<String> = groups["second_group"]["sample_IDs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(second, vec!["plate_01_A3", "plate_01_A8", "plate_01_A7"]);

    let third: Vec<String> = groups["third_group"]["sample_IDs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(third, vec!["plate_01_A1", "plate_01_A2", "plate_01_A3"]);
}

#[test]
fn read_data_bad_instrument_type() {
    let err = with_workspace(|| {
        read_data(fixture("bad_inst_type.xlsx").to_str().unwrap()).unwrap_err()
    });
    let msg = err.to_string();
    assert!(
        msg.to_lowercase().contains("gibberish") || msg.to_lowercase().contains("unknown instrument"),
        "expected error about unknown instrument type, got: {}",
        msg
    );
}

#[test]
fn write_esm_roundtrip() {
    let es = with_workspace(|| read_data(fixture("example.xlsx").to_str().unwrap()).unwrap());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tmp.esm");
    write_esm_data(&es, path.to_str().unwrap()).unwrap();

    let written = read_esm(path.to_str().unwrap()).unwrap();

    // names: every `sample.channel` from the source appears in the written ESM
    let mut expected_names: Vec<String> = Vec::new();
    for (sid, sample) in es["samples"].as_object().unwrap() {
        for ch in sample["values"].as_object().unwrap().keys() {
            expected_names.push(format!("{}.{}", sid.to_lowercase(), ch));
        }
    }
    let got_names = sample_names(&written);
    assert_eq!(
        sorted(&expected_names),
        sorted(&got_names),
        "round-tripped sample names differ"
    );

    // channels
    let channel_col = written.samples.column("channel").unwrap();
    let mut channels: Vec<String> = channel_col
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    channels.sort();
    channels.dedup();
    let mut expected_channels: Vec<String> = [
        "OD", "flo", "FL1_H", "SSC_H", "FL3_H", "FL1_A", "FL4_H", "FSC_H",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    expected_channels.sort();
    assert_eq!(channels, expected_channels);

    // type: every non-(OD|flo) row is "population"
    let name_col = written.samples.column("name").unwrap();
    let type_col = written.samples.column("type").unwrap();
    let mut pop_count = 0;
    for (n, t) in name_col.iter().zip(type_col.iter()) {
        let name = n.as_str().unwrap_or("");
        if name.contains("OD") || name.contains("flo") {
            continue;
        }
        assert_eq!(t.as_str().unwrap_or(""), "population");
        pop_count += 1;
    }
    assert_eq!(pop_count, 6);

    // values: plate_01_b3.flo should match the source
    let target = "plate_01_b3.flo";
    let row_idx = name_col
        .iter()
        .position(|v| v.as_str() == Some(target))
        .expect("plate_01_b3.flo not in written samples");
    let got_values = get_sample_values(&written, row_idx);
    let expected_values: Vec<f64> = es["samples"]["plate_01_b3"]["values"]["flo"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(got_values, expected_values);

    // metadata keys for FCS rows
    use std::collections::BTreeSet;
    let expected_meta_keys: BTreeSet<String> = [
        "range", "ex_pow", "filter", "det_volt", "amp_type", "ex_wav",
        "amp_gain", "name_s", "name", "det_type", "perc_em", "raw_metadata",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let metadata_col = written.samples.column("metadata").unwrap();
    for (n, m) in name_col.iter().zip(metadata_col.iter()) {
        let name = n.as_str().unwrap_or("");
        if name.contains("OD") || name.contains("flo") {
            continue;
        }
        let meta_str = m.as_str().unwrap_or("{}");
        let meta: serde_json::Value = serde_json::from_str(meta_str).unwrap();
        let keys: BTreeSet<String> = meta
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(expected_meta_keys, keys, "metadata keys differ for {}", name);
    }

    // groups
    let group_col = written.groups.column("group").unwrap();
    let groups: Vec<String> = group_col
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    let expected_groups = [
        "plate_01", "plate_02", "first_group", "second_group", "third_group",
    ];
    for g in &expected_groups {
        assert!(groups.iter().any(|x| x == g), "missing group {}", g);
    }

    // plate_02 sample IDs and metadata
    let plate02_idx = group_col
        .iter()
        .position(|v| v.as_str() == Some("plate_02"))
        .unwrap();
    let ids_str = written
        .groups
        .column("sample_IDs")
        .unwrap()
        .get(plate02_idx)
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    let ids: Vec<String> = serde_json::from_str(&ids_str).unwrap();
    assert_eq!(ids, vec!["plate_02_a1".to_string()]);

    let meta_str = written
        .groups
        .column("metadata")
        .unwrap()
        .get(plate02_idx)
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();
    let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap();
    assert_eq!(meta["autodefined"], JsonValue::String("true".into()));

    // transformations
    let trans_keys: Vec<String> = written.transformations.keys().cloned().collect();
    for k in ["flow_sub", "od_sub"] {
        assert!(trans_keys.iter().any(|x| x == k), "missing transformation {}", k);
    }
    assert_eq!(
        written.transformations["flow_sub"]["equation"],
        "hcat(first_group.flo,second_group.flo).-colmean(third_group.flo)"
    );

    // views
    let view_keys: Vec<String> = written.views.keys().cloned().collect();
    for k in ["flowsub", "mega", "group2", "sample", "odsub", "group1", "group3"] {
        assert!(view_keys.iter().any(|x| x == k), "missing view {}", k);
    }
    assert_eq!(
        written.views["mega"]["data"],
        vec!["first_group".to_string(), "second_group".to_string()]
    );
}

fn sorted(v: &[String]) -> Vec<String> {
    let mut out = v.to_vec();
    out.sort();
    out
}

#[test]
fn expand_group_cases() {
    assert_eq!(
        expand_group("plate_01_a[1:3]"),
        vec!["plate_01_a1", "plate_01_a2", "plate_01_a3"]
    );
    assert_eq!(
        expand_group("plate_01_a[2:4]"),
        vec!["plate_01_a2", "plate_01_a3", "plate_01_a4"]
    );
    assert_eq!(
        expand_group("plate_01_[a:d]1"),
        vec!["plate_01_a1", "plate_01_b1", "plate_01_c1", "plate_01_d1"]
    );
    assert_eq!(
        expand_group("plate_01_[a:c]1[2:3]"),
        vec![
            "plate_01_a12",
            "plate_01_b12",
            "plate_01_c12",
            "plate_01_a13",
            "plate_01_b13",
            "plate_01_c13"
        ]
    );
    assert_eq!(
        expand_group("plate_01_[a:2:e]1"),
        vec!["plate_01_a1", "plate_01_c1", "plate_01_e1"]
    );
    assert_eq!(
        expand_group("plate_01_a[1:3:10]"),
        vec![
            "plate_01_a1",
            "plate_01_a4",
            "plate_01_a7",
            "plate_01_a10"
        ]
    );
}

#[test]
fn expand_groups_cases() {
    assert_eq!(
        expand_groups("plate_01_a[1:3],plate_01_b[2:4], plate_02_c5,plate_01_d[1,2]"),
        vec![
            "plate_01_a1",
            "plate_01_a2",
            "plate_01_a3",
            "plate_01_b2",
            "plate_01_b3",
            "plate_01_b4",
            "plate_02_c5",
            "plate_01_d1",
            "plate_01_d2",
        ]
    );
    assert_eq!(
        expand_groups("plate_01_a[1:3]"),
        vec!["plate_01_a1", "plate_01_a2", "plate_01_a3"]
    );
}

#[test]
fn flow_cytometry_data_roundtrip() {
    let es = with_workspace(|| read_data(fixture("small.xlsx").to_str().unwrap()).unwrap());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tmp.esm");
    write_esm_data(&es, path.to_str().unwrap()).unwrap();
    let written = read_esm(path.to_str().unwrap()).unwrap();

    let names = sample_names(&written);
    let expected = [
        "plate_01_a1.FL1_H",
        "plate_01_a1.SSC_H",
        "plate_01_a1.forward",
        "plate_01_a1.newtime",
    ];
    let names_set: std::collections::BTreeSet<String> = names.into_iter().collect();
    let expected_set: std::collections::BTreeSet<String> =
        expected.iter().map(|s| s.to_string()).collect();
    assert_eq!(names_set, expected_set);

    // Confirm we can also fetch raw values from at least one channel
    let name_col = written.samples.column("name").unwrap();
    let forward_idx = name_col
        .iter()
        .position(|v| v.as_str() == Some("plate_01_a1.forward"))
        .unwrap();
    let forward_values = get_sample_values(&written, forward_idx);
    assert_eq!(forward_values, vec![628.0, 1023.0, 373.0, 1023.0]);
}

// Suppress unused-imports warnings in cases the compiler doesn't notice.
#[allow(dead_code)]
fn _use_value(_: Value) {}
