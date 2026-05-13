//! Port of `test/read_tests.jl`.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use esm::dataframe::{DataFrame, Value};
use esm::esm_files::{get_sample_values, read_esm, read_esm_from_str, to_rfi};
use esm::read_data::{read_data, write_esm_data};
use esm::utils::{at_od, at_time, between_times, index_between_vals_df};

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

const MOCK_ESM: &str = r#"
{
    "samples": {
        "plate_01_a1": {
            "values": {
                "FL1_A": [54.0, 143.0, 25.0, 71.0],
                "SSC_H": [534.0, 645.0, 346.0, 1254.0],
                "FSC_H": [634.0, 965.0, 643.0, 1015.0]
            },
            "type": "population",
            "metadata": {
                "FL1_A": {"range": "1024", "ex_pow": null, "filter": null,
                          "det_volt": null, "amp_type": "0,0", "ex_wav": null,
                          "amp_gain": null, "name_s": null, "name": "FL1-A",
                          "det_type": null, "perc_em": null, "raw_metadata": {}},
                "SSC_H": {"range": "1024", "ex_pow": null, "filter": null,
                          "det_volt": null, "amp_type": "2,0.01", "ex_wav": null,
                          "amp_gain": null, "name_s": "SSC-H", "name": "SSC-H",
                          "det_type": null, "perc_em": null, "raw_metadata": {}},
                "FSC_H": {"range": "1024", "ex_pow": null, "filter": null,
                          "det_volt": null, "amp_type": "0,0", "ex_wav": null,
                          "amp_gain": null, "name_s": "FSC-H", "name": "FSC-H",
                          "det_type": null, "perc_em": null, "raw_metadata": {}}
            }
        },
        "plate_01_a2": {
            "values": {
                "FL1_A": [0.0, 143.0, 0.0, 61.0],
                "SSC_H": [280.0, 735.0, 128.0, 1023.0],
                "FSC_H": [628.0, 1023.0, 373.0, 1023.0]
            },
            "type": "population",
            "metadata": {
                "FL1_A": {"range": "1024", "ex_pow": null, "filter": null,
                          "det_volt": null, "amp_type": "0,0", "ex_wav": null,
                          "amp_gain": null, "name_s": null, "name": "FL1-A",
                          "det_type": null, "perc_em": null, "raw_metadata": {}},
                "SSC_H": {"range": "1024", "ex_pow": null, "filter": null,
                          "det_volt": null, "amp_type": "2,0.01", "ex_wav": null,
                          "amp_gain": null, "name_s": "SSC-H", "name": "SSC-H",
                          "det_type": null, "perc_em": null, "raw_metadata": {}},
                "FSC_H": {"range": "1024", "ex_pow": null, "filter": null,
                          "det_volt": null, "amp_type": "0,0", "ex_wav": null,
                          "amp_gain": null, "name_s": "FSC-H", "name": "FSC-H",
                          "det_type": null, "perc_em": null, "raw_metadata": {}}
            }
        }
    },
    "groups": {
        "plate_01": {
            "type": "physical",
            "sample_IDs": ["plate_01_a1", "plate_01_a2"],
            "metadata": {"autodefined": "true"}
        }
    },
    "transformations": {"flow_cyt": {"equation": "1"}},
    "views": {"flow_cy": {"data": ["flow_cyt"]}},
    "metadata": {
        "date_created": "2026-05-04T15:12:08.285",
        "date_modified": "2026-05-04T15:12:08.285",
        "esm_version": "0.1.0",
        "schema_version": "0.1.0",
        "description": ""
    }
}
"#;

#[test]
fn read_esm_basic() {
    let es = read_esm_from_str(MOCK_ESM).unwrap();
    let names: BTreeSet<String> = es
        .samples
        .column("name")
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    let expected: BTreeSet<String> = [
        "plate_01_a1.FL1_A",
        "plate_01_a1.SSC_H",
        "plate_01_a1.FSC_H",
        "plate_01_a2.FL1_A",
        "plate_01_a2.SSC_H",
        "plate_01_a2.FSC_H",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(names, expected);

    let channels: BTreeSet<String> = es
        .samples
        .column("channel")
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    assert_eq!(
        channels,
        ["FL1_A", "SSC_H", "FSC_H"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    );

    let types: BTreeSet<String> = es
        .samples
        .column("type")
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    assert_eq!(types, ["population".to_string()].into_iter().collect());

    let name_col = es.samples.column("name").unwrap();
    let idx_a2_fl1 = name_col
        .iter()
        .position(|v| v.as_str() == Some("plate_01_a2.FL1_A"))
        .unwrap();
    assert_eq!(get_sample_values(&es, idx_a2_fl1), vec![0.0, 143.0, 0.0, 61.0]);

    // Metadata keys per row
    let expected_meta: BTreeSet<String> = [
        "range", "ex_pow", "filter", "det_volt", "amp_type", "ex_wav",
        "amp_gain", "name_s", "name", "det_type", "perc_em", "raw_metadata",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let meta_col = es.samples.column("metadata").unwrap();
    for v in meta_col {
        let s = v.as_str().unwrap_or("{}");
        let parsed: serde_json::Value = serde_json::from_str(s).unwrap();
        let keys: BTreeSet<String> = parsed
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(keys, expected_meta);
    }

    let group_col = es.groups.column("group").unwrap();
    let groups: BTreeSet<String> = group_col
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    assert_eq!(groups, ["plate_01".to_string()].into_iter().collect());

    let trans = &es.transformations;
    assert_eq!(trans.get("flow_cyt").unwrap()["equation"], "1");

    let views = &es.views;
    assert_eq!(views["flow_cy"]["data"], vec!["flow_cyt".to_string()]);
}

#[test]
fn index_between_vals_basic() {
    let mut df = DataFrame::new();
    df.insert_f64_column("A", (1..=10).map(|i| i as f64).collect());
    df.insert_f64_column("B", (11..=20).map(|i| i as f64).collect());

    let r = index_between_vals_df(&df, 3.0, 8.0);
    assert_eq!(r["A"], (Some(2), Some(7))); // 0-based
    assert_eq!(r["B"], (None, None));

    let r = index_between_vals_df(&df, 5.0, 10.0);
    assert_eq!(r["A"], (Some(4), Some(9)));
    assert_eq!(r["B"], (None, None));

    let r = index_between_vals_df(&df, 0.0, 15.0);
    assert_eq!(r["A"], (Some(0), Some(9)));
    assert_eq!(r["B"], (Some(0), Some(4)));

    let r = index_between_vals_df(&df, 2.5, 13.5);
    assert_eq!(r["A"], (Some(2), Some(9)));
    assert_eq!(r["B"], (Some(0), Some(2)));

    let r = index_between_vals_df(&df, f64::NEG_INFINITY, f64::INFINITY);
    assert_eq!(r["A"], (Some(0), Some(9)));
    assert_eq!(r["B"], (Some(0), Some(9)));
}

fn make_ab_df() -> DataFrame {
    let mut df = DataFrame::new();
    df.insert_f64_column("A", (1..=10).map(|i| i as f64).collect());
    df.insert_f64_column("B", (11..=20).map(|i| i as f64).collect());
    df
}

#[test]
fn between_times_basic() {
    let df = make_ab_df();
    let mut time_col = DataFrame::new();
    time_col.insert_f64_column(
        "Time",
        vec![
            518000.0, 1118000.0, 1718000.0, 2318000.0, 2918000.0, 3518000.0,
            4118000.0, 4718000.0, 5318000.0, 5400000.0,
        ],
    );

    let r = between_times(&df, &time_col, 0.0, 0.0);
    assert_eq!(r.nrow(), 0);

    let r = between_times(&df, &time_col, 1e-11, 3e-11);
    assert_eq!(r.nrow(), 0);

    let r = between_times(&df, &time_col, 9.0, 15.0);
    assert_eq!(r.nrow(), 0);

    let r = between_times(&df, &time_col, 0.0, 50.0);
    assert_eq!(r.column_f64("A").unwrap(), (1..=5).map(|i| i as f64).collect::<Vec<_>>());
    assert_eq!(r.column_f64("B").unwrap(), (11..=15).map(|i| i as f64).collect::<Vec<_>>());

    let r = between_times(&df, &time_col, 90.0, 90.0);
    assert_eq!(r.column_f64("A").unwrap(), vec![10.0]);
    assert_eq!(r.column_f64("B").unwrap(), vec![20.0]);
}

#[test]
fn at_time_basic() {
    let df = make_ab_df();
    let mut time_col = DataFrame::new();
    time_col.insert_f64_column(
        "Time",
        vec![
            518000.0, 1118000.0, 1718000.0, 2318000.0, 2918000.0, 3518000.0,
            4118000.0, 4718000.0, 5318000.0, 5918000.0,
        ],
    );

    let r = at_time(&df, &time_col, 30.0);
    assert_eq!(r.column_f64("A").unwrap(), vec![3.0]);
    assert_eq!(r.column_f64("B").unwrap(), vec![13.0]);

    let r = at_time(&df, &time_col, 0.0);
    assert_eq!(r.nrow(), 0);

    let r = at_time(&df, &time_col, 1000.0);
    assert_eq!(r.column_f64("A").unwrap(), vec![10.0]);
    assert_eq!(r.column_f64("B").unwrap(), vec![20.0]);
}

#[test]
fn at_od_basic() {
    let mut od_df = DataFrame::new();
    od_df.insert_f64_column("A", vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    od_df.insert_f64_column("B", vec![0.2, 0.3, 0.4, 0.5, 0.6]);
    let mut target_df = DataFrame::new();
    target_df.insert_f64_column("A", vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    target_df.insert_f64_column("B", vec![20.0, 30.0, 40.0, 50.0, 60.0]);

    let r = at_od(&od_df, &target_df, 0.3);
    assert_eq!(r.column("A").unwrap()[0], Value::Float(30.0));
    assert_eq!(r.column("B").unwrap()[0], Value::Float(30.0));

    let r = at_od(&od_df, &target_df, 0.1);
    assert_eq!(r.column("A").unwrap()[0], Value::Float(10.0));
    assert!(matches!(r.column("B").unwrap()[0], Value::Missing));

    let r = at_od(&od_df, &target_df, 0.5);
    assert_eq!(r.column("A").unwrap()[0], Value::Float(50.0));
    assert_eq!(r.column("B").unwrap()[0], Value::Float(50.0));
}

#[test]
fn produce_views_example() {
    let dir = tempfile::tempdir().unwrap();
    let esm_path = dir.path().join("example.esm");
    let data = with_workspace(|| read_data(fixture("example.xlsx").to_str().unwrap()).unwrap());
    write_esm_data(&data, esm_path.to_str().unwrap()).unwrap();
    let es = read_esm(esm_path.to_str().unwrap()).unwrap();

    let trans_map: indexmap::IndexMap<String, String> = es
        .transformations
        .iter()
        .filter_map(|(k, v)| v.get("equation").map(|eq| (k.clone(), eq.clone())))
        .collect();

    let to_out: Vec<&str> = Vec::new();
    let views = esm::views::produce_views(&es, &trans_map, &to_out).unwrap();

    let keys: BTreeSet<String> = views.keys().cloned().collect();
    let expected: BTreeSet<String> = [
        "group1", "group2", "group3", "flowsub", "odsub", "sample", "mega",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(keys, expected);

    use esm::views::ViewResult;
    let frame = match &views["group1"] {
        ViewResult::Frame(df) => df.clone(),
        other => panic!("group1 not a frame: {:?}", other),
    };
    let cols: BTreeSet<String> = frame.names().iter().map(|s| s.to_string()).collect();
    for expected in [
        "plate_01_a5.OD",
        "plate_01_a5.flo",
        "plate_01_a1.OD",
        "plate_01_a1.flo",
        "plate_01_a9.OD",
        "plate_01_a9.flo",
    ] {
        assert!(cols.contains(expected), "group1 missing {}", expected);
    }
    let a5_od = frame.column_f64("plate_01_a5.OD").unwrap();
    assert_eq!(&a5_od[..3], &[0.169, 0.173, 0.177]);

    let frame = match &views["group2"] {
        ViewResult::Frame(df) => df.clone(),
        other => panic!("group2 not a frame: {:?}", other),
    };
    let a8_od = frame.column_f64("plate_01_a8.OD").unwrap();
    assert_eq!(&a8_od[1..4], &[0.152, 0.154, 0.157]);

    let frame = match &views["group3"] {
        ViewResult::Frame(df) => df.clone(),
        other => panic!("group3 not a frame: {:?}", other),
    };
    let a3_flo = frame.column_f64("plate_01_a3.flo").unwrap();
    let tail: Vec<f64> = a3_flo.iter().rev().take(3).copied().collect::<Vec<_>>().into_iter().rev().collect();
    assert_eq!(tail, vec![211.0, 201.0, 209.0]);

    // Sample view
    let frame = match &views["sample"] {
        ViewResult::Frame(df) => df.clone(),
        other => panic!("sample not a frame: {:?}", other),
    };
    assert_eq!(frame.names(), vec!["plate_01_time.flo"]);
    let times = frame.column_f64("plate_01_time.flo").unwrap();
    assert_eq!(times[0], 544714.0);
    assert_eq!(times[1], 1144314.0);
    assert_eq!(times[times.len() - 2], 66544764.0);
    assert_eq!(times[times.len() - 1], 67144719.0);

    // flowsub / odsub: column set is the same six samples (no .OD/.flo suffix)
    let frame = match &views["flowsub"] {
        ViewResult::Frame(df) => df.clone(),
        other => panic!("flowsub not a frame: {:?}", other),
    };
    let names: BTreeSet<String> = frame.names().iter().map(|s| s.to_string()).collect();
    let expected: BTreeSet<String> = [
        "plate_01_a5",
        "plate_01_a1",
        "plate_01_a9",
        "plate_01_a8",
        "plate_01_a3",
        "plate_01_a7",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(names, expected);

    // Numerical values for flowsub & odsub
    let frame = match &views["flowsub"] {
        ViewResult::Frame(df) => df.clone(),
        _ => unreachable!(),
    };
    let col = frame.column_f64("plate_01_a9").unwrap();
    let nrow = col.len();
    let probe = [col[0], col[1], col[nrow - 2], col[nrow - 1]];
    let expected = [0.3333333333333322, -2.666666666666668, -160.66666666666669, -162.33333333333331];
    for (g, e) in probe.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-6, "flowsub mismatch: got {} expected {}", g, e);
    }

    let frame = match &views["odsub"] {
        ViewResult::Frame(df) => df.clone(),
        _ => unreachable!(),
    };
    let col = frame.column_f64("plate_01_a3").unwrap();
    let nrow = col.len();
    let probe = [col[0], col[1], col[nrow - 2], col[nrow - 1]];
    let expected = [0.0026666666666666783, 0.0026666666666666783, -0.10200000000000009, -0.10133333333333328];
    for (g, e) in probe.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-6, "odsub mismatch: got {} expected {}", g, e);
    }
}

#[test]
fn to_rfi_example() {
    let dir = tempfile::tempdir().unwrap();
    let esm_path = dir.path().join("example.esm");
    let data = with_workspace(|| read_data(fixture("example.xlsx").to_str().unwrap()).unwrap());
    write_esm_data(&data, esm_path.to_str().unwrap()).unwrap();
    let es = read_esm(esm_path.to_str().unwrap()).unwrap();

    let out = to_rfi(&es, "plate_02_a1");

    let fsc_h = out.column_f64("FSC_H").unwrap();
    assert_eq!(fsc_h, vec![628.0, 1023.0, 373.0, 1023.0]);
    let fsc_h_max = out.column_f64("FSC_H.max").unwrap();
    assert_eq!(fsc_h_max, vec![1024.0; 4]);
    let fsc_h_min = out.column_f64("FSC_H.min").unwrap();
    assert_eq!(fsc_h_min, vec![1.0; 4]);

    let fl1_h = out.column_f64("FL1_H").unwrap();
    let expected = [2.26449442, 134.40293884, 1.53816354, 64.86381531];
    for (g, e) in fl1_h.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-5, "FL1_H got {}, expected {}", g, e);
    }

    let ssc_h = out.column_f64("SSC_H").unwrap();
    let expected = [0.03522695, 0.2726132, 0.01778279, 0.99551286];
    for (g, e) in ssc_h.iter().zip(expected.iter()) {
        assert!((g - e).abs() < 1e-5, "SSC_H got {}, expected {}", g, e);
    }
}

#[test]
fn raw_metadata_present() {
    let dir = tempfile::tempdir().unwrap();
    let esm_path = dir.path().join("example.esm");
    let data = with_workspace(|| read_data(fixture("example.xlsx").to_str().unwrap()).unwrap());
    write_esm_data(&data, esm_path.to_str().unwrap()).unwrap();
    let es = read_esm(esm_path.to_str().unwrap()).unwrap();

    let name_col = es.samples.column("name").unwrap();
    let metadata_col = es.samples.column("metadata").unwrap();
    let idx = name_col
        .iter()
        .position(|v| v.as_str() == Some("plate_02_a1.FL1_A"))
        .expect("plate_02_a1.FL1_A not found");
    let meta_str = metadata_col[idx].as_str().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(meta_str).unwrap();
    assert!(parsed["raw_metadata"].is_object());
    let raw = &parsed["raw_metadata"];
    assert_eq!(raw["CREATOR"], serde_json::Value::String("CELLQuest<aa> 3.3".into()));
    assert_eq!(raw["FCSVERSION"], serde_json::Value::String("3".into()));
}

#[test]
fn empty_template_rows() {
    let es = with_workspace(|| read_data(fixture("blank_rows.xlsx").to_str().unwrap()).unwrap());
    assert!(es["samples"].as_object().unwrap().contains_key("plate_01_time"));
    assert!(es["samples"].as_object().unwrap().contains_key("plate_02_time"));
    assert!(es["groups"].as_object().unwrap().contains_key("g1"));
    assert!(es["groups"].as_object().unwrap().contains_key("g2"));
    assert!(es["transformations"].as_object().unwrap().contains_key("t1"));
    assert!(es["transformations"].as_object().unwrap().contains_key("t2"));
    assert!(es["views"].as_object().unwrap().contains_key("v1"));
    assert!(es["views"].as_object().unwrap().contains_key("v2"));
}

#[test]
fn invalid_channel_map() {
    let err = with_workspace(|| {
        read_data(fixture("invalid_channel_map.xlsx").to_str().unwrap()).unwrap_err()
    });
    assert!(
        err.to_string().to_lowercase().contains("valid format")
            || err.to_string().to_lowercase().contains("not valid"),
        "expected invalid channel map error, got: {}",
        err
    );
}

#[test]
fn requested_missing_channel() {
    let err = with_workspace(|| {
        read_data(
            fixture("requested_but_missing_channel.xlsx").to_str().unwrap(),
        )
        .unwrap_err()
    });
    assert!(
        err.to_string().contains("missing_channel_i_want")
            || err.to_string().to_lowercase().contains("not found"),
        "expected missing channel error, got: {}",
        err
    );
}

#[test]
fn no_specified_channels() {
    let es = with_workspace(|| {
        read_data(fixture("no-specified-channels.xlsx").to_str().unwrap()).unwrap()
    });
    let samples = es["samples"].as_object().unwrap();
    let p1_keys: BTreeSet<&str> = samples["plate_01_a1"]["values"]
        .as_object()
        .unwrap()
        .keys()
        .map(|s| s.as_str())
        .collect();
    for k in ["abs600", "abs700", "535_485"] {
        assert!(p1_keys.contains(k), "plate_01_a1 missing channel {}", k);
    }
    let p2_keys: BTreeSet<&str> = samples["plate_02_a1"]["values"]
        .as_object()
        .unwrap()
        .keys()
        .map(|s| s.as_str())
        .collect();
    assert!(p2_keys.contains("FL1_A"));
}

#[test]
fn group_metadata() {
    let es = with_workspace(|| read_data(fixture("extra-metadata.xlsx").to_str().unwrap()).unwrap());
    let g = &es["groups"];
    let pr_meta = g["plate1"]["metadata"].as_object().unwrap();
    assert_eq!(pr_meta["autodefined"], serde_json::Value::String("false".into()));
    // "Plate reader" key should be present, value should be boolean true (or string).
    let pr = &pr_meta["Plate reader"];
    let truthy = matches!(pr, serde_json::Value::Bool(true))
        || pr.as_str().map(|s| s.eq_ignore_ascii_case("true")).unwrap_or(false);
    assert!(truthy, "plate1 Plate reader expected truthy, got {:?}", pr);

    let other_meta = g["otherdata"]["metadata"].as_object().unwrap();
    let pr = &other_meta["Plate reader"];
    let falsy = matches!(pr, serde_json::Value::Bool(false))
        || pr.as_str().map(|s| s.eq_ignore_ascii_case("false")).unwrap_or(false);
    assert!(falsy, "otherdata Plate reader expected falsy, got {:?}", pr);
}

#[test]
fn issue_34() {
    // Importing a single-channel sample shouldn't blow up.
    with_workspace(|| read_data(fixture("issue34.xlsx").to_str().unwrap()).unwrap());
}
