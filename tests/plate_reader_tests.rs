use esm::plate_readers::PlateReaderParser;
use std::path::Path;

fn test_data_path(name: &str) -> String {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("test")
        .join("inputs")
        .join(name);
    base.to_str().unwrap().to_string()
}

fn set_github_workspace() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    // SAFETY: We're in single-threaded test context and this env var is only used by our code
    unsafe { std::env::set_var("GITHUB_WORKSPACE", &workspace); }
}

#[test]
fn test_read_spectramax() {
    let file = test_data_path("spectramax-data.txt");
    let reader = esm::plate_readers::spectramax::SpectraMax;
    let (data, _metadata) = reader
        .read(&file, Some(&["600", "700"]))
        .unwrap();

    assert!(data.contains_key("600"));
    assert!(data.contains_key("700"));
    assert_eq!(data.len(), 2);

    // Check data has correct structure
    let df600 = &data["600"];
    assert!(df600.has_column("time"));
    assert!(df600.has_column("temperature"));
    assert!(df600.nrow() > 0);
}

#[test]
fn test_read_spectramax_all_channels() {
    let file = test_data_path("spectramax-data.txt");
    let reader = esm::plate_readers::spectramax::SpectraMax;
    let (data, _metadata) = reader.read(&file, None).unwrap();

    // Should have multiple channels (fluorescence + absorbance)
    assert!(data.len() >= 2);
}

#[test]
fn test_read_spectramax_duplicate_channels() {
    let file = test_data_path("spectramax-data2.txt");
    let reader = esm::plate_readers::spectramax::SpectraMax;
    let (data, _metadata) = reader
        .read(&file, Some(&["530_485_1", "530_485_2", "530_485_3"]))
        .unwrap();

    assert!(data.contains_key("530_485_1"));
    assert!(data.contains_key("530_485_2"));
    assert!(data.contains_key("530_485_3"));
    assert_eq!(data.len(), 3);
}

#[test]
fn test_read_biotek() {
    let file = test_data_path("biotek-data.csv");
    let reader = esm::plate_readers::biotek::BioTek;
    let (data, _metadata) = reader
        .read(&file, Some(&["OD_600"]))
        .unwrap();

    assert!(data.contains_key("OD_600"));
    assert_eq!(data.len(), 1);

    let df = &data["OD_600"];
    assert!(df.has_column("time"));
    assert!(df.nrow() > 0);
}

#[test]
fn test_read_bmg() {
    let file = test_data_path("bmg-data.csv");
    let reader = esm::plate_readers::bmg::Bmg;
    let (data, _metadata) = reader
        .read(&file, Some(&["ABS_600_0_nm", "FI_YFP_pAN1717"]))
        .unwrap();

    assert!(data.contains_key("ABS_600_0_nm"));
    assert!(data.contains_key("FI_YFP_pAN1717"));
    assert_eq!(data.len(), 2);
}

#[test]
fn test_read_bmg_three_channels() {
    let file = test_data_path("bmg-data.csv");
    let reader = esm::plate_readers::bmg::Bmg;
    let (data, _metadata) = reader
        .read(
            &file,
            Some(&["ABS_600_0_nm", "ABS_700_0_nm", "FI_YFP_pAN1717"]),
        )
        .unwrap();

    assert!(data.contains_key("ABS_600_0_nm"));
    assert!(data.contains_key("ABS_700_0_nm"));
    assert!(data.contains_key("FI_YFP_pAN1717"));
    assert_eq!(data.len(), 3);
}

#[test]
fn test_read_bmg_values() {
    let file = test_data_path("bmg-data.csv");
    let reader = esm::plate_readers::bmg::Bmg;
    let (data, _metadata) = reader.read(&file, None).unwrap();

    // Check A1 values for ABS 600 channel
    let abs600_key = data.keys().find(|k| k.contains("600")).unwrap();
    let df = &data[abs600_key];
    let a1 = df.column("A1").unwrap();

    // First value should be 0.182
    if let esm::dataframe::Value::Float(v) = &a1[0] {
        assert!((*v - 0.182).abs() < 0.001, "Expected 0.182, got {}", v);
    } else {
        panic!("Expected float value for A1[0]");
    }

    // Check time values are in milliseconds
    let time = df.column("time").unwrap();
    if let esm::dataframe::Value::Int(t) = &time[0] {
        assert_eq!(*t, 0, "First time should be 0");
    }
}

#[test]
fn test_read_bmg_time_error() {
    // This checks if floating point time that doesn't fall on an integer reads correctly
    let file = test_data_path("bmg-time-error.csv");
    let reader = esm::plate_readers::bmg::Bmg;
    let (data, _metadata) = reader.read(&file, None).unwrap();

    // Find the 700 channel
    let key = data.keys().find(|k| k.contains("700")).unwrap();
    let df = &data[key];
    let time_col = df.column("time").unwrap();

    // Check that 33047800 is in the time values
    let times: Vec<i64> = time_col
        .iter()
        .filter_map(|v| v.as_i64())
        .collect();
    assert!(
        times.contains(&33047800),
        "Expected 33047800 in time values, got: {:?}",
        &times[50..60.min(times.len())]
    );
}

#[test]
fn test_read_tecan() {
    let file = test_data_path("tecan-data.xlsx");
    let reader = esm::plate_readers::tecan::Tecan;
    let (data, _metadata) = reader
        .read(&file, Some(&["OD_600", "GFP"]))
        .unwrap();

    assert!(data.contains_key("OD_600"));
    assert!(data.contains_key("GFP"));
    assert_eq!(data.len(), 2);

    let df = &data["OD_600"];
    assert!(df.has_column("time"), "Columns: {:?}", df.names());
    assert!(df.nrow() > 0);
}

#[test]
fn test_read_generic_tabular() {
    let file = test_data_path("pr_folder");
    let reader = esm::plate_readers::generic::GenericTabular;
    let (data, _metadata) = reader.read(&file, Some(&["OD"])).unwrap();

    assert!(data.contains_key("OD"));
    assert_eq!(data.len(), 1);

    let df = &data["OD"];
    assert!(df.has_column("time"));
    assert!(df.has_column("A1"));
    assert!(df.nrow() > 0);

    // Check time values - first time is "00:08:38" = 518000ms
    let time_col = df.column("time").unwrap();
    if let esm::dataframe::Value::Int(t) = &time_col[0] {
        assert_eq!(*t, 518000, "First time should be 518000ms");
    } else {
        panic!("Expected Int time value");
    }

    // Check A1 values
    let a1 = df.column("A1").unwrap();
    if let esm::dataframe::Value::Float(v) = &a1[0] {
        assert!((*v - 0.165).abs() < 0.001, "Expected 0.165, got {}", v);
    }
}

#[test]
fn test_read_generic_tabular_all() {
    let file = test_data_path("pr_folder");
    let reader = esm::plate_readers::generic::GenericTabular;
    let (data, _metadata) = reader.read(&file, None).unwrap();

    // Should have both OD and flo channels
    assert!(data.contains_key("OD"));
    assert!(data.contains_key("flo"));
    assert_eq!(data.len(), 2);
}

#[test]
fn test_read_data_bmg_xlsx() {
    set_github_workspace();
    let file = test_data_path("bmg.xlsx");
    let data = esm::read_data::read_data(&file).unwrap();

    // Check structure
    assert!(data.contains_key("samples"));
    assert!(data.contains_key("groups"));
    assert!(data.contains_key("transformations"));
    assert!(data.contains_key("views"));
    assert!(data.contains_key("metadata"));

    let samples = data["samples"].as_object().unwrap();

    // Check specific well values
    assert!(samples.contains_key("plate_01_a1"), "Missing plate_01_a1. Keys: {:?}", samples.keys().take(5).collect::<Vec<_>>());

    // Check A1 ABS 600 values
    let a1 = samples["plate_01_a1"].as_object().unwrap();
    let a1_vals = a1["values"].as_object().unwrap();

    // Find the 600nm absorbance channel (might be named abs600 or ABS_600_0_nm)
    let abs_key = a1_vals.keys().find(|k| k.contains("600") || k.contains("abs")).unwrap();
    let abs_vals = a1_vals[abs_key].as_array().unwrap();
    let first_val = abs_vals[0].as_f64().unwrap();
    let second_val = abs_vals[1].as_f64().unwrap();
    let second_last_val = abs_vals[abs_vals.len() - 2].as_f64().unwrap();
    let last_val = abs_vals[abs_vals.len() - 1].as_f64().unwrap();

    assert!((first_val - 0.182).abs() < 0.001, "Expected 0.182, got {}", first_val);
    assert!((second_val - 0.185).abs() < 0.001, "Expected 0.185, got {}", second_val);
    assert!((second_last_val - 0.169).abs() < 0.001, "Expected 0.169, got {}", second_last_val);
    assert!((last_val - 0.169).abs() < 0.001, "Expected 0.169, got {}", last_val);
}

#[test]
fn test_read_data_generic_xlsx() {
    set_github_workspace();
    let file = test_data_path("example.xlsx");
    let data = esm::read_data::read_data(&file).unwrap();

    let samples = data["samples"].as_object().unwrap();

    // Check time values for OD channel
    let time_key = samples.keys().find(|k| k.contains("time")).unwrap();
    let time_entry = samples[time_key].as_object().unwrap();
    let time_vals = time_entry["values"].as_object().unwrap();
    let od_key = time_vals.keys().find(|k| k.to_lowercase().contains("od")).unwrap();
    let od_times = time_vals[od_key].as_array().unwrap();

    // First time should be 518000ms (00:08:38)
    assert_eq!(od_times[0].as_i64().unwrap(), 518000);

    // Check A1 OD values
    let a1 = samples["plate_01_a1"].as_object().unwrap();
    let a1_vals = a1["values"].as_object().unwrap();
    let od_key = a1_vals.keys().find(|k| k.to_lowercase().contains("od")).unwrap();
    let od_vals = a1_vals[od_key].as_array().unwrap();
    assert!((od_vals[0].as_f64().unwrap() - 0.165).abs() < 0.001);
}

#[test]
fn test_read_multipr_unknown_type() {
    let result = esm::plate_readers::read_multipr_file(
        "anyfile/path",
        "random_string",
        &["OD".to_string()],
        &indexmap::IndexMap::new(),
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Unknown plate reader type: random_string"));
}
