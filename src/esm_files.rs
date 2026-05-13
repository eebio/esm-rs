use std::fs;

use indexmap::IndexMap;
use log::info;
use serde_json::Value as JsonValue;

use crate::dataframe::{DataFrame, Value};
use crate::esm_types::{EsmFile, EsmZones};

/// Parse an ESM file into an EsmZones object.
pub fn read_esm(file: &str) -> Result<EsmZones, Box<dyn std::error::Error>> {
    info!("Reading ESM file at: {}", file);
    let contents = fs::read_to_string(file)?;
    read_esm_from_str(&contents)
}

/// Parse ESM data from a JSON string.
pub fn read_esm_from_str(contents: &str) -> Result<EsmZones, Box<dyn std::error::Error>> {
    let normalised = permit_special_floats(contents);
    let ef: EsmFile = serde_json::from_str(&normalised)?;

    // Collect group names in order
    let group_names: Vec<String> = ef.groups.keys().cloned().collect();

    // Build samples DataFrame
    let mut sample_names: Vec<Value> = Vec::new();
    let mut sample_channels: Vec<Value> = Vec::new();
    let mut sample_types: Vec<Value> = Vec::new();
    let mut sample_values: Vec<Value> = Vec::new();
    let mut sample_metadata: Vec<Value> = Vec::new();
    let mut group_flags: IndexMap<String, Vec<Value>> = IndexMap::new();
    for gn in &group_names {
        group_flags.insert(gn.clone(), Vec::new());
    }

    for (sample_name, sample) in &ef.samples {
        let lower_name = sample_name.to_lowercase();
        for (channel_name, values) in &sample.values {
            // Convert JSON values to f64, replacing null with NaN
            let float_values: Vec<f64> = values
                .iter()
                .map(|v| match v {
                    JsonValue::Number(n) => n.as_f64().unwrap_or(f64::NAN),
                    JsonValue::Null => f64::NAN,
                    _ => f64::NAN,
                })
                .collect();

            let full_name = format!("{}.{}", lower_name, channel_name);
            sample_names.push(Value::Str(full_name));
            sample_channels.push(Value::Str(channel_name.clone()));
            sample_types.push(Value::Str(sample.sample_type.clone()));
            // Store values as a JSON string for now (since DataFrame columns are homogeneous)
            sample_values.push(Value::Str(serde_json::to_string(&float_values)?));

            // Extract metadata for this channel
            let meta = if sample.metadata.len() > 1 {
                // Multiple metadata entries - look for channel-specific metadata
                if let Some(chan_meta) = sample.metadata.get(channel_name) {
                    let mut meta_map: IndexMap<String, JsonValue> = if chan_meta.is_object() {
                        serde_json::from_value(chan_meta.clone())?
                    } else {
                        IndexMap::new()
                    };
                    // Add raw_metadata
                    if !meta_map.contains_key("raw_metadata") {
                        if let Some(raw) = sample.metadata.get("raw_metadata") {
                            meta_map.insert("raw_metadata".to_string(), raw.clone());
                        }
                    }
                    serde_json::to_string(&meta_map)?
                } else {
                    serde_json::to_string(&sample.metadata)?
                }
            } else {
                serde_json::to_string(&sample.metadata)?
            };
            sample_metadata.push(Value::Str(meta));

            // Set group membership flags
            for (gn, group) in &ef.groups {
                let is_member = group
                    .sample_ids
                    .iter()
                    .any(|id| id.to_lowercase() == lower_name);
                group_flags
                    .get_mut(gn)
                    .unwrap()
                    .push(Value::Bool(is_member));
            }
        }
    }

    let mut samples = DataFrame::new();
    samples.insert_column("name", sample_names);
    samples.insert_column("channel", sample_channels);
    samples.insert_column("type", sample_types);
    samples.insert_column("values", sample_values);
    samples.insert_column("metadata", sample_metadata);
    for (gn, flags) in group_flags {
        samples.insert_column(gn, flags);
    }

    // Build groups DataFrame
    let mut group_name_col: Vec<Value> = Vec::new();
    let mut group_sample_ids_col: Vec<Value> = Vec::new();
    let mut group_metadata_col: Vec<Value> = Vec::new();

    for (gn, group) in &ef.groups {
        group_name_col.push(Value::Str(gn.clone()));
        let ids: Vec<String> = group.sample_ids.iter().map(|s| s.to_lowercase()).collect();
        group_sample_ids_col.push(Value::Str(serde_json::to_string(&ids)?));
        group_metadata_col.push(Value::Str(serde_json::to_string(&group.metadata)?));
    }

    let mut groups = DataFrame::new();
    groups.insert_column("group", group_name_col);
    groups.insert_column("sample_IDs", group_sample_ids_col);
    groups.insert_column("metadata", group_metadata_col);

    // Build transformations
    let transformations = ef.transformations;

    // Build views - convert JsonValue arrays to String arrays
    let mut views: IndexMap<String, IndexMap<String, Vec<String>>> = IndexMap::new();
    for (name, view_map) in &ef.views {
        let mut converted: IndexMap<String, Vec<String>> = IndexMap::new();
        for (key, val) in view_map {
            if let JsonValue::Array(arr) = val {
                let strings: Vec<String> = arr
                    .iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect();
                converted.insert(key.clone(), strings);
            }
        }
        views.insert(name.clone(), converted);
    }

    info!("ESM file successfully read.");
    Ok(EsmZones {
        samples,
        groups,
        transformations,
        views,
        metadata: ef.metadata,
    })
}

/// Replace bare `NaN`, `Infinity`, and `-Infinity` tokens (which Julia's
/// `JSON.parsefile(...; allownan=true)` emits) with `null`, leaving identical
/// substrings inside JSON strings untouched. The existing pipeline maps null
/// to `f64::NAN` when populating numeric arrays.
fn permit_special_floats(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut in_string = false;
    let mut escape = false;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if in_string {
            out.push(c as char);
            if escape {
                escape = false;
            } else if c == b'\\' {
                escape = true;
            } else if c == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if c == b'"' {
            in_string = true;
            out.push(c as char);
            i += 1;
            continue;
        }
        if let Some(len) = match_special_token(bytes, i) {
            out.push_str("null");
            i += len;
            continue;
        }
        out.push(c as char);
        i += 1;
    }
    out
}

fn match_special_token(bytes: &[u8], i: usize) -> Option<usize> {
    for token in [b"-Infinity".as_slice(), b"Infinity".as_slice(), b"NaN".as_slice()] {
        let end = i + token.len();
        if end > bytes.len() || bytes[i..end] != *token {
            continue;
        }
        let prev_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
        let next_ok = end == bytes.len() || !is_ident_byte(bytes[end]);
        if prev_ok && next_ok {
            return Some(token.len());
        }
    }
    None
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'.'
}

/// Write ESM data to a JSON file.
pub fn write_esm(
    data: &IndexMap<String, JsonValue>,
    file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(data)?;
    fs::write(file, json)?;
    info!("ESM written to {}", file);
    Ok(())
}

// ---- Helper functions for accessing EsmZones data ----

/// Get the parsed values (Vec<f64>) for a sample row.
pub fn get_sample_values(es: &EsmZones, row_idx: usize) -> Vec<f64> {
    if let Some(col) = es.samples.column("values") {
        if let Some(Value::Str(s)) = col.get(row_idx) {
            serde_json::from_str(s).unwrap_or_default()
        } else {
            vec![]
        }
    } else {
        vec![]
    }
}

/// Get the parsed metadata for a sample row.
pub fn get_sample_metadata(
    es: &EsmZones,
    row_idx: usize,
) -> IndexMap<String, JsonValue> {
    if let Some(col) = es.samples.column("metadata") {
        if let Some(Value::Str(s)) = col.get(row_idx) {
            serde_json::from_str(s).unwrap_or_default()
        } else {
            IndexMap::new()
        }
    } else {
        IndexMap::new()
    }
}

/// Get the sample_IDs for a group by name.
pub fn find_group(es: &EsmZones, group_name: &str) -> Vec<String> {
    let group_col = es.groups.column("group").unwrap();
    let ids_col = es.groups.column("sample_IDs").unwrap();
    for (g, ids) in group_col.iter().zip(ids_col.iter()) {
        if let (Value::Str(gn), Value::Str(ids_str)) = (g, ids) {
            if gn == group_name {
                return serde_json::from_str(ids_str).unwrap_or_default();
            }
        }
    }
    vec![]
}

/// Get group names.
pub fn group_names(es: &EsmZones) -> Vec<String> {
    es.groups
        .column("group")
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect()
}

/// Get sample names.
pub fn sample_names(es: &EsmZones) -> Vec<String> {
    es.samples
        .column("name")
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect()
}

/// Get sample name without extension (channel).
pub fn get_sample_id(name: &str) -> &str {
    name.rsplit_once('.').map_or(name, |(base, _)| base)
}

/// Build a DataFrame from sample values for a set of sample rows.
/// Each column is named by the sample name, values are the stored f64 arrays.
pub fn form_df(es: &EsmZones, row_indices: &[usize]) -> DataFrame {
    let name_col = es.samples.column("name").unwrap();

    // Find maximum length
    let mut max_len = 0;
    for &idx in row_indices {
        let vals = get_sample_values(es, idx);
        if vals.len() > max_len {
            max_len = vals.len();
        }
    }

    let mut df = DataFrame::new();
    for &idx in row_indices {
        if let Some(Value::Str(name)) = name_col.get(idx) {
            let mut vals = get_sample_values(es, idx);
            // Pad with NaN to match max length
            vals.resize(max_len, f64::NAN);
            df.insert_f64_column(name.clone(), vals);
        }
    }
    df
}

/// Filter rows where a group flag is true.
pub fn filter_row(es: &EsmZones, group: &str) -> Vec<usize> {
    if let Some(col) = es.samples.column(group) {
        col.iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if v.as_bool().unwrap_or(false) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    } else {
        vec![]
    }
}

/// Filter a DataFrame to only include columns matching a channel name.
/// Removes the channel suffix from column names.
pub fn filter_channel(df: &DataFrame, channel: &str) -> DataFrame {
    let mut result = DataFrame::new();
    for name in df.names() {
        let parts: Vec<&str> = name.split('.').collect();
        if parts.contains(&channel) || name == "id" {
            let new_name = name.replace(&format!(".{}", channel), "");
            result.insert_column(new_name, df.column(name).unwrap().clone());
        }
    }
    result
}

/// Convert RFI (relative fluorescence intensity) for a sample.
pub fn to_rfi(es: &EsmZones, sample_name: &str) -> DataFrame {
    let name_col = es.samples.column("name").unwrap();
    let channel_col = es.samples.column("channel").unwrap();

    // Find all rows matching this sample
    let pattern = format!("{}.", sample_name);
    let matching_indices: Vec<usize> = name_col
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            if let Value::Str(n) = v {
                if n.starts_with(&pattern) {
                    Some(i)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    let mut result = DataFrame::new();
    let mut data_len = 0;

    for &idx in &matching_indices {
        let channel = if let Value::Str(c) = &channel_col[idx] {
            c.clone()
        } else {
            continue;
        };

        let values = get_sample_values(es, idx);
        let meta = get_sample_metadata(es, idx);

        // Parse amp_type
        let amp_type_str = meta
            .get("amp_type")
            .and_then(|v| v.as_str())
            .unwrap_or("0,0");
        let amp_parts: Vec<f64> = amp_type_str
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();

        let range_val: f64 = meta
            .get("range")
            .and_then(|v| v.as_str().or_else(|| v.as_f64().map(|_| "")))
            .and_then(|s| {
                if s.is_empty() {
                    meta.get("range").and_then(|v| v.as_f64())
                } else {
                    s.parse().ok()
                }
            })
            .unwrap_or(1024.0);

        let (data, min_val, max_val) = if amp_parts.first().copied().unwrap_or(0.0) == 0.0 {
            // Linear gain
            let amp_gain: f64 = meta
                .get("amp_gain")
                .and_then(|v| {
                    if v.is_null() {
                        None
                    } else {
                        v.as_str()
                            .and_then(|s| s.replace(".0", "").parse().ok())
                            .or_else(|| v.as_f64())
                    }
                })
                .unwrap_or(1.0);

            let data: Vec<f64> = values.iter().map(|&v| v / amp_gain).collect();
            let min = 1.0 / amp_gain;
            let max = range_val / amp_gain;
            (data, min, max)
        } else {
            // Non-linear (log) gain
            let a = amp_parts[0];
            let b = amp_parts.get(1).copied().unwrap_or(0.01);
            let data: Vec<f64> = values
                .iter()
                .map(|&v| b * 10.0_f64.powf(a * (v / range_val)))
                .collect();
            let min = b * 10.0_f64.powf(a * (1.0 / range_val));
            let max = b * 10.0_f64.powf(a * (range_val / range_val));
            (data, min, max)
        };

        data_len = data.len();
        result.insert_f64_column(&channel, data);
        result.insert_f64_column(format!("{}.min", channel), vec![min_val; data_len]);
        result.insert_f64_column(format!("{}.max", channel), vec![max_val; data_len]);
    }

    // Add id column
    if data_len > 0 {
        result.insert_column(
            "id",
            (1..=data_len as i64).map(Value::Int).collect(),
        );
    }

    result.sort_columns()
}

/// Get data for an entire group.
pub fn get_group(es: &EsmZones, group_name: &str) -> DataFrame {
    let row_indices = filter_row(es, group_name);
    if row_indices.is_empty() {
        return DataFrame::new();
    }

    // Check sample types
    let type_col = es.samples.column("type").unwrap();
    let first_type = type_col[row_indices[0]].as_str().unwrap_or("");

    if first_type == "population" {
        // Flow cytometry data - concatenate RFI DataFrames
        let sample_ids = find_group(es, group_name);
        let mut combined: Option<DataFrame> = None;

        for sample_id in &sample_ids {
            let mut rfi = to_rfi(es, sample_id);
            if let Some(ref existing) = combined {
                // Offset IDs
                let max_id = existing
                    .column("id")
                    .unwrap()
                    .iter()
                    .filter_map(|v| v.as_i64())
                    .max()
                    .unwrap_or(0);
                if let Some(id_col) = rfi.columns.get_mut("id") {
                    for v in id_col.iter_mut() {
                        if let Value::Int(i) = v {
                            *i += max_id;
                        }
                    }
                }
                combined = Some(existing.vcat(&rfi));
            } else {
                combined = Some(rfi);
            }
        }
        combined.unwrap_or_default()
    } else {
        // Timeseries data
        form_df(es, &row_indices)
    }
}

/// Get data for a single sample.
pub fn get_sample(es: &EsmZones, sample_name: &str) -> DataFrame {
    let name_col = es.samples.column("name").unwrap();
    let type_col = es.samples.column("type").unwrap();

    // Find matching rows
    let matching: Vec<usize> = name_col
        .iter()
        .enumerate()
        .filter_map(|(i, v)| {
            if let Value::Str(n) = v {
                if get_sample_id(n) == sample_name {
                    Some(i)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    if matching.is_empty() {
        return DataFrame::new();
    }

    let sample_type = type_col[matching[0]].as_str().unwrap_or("");
    if sample_type == "population" {
        to_rfi(es, sample_name)
    } else {
        form_df(es, &matching)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_esm_json() -> &'static str {
        r#"{
    "samples": {
        "plate_01_a1": {
            "values": {
                "FL1_A": [54.0, 143.0, 25.0, 71.0],
                "SSC_H": [534.0, 645.0, 346.0, 1254.0],
                "FSC_H": [634.0, 965.0, 643.0, 1015.0]
            },
            "type": "population",
            "metadata": {
                "FL1_A": {
                    "range": "1024", "ex_pow": null, "filter": null,
                    "det_volt": null, "amp_type": "0,0", "ex_wav": null,
                    "amp_gain": null, "name_s": null, "name": "FL1-A",
                    "det_type": null, "perc_em": null, "raw_metadata": {}
                },
                "SSC_H": {
                    "range": "1024", "ex_pow": null, "filter": null,
                    "det_volt": null, "amp_type": "2,0.01", "ex_wav": null,
                    "amp_gain": null, "name_s": "SSC-H", "name": "SSC-H",
                    "det_type": null, "perc_em": null, "raw_metadata": {}
                },
                "FSC_H": {
                    "range": "1024", "ex_pow": null, "filter": null,
                    "det_volt": null, "amp_type": "0,0", "ex_wav": null,
                    "amp_gain": null, "name_s": "FSC-H", "name": "FSC-H",
                    "det_type": null, "perc_em": null, "raw_metadata": {}
                }
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
                "FL1_A": {
                    "range": "1024", "ex_pow": null, "filter": null,
                    "det_volt": null, "amp_type": "0,0", "ex_wav": null,
                    "amp_gain": null, "name_s": null, "name": "FL1-A",
                    "det_type": null, "perc_em": null, "raw_metadata": {}
                },
                "SSC_H": {
                    "range": "1024", "ex_pow": null, "filter": null,
                    "det_volt": null, "amp_type": "2,0.01", "ex_wav": null,
                    "amp_gain": null, "name_s": "SSC-H", "name": "SSC-H",
                    "det_type": null, "perc_em": null, "raw_metadata": {}
                },
                "FSC_H": {
                    "range": "1024", "ex_pow": null, "filter": null,
                    "det_volt": null, "amp_type": "0,0", "ex_wav": null,
                    "amp_gain": null, "name_s": "FSC-H", "name": "FSC-H",
                    "det_type": null, "perc_em": null, "raw_metadata": {}
                }
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
    "transformations": {
        "flow_cyt": {"equation": "1"}
    },
    "views": {
        "flow_cy": {"data": ["flow_cyt"]}
    },
    "metadata": {
        "esm_version": "0.1.0",
        "schema_version": "0.1.0",
        "description": ""
    }
}"#
    }

    #[test]
    fn test_read_esm() {
        let es = read_esm_from_str(mock_esm_json()).unwrap();

        // Check sample names
        let names: Vec<String> = sample_names(&es);
        assert!(names.contains(&"plate_01_a1.FL1_A".to_string()));
        assert!(names.contains(&"plate_01_a1.SSC_H".to_string()));
        assert!(names.contains(&"plate_01_a1.FSC_H".to_string()));
        assert!(names.contains(&"plate_01_a2.FL1_A".to_string()));
        assert!(names.contains(&"plate_01_a2.SSC_H".to_string()));
        assert!(names.contains(&"plate_01_a2.FSC_H".to_string()));
        assert_eq!(names.len(), 6);

        // Check groups
        let gnames = group_names(&es);
        assert_eq!(gnames, vec!["plate_01"]);

        let ids = find_group(&es, "plate_01");
        assert_eq!(ids, vec!["plate_01_a1", "plate_01_a2"]);

        // Check transformations
        assert_eq!(
            es.transformations["flow_cyt"]["equation"],
            "1"
        );

        // Check views
        assert_eq!(
            es.views["flow_cy"]["data"],
            vec!["flow_cyt"]
        );
    }

    #[test]
    fn test_to_rfi_linear() {
        let es = read_esm_from_str(mock_esm_json()).unwrap();
        let rfi = to_rfi(&es, "plate_01_a1");

        // Linear with no gain: FSC_H values should be unchanged
        let fsc = rfi.column_f64("FSC_H").unwrap();
        assert_eq!(fsc, vec![634.0, 965.0, 643.0, 1015.0]);

        // FL1_A with no gain: values should be unchanged
        let fl1 = rfi.column_f64("FL1_A").unwrap();
        assert_eq!(fl1, vec![54.0, 143.0, 25.0, 71.0]);

        // SSC_H with log gain (amp_type = "2,0.01"):
        // data = 0.01 * 10^(2 * value/1024)
        let ssc = rfi.column_f64("SSC_H").unwrap();
        let expected_ssc: Vec<f64> = vec![534.0, 645.0, 346.0, 1254.0]
            .iter()
            .map(|&v| 0.01 * 10.0_f64.powf(2.0 * v / 1024.0))
            .collect();
        for (a, b) in ssc.iter().zip(expected_ssc.iter()) {
            assert!((a - b).abs() < 1e-6, "{} != {}", a, b);
        }
    }
}
