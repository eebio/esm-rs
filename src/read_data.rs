//! Read XLSX template files into ESM data structure.

use crate::dataframe::Value;
use crate::fcs::{channel_meta_to_dict, read_fcs};
use crate::plate_readers;
use crate::utils::{expand_groups, format_channel};
use calamine::{open_workbook, Data, Reader, Xlsx};
use indexmap::IndexMap;
use log::info;
use serde_json::Value as JsonValue;
use std::io::BufReader;

/// Read an XLSX template file and process all data into ESM format.
///
/// Returns an ordered dict structure matching the Julia `read_data` output:
/// { "samples" -> {sample_id -> {type, values, metadata}},
///   "groups" -> {group_name -> {sample_IDs, type, metadata}},
///   "transformations" -> {name -> {equation}},
///   "views" -> {name -> {data: [...]}},
///   "metadata" -> {...} }
pub fn read_data(
    file: &str,
) -> Result<IndexMap<String, JsonValue>, Box<dyn std::error::Error>> {
    let mut workbook: Xlsx<BufReader<std::fs::File>> = open_workbook(file)?;

    // Read sheets
    let samples_df = read_sheet(&mut workbook, "Samples")?;
    let groups_df = read_sheet(&mut workbook, "Groups")?;
    let trans_df = read_sheet(&mut workbook, "Transformations")?;
    let views_df = read_sheet(&mut workbook, "Views")?;
    let channel_map_df = read_sheet(&mut workbook, "Channel Map")?;

    // Build channel map
    let mut channel_map: IndexMap<String, String> = IndexMap::new();
    if let (Some(ch_col), Some(new_col)) = (
        find_col_idx(&channel_map_df, "Channel"),
        find_col_idx(&channel_map_df, "New name"),
    ) {
        for row in &channel_map_df.rows {
            let ch = cell_to_string(&row[ch_col]);
            let new_name = cell_to_string(&row[new_col]);
            if !ch.is_empty() && !new_name.is_empty() {
                if format_channel(&new_name) != new_name {
                    return Err(format!(
                        "Channel name '{}' in channel map is not valid. Channels should only contain letters, numbers, and underscores.",
                        new_name
                    ).into());
                }
                channel_map.insert(ch, new_name);
            }
        }
    }

    // Build groups dict
    let mut group_dict: IndexMap<String, JsonValue> = IndexMap::new();
    let g_name_idx = find_col_idx(&groups_df, "Name").ok_or("Groups sheet missing 'Name' column")?;
    let g_samples_idx =
        find_col_idx(&groups_df, "Samples").ok_or("Groups sheet missing 'Samples' column")?;
    for row in &groups_df.rows {
        let name = cell_to_string(&row[g_name_idx]);
        if name.is_empty() {
            continue;
        }
        let samples_str = cell_to_string(&row[g_samples_idx]);
        let sample_ids = expand_groups(&samples_str);

        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "autodefined".to_string(),
            JsonValue::String("false".to_string()),
        );
        // Add extra columns as metadata
        for (i, header) in groups_df.headers.iter().enumerate() {
            if header != "Name" && header != "Samples" {
                metadata.insert(header.clone(), JsonValue::String(cell_to_string(&row[i])));
            }
        }

        let mut group_entry = serde_json::Map::new();
        group_entry.insert(
            "sample_IDs".to_string(),
            JsonValue::Array(sample_ids.into_iter().map(JsonValue::String).collect()),
        );
        group_entry.insert("type".to_string(), JsonValue::String("experimental".to_string()));
        group_entry.insert("metadata".to_string(), JsonValue::Object(metadata));

        group_dict.insert(name, JsonValue::Object(group_entry));
    }

    // Process samples by plate
    let mut sample_dict: IndexMap<String, JsonValue> = IndexMap::new();

    let s_plate_idx = find_col_idx(&samples_df, "Plate").ok_or("Samples sheet missing 'Plate'")?;
    let s_type_idx = find_col_idx(&samples_df, "Type").ok_or("Samples sheet missing 'Type'")?;
    let s_loc_idx = find_col_idx(&samples_df, "Data Location")
        .ok_or("Samples sheet missing 'Data Location'")?;
    let s_brand_idx = find_col_idx(&samples_df, "Plate brand");
    let s_channels_idx = find_col_idx(&samples_df, "Channels");
    let s_well_idx = find_col_idx(&samples_df, "Well");
    let s_name_idx = find_col_idx(&samples_df, "Name");

    // Group samples by plate
    let mut plates: IndexMap<String, Vec<usize>> = IndexMap::new();
    for (i, row) in samples_df.rows.iter().enumerate() {
        let plate = cell_to_string(&row[s_plate_idx]);
        if plate.is_empty() {
            continue;
        }
        plates.entry(plate).or_default().push(i);
    }

    let mut plate_num = 0;
    for (plate_id, row_indices) in &plates {
        plate_num += 1;
        if row_indices.is_empty() {
            continue;
        }

        let first_row = &samples_df.rows[row_indices[0]];
        let ins_type = cell_to_string(&first_row[s_type_idx]).to_lowercase();

        // Resolve data location (handle $GITHUB_WORKSPACE)
        let mut data_loc = cell_to_string(&first_row[s_loc_idx]);
        if data_loc.contains("$GITHUB_WORKSPACE") {
            if let Ok(workspace) = std::env::var("GITHUB_WORKSPACE") {
                data_loc = data_loc.replace("$GITHUB_WORKSPACE", &workspace);
            }
        }

        // Process channels
        let channels_str = s_channels_idx
            .map(|idx| cell_to_string(&first_row[idx]))
            .unwrap_or_default();
        let mut channels: Vec<String> = channels_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s != "missing")
            .collect();
        channels.sort();
        channels.dedup();

        // Ensure channels are in the channel map
        for ch in &channels {
            if !channel_map.contains_key(ch) {
                channel_map.insert(ch.clone(), ch.clone());
            }
        }

        let mut broad_g: Vec<String> = Vec::new();

        if ins_type.contains("plate reader") {
            // Read plate reader data
            let ptype = s_brand_idx
                .map(|idx| cell_to_string(&first_row[idx]))
                .unwrap_or_default();

            let (data, raw_metadata) =
                plate_readers::read_multipr_file(&data_loc, &ptype, &channels, &channel_map)?;

            let channel_keys: Vec<String> = data.keys().cloned().collect();
            // Get well names from the first channel's DataFrame
            if let Some(first_df) = data.values().next() {
                let well_names: Vec<String> = first_df
                    .names()
                    .into_iter()
                    .filter(|n| *n != "time" && *n != "temperature")
                    .map(|s| s.to_string())
                    .collect();

                // Create sample entries for each well
                for well in &well_names {
                    if !is_valid_well_name(well) {
                        continue;
                    }
                    let sample_id = format!("plate_0{}_{}", plate_id, well.to_lowercase());

                    let mut values = serde_json::Map::new();
                    for ch in &channel_keys {
                        if let Some(df) = data.get(ch) {
                            if let Some(col) = df.column(well) {
                                let vals: Vec<JsonValue> = col
                                    .iter()
                                    .map(|v| match v {
                                        Value::Float(f) => {
                                            serde_json::Number::from_f64(*f)
                                                .map(JsonValue::Number)
                                                .unwrap_or(JsonValue::Null)
                                        }
                                        Value::Int(i) => JsonValue::Number((*i).into()),
                                        _ => JsonValue::Null,
                                    })
                                    .collect();
                                values.insert(ch.clone(), JsonValue::Array(vals));
                            }
                        }
                    }

                    let mut entry = serde_json::Map::new();
                    entry.insert(
                        "type".to_string(),
                        JsonValue::String("timeseries".to_string()),
                    );
                    entry.insert("values".to_string(), JsonValue::Object(values));
                    let mut meta = serde_json::Map::new();
                    meta.insert(
                        "raw_metadata".to_string(),
                        JsonValue::String(raw_metadata.clone()),
                    );
                    entry.insert("metadata".to_string(), JsonValue::Object(meta));

                    broad_g.push(sample_id.clone());
                    sample_dict.insert(sample_id, JsonValue::Object(entry));
                }

                // Add time and temperature as special samples
                for ch in &channel_keys {
                    if let Some(df) = data.get(ch) {
                        // Time
                        if let Some(time_col) = df.column("time") {
                            let time_id = format!("plate_0{}_time", plate_id);
                            let entry = sample_dict
                                .entry(time_id)
                                .or_insert_with(|| {
                                    let mut e = serde_json::Map::new();
                                    e.insert(
                                        "type".to_string(),
                                        JsonValue::String("timeseries".to_string()),
                                    );
                                    e.insert(
                                        "values".to_string(),
                                        JsonValue::Object(serde_json::Map::new()),
                                    );
                                    e.insert(
                                        "metadata".to_string(),
                                        JsonValue::Object(serde_json::Map::new()),
                                    );
                                    JsonValue::Object(e)
                                });
                            if let Some(vals_map) = entry.get_mut("values").and_then(|v| v.as_object_mut()) {
                                let vals: Vec<JsonValue> = time_col
                                    .iter()
                                    .map(|v| match v {
                                        Value::Int(i) => JsonValue::Number((*i).into()),
                                        Value::Float(f) => serde_json::Number::from_f64(*f)
                                            .map(JsonValue::Number)
                                            .unwrap_or(JsonValue::Null),
                                        _ => JsonValue::Null,
                                    })
                                    .collect();
                                vals_map.insert(ch.clone(), JsonValue::Array(vals));
                            }
                        }

                        // Temperature
                        if let Some(temp_col) = df.column("temperature") {
                            let temp_id = format!("plate_0{}_temperature", plate_id);
                            let entry = sample_dict
                                .entry(temp_id)
                                .or_insert_with(|| {
                                    let mut e = serde_json::Map::new();
                                    e.insert(
                                        "type".to_string(),
                                        JsonValue::String("timeseries".to_string()),
                                    );
                                    e.insert(
                                        "values".to_string(),
                                        JsonValue::Object(serde_json::Map::new()),
                                    );
                                    e.insert(
                                        "metadata".to_string(),
                                        JsonValue::Object(serde_json::Map::new()),
                                    );
                                    JsonValue::Object(e)
                                });
                            if let Some(vals_map) = entry.get_mut("values").and_then(|v| v.as_object_mut()) {
                                let vals: Vec<JsonValue> = temp_col
                                    .iter()
                                    .map(|v| match v {
                                        Value::Float(f) => serde_json::Number::from_f64(*f)
                                            .map(JsonValue::Number)
                                            .unwrap_or(JsonValue::Null),
                                        Value::Int(i) => JsonValue::Number((*i).into()),
                                        _ => JsonValue::Null,
                                    })
                                    .collect();
                                vals_map.insert(ch.clone(), JsonValue::Array(vals));
                            }
                        }
                    }
                }
            }
        } else if ins_type.contains("flow") {
            // Read flow cytometry data
            for &row_idx in row_indices {
                let row = &samples_df.rows[row_idx];
                let well = s_well_idx
                    .map(|idx| cell_to_string(&row[idx]))
                    .unwrap_or_default();
                let custom_name = s_name_idx
                    .map(|idx| cell_to_string(&row[idx]))
                    .unwrap_or_default();

                let sample_name = if !custom_name.is_empty() && custom_name != "missing" {
                    custom_name
                } else {
                    format!("plate_0{}_{}", plate_id, well.to_lowercase())
                };

                let fcs_loc = cell_to_string(&row[s_loc_idx]);
                let mut resolved_loc = fcs_loc.clone();
                if resolved_loc.contains("$GITHUB_WORKSPACE") {
                    if let Ok(workspace) = std::env::var("GITHUB_WORKSPACE") {
                        resolved_loc = resolved_loc.replace("$GITHUB_WORKSPACE", &workspace);
                    }
                }

                let fcs = read_fcs(&resolved_loc)?;

                // Build values and metadata
                let mut values = serde_json::Map::new();
                let mut metadata = serde_json::Map::new();

                // Use raw FCS channel names from the Samples sheet; the
                // channel_map applies on the output side (mapped_name below).
                let fcs_channels: Vec<String> = if channels.is_empty() {
                    fcs.data.keys().cloned().collect()
                } else {
                    channels.clone()
                };

                for ch in &fcs_channels {
                    let mapped_name = channel_map.get(ch).cloned().unwrap_or_else(|| ch.clone());
                    // Julia's `flow_channel` maps the spreadsheet alias "time"
                    // back to the canonical FCS channel "Time".
                    let lookup = if ch == "time" {
                        "Time".to_string()
                    } else {
                        ch.clone()
                    };
                    if let Some(data) = fcs.data.get(&lookup) {
                        let vals: Vec<JsonValue> = data
                            .iter()
                            .map(|&v| {
                                serde_json::Number::from_f64(v)
                                    .map(JsonValue::Number)
                                    .unwrap_or(JsonValue::Null)
                            })
                            .collect();
                        values.insert(mapped_name.clone(), JsonValue::Array(vals));
                    }

                    if let Some(meta) = fcs.channel_meta.get(&lookup) {
                        let meta_dict = channel_meta_to_dict(meta);
                        let mut meta_json = serde_json::Map::new();
                        for (k, v) in meta_dict {
                            meta_json.insert(k, v);
                        }
                        metadata.insert(mapped_name.clone(), JsonValue::Object(meta_json));
                    }
                }

                // Add raw_metadata. FCSFiles.jl uppercases parameter keys
                // (e.g. `FCSversion` → `FCSVERSION`); match that so the same
                // lookups succeed.
                let raw_meta: serde_json::Map<String, JsonValue> = fcs
                    .params
                    .iter()
                    .map(|(k, v)| (k.to_uppercase(), JsonValue::String(v.clone())))
                    .collect();
                metadata.insert("raw_metadata".to_string(), JsonValue::Object(raw_meta));

                let mut entry = serde_json::Map::new();
                entry.insert(
                    "type".to_string(),
                    JsonValue::String("population".to_string()),
                );
                entry.insert("values".to_string(), JsonValue::Object(values));
                entry.insert("metadata".to_string(), JsonValue::Object(metadata));

                broad_g.push(sample_name.clone());
                sample_dict.insert(sample_name, JsonValue::Object(entry));
            }
        } else {
            return Err(format!("Unknown instrument type: {}", ins_type).into());
        }

        // Add physical plate group
        group_dict.insert(
            format!("plate_0{}", plate_num),
            serde_json::json!({
                "sample_IDs": broad_g,
                "type": "physical",
                "metadata": {"autodefined": "true"}
            }),
        );
    }

    // Build transformations
    let mut trans_dict: IndexMap<String, JsonValue> = IndexMap::new();
    let t_name_idx = find_col_idx(&trans_df, "Name");
    let t_eq_idx = find_col_idx(&trans_df, "Equation");
    if let (Some(name_idx), Some(eq_idx)) = (t_name_idx, t_eq_idx) {
        for row in &trans_df.rows {
            let name = cell_to_string(&row[name_idx]);
            let equation = cell_to_string(&row[eq_idx]);
            if !name.is_empty() {
                trans_dict.insert(
                    name,
                    serde_json::json!({"equation": equation}),
                );
            }
        }
    }

    // Build views
    let mut views_dict: IndexMap<String, JsonValue> = IndexMap::new();
    let v_name_idx = find_col_idx(&views_df, "Name");
    let v_view_idx = find_col_idx(&views_df, "View");
    if let (Some(name_idx), Some(view_idx)) = (v_name_idx, v_view_idx) {
        for row in &views_df.rows {
            let name = cell_to_string(&row[name_idx]);
            let view_str = cell_to_string(&row[view_idx]);
            if !name.is_empty() {
                let view_items: Vec<JsonValue> = view_str
                    .split(',')
                    .map(|s| JsonValue::String(s.trim().to_string()))
                    .collect();
                views_dict.insert(
                    name,
                    serde_json::json!({"data": view_items}),
                );
            }
        }
    }

    // Build metadata
    let metadata = serde_json::json!({
        "description": "",
        "esm_version": env!("CARGO_PKG_VERSION"),
        "schema_version": "0.1.0",
        "date_created": chrono_now(),
        "date_modified": chrono_now(),
    });

    // Assemble result
    let mut result = IndexMap::new();
    result.insert(
        "samples".to_string(),
        serde_json::to_value(&sample_dict)?,
    );
    result.insert(
        "groups".to_string(),
        serde_json::to_value(&group_dict)?,
    );
    result.insert(
        "transformations".to_string(),
        serde_json::to_value(&trans_dict)?,
    );
    result.insert(
        "views".to_string(),
        serde_json::to_value(&views_dict)?,
    );
    result.insert("metadata".to_string(), metadata);

    info!("Data read successfully from {}", file);
    Ok(result)
}

/// Write ESM data to a JSON file.
pub fn write_esm_data(
    data: &IndexMap<String, JsonValue>,
    file: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(data)?;
    std::fs::write(file, json)?;
    info!("ESM written to {}", file);
    Ok(())
}

// ---- Internal helpers ----

struct SheetData {
    headers: Vec<String>,
    rows: Vec<Vec<Data>>,
}

fn read_sheet(
    workbook: &mut Xlsx<BufReader<std::fs::File>>,
    sheet_name: &str,
) -> Result<SheetData, Box<dyn std::error::Error>> {
    let range = workbook.worksheet_range(sheet_name)?;

    let mut rows_iter = range.rows();
    let headers: Vec<String> = rows_iter
        .next()
        .map(|row| row.iter().map(|d| cell_to_string(d)).collect())
        .unwrap_or_default();

    let rows: Vec<Vec<Data>> = rows_iter
        .filter(|row: &&[Data]| {
            // Stop at empty rows
            row.iter().any(|d| !matches!(d, Data::Empty))
        })
        .map(|row: &[Data]| {
            let mut v = row.to_vec();
            // Pad to header length
            while v.len() < headers.len() {
                v.push(Data::Empty);
            }
            v
        })
        .collect();

    Ok(SheetData { headers, rows })
}

fn find_col_idx(sheet: &SheetData, name: &str) -> Option<usize> {
    sheet.headers.iter().position(|h| h == name)
}

fn cell_to_string(d: &Data) -> String {
    match d {
        Data::Int(i) => i.to_string(),
        Data::Float(f) => {
            if *f == (*f as i64) as f64 {
                format!("{}", *f as i64)
            } else {
                f.to_string()
            }
        }
        Data::String(s) => s.clone(),
        Data::Bool(b) => b.to_string(),
        Data::Empty => String::new(),
        Data::DateTime(dt) => dt.to_string(),
        Data::DateTimeIso(s) => s.clone(),
        Data::DurationIso(s) => s.clone(),
        Data::Error(e) => format!("{:?}", e),
    }
}

fn is_valid_well_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let first = name.chars().next().unwrap();
    first.is_ascii_alphabetic() && name.len() <= 4
}

fn chrono_now() -> String {
    // Simple timestamp without external dependency
    use std::time::SystemTime;
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", dur.as_secs())
}
