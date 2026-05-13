//! Plate reader file parsing for various instrument formats.

pub mod biotek;
pub mod bmg;
pub mod generic;
pub mod spectramax;
pub mod tecan;

use crate::dataframe::DataFrame;
use indexmap::IndexMap;
use regex::Regex;

/// Trait for plate reader parsers.
pub trait PlateReaderParser {
    /// Read data from a file.
    ///
    /// Returns a map of channel name -> DataFrame, and raw metadata string.
    fn read(
        &self,
        file: &str,
        channels: Option<&[&str]>,
    ) -> Result<(IndexMap<String, DataFrame>, String), Box<dyn std::error::Error>>;
}

/// Read a plate reader file given a type string.
pub fn read_multipr_file(
    file: &str,
    ptype: &str,
    channels: &[String],
    channel_map: &IndexMap<String, String>,
) -> Result<(IndexMap<String, DataFrame>, String), Box<dyn std::error::Error>> {
    let channel_refs: Vec<&str> = channels.iter().map(|s| s.as_str()).collect();
    let channel_opt = if channels.is_empty() {
        None
    } else {
        Some(channel_refs.as_slice())
    };

    let (o_dict, raw_metadata) = match ptype.to_lowercase().as_str() {
        "spectramax" => spectramax::SpectraMax.read(file, channel_opt)?,
        "biotek" => biotek::BioTek.read(file, channel_opt)?,
        "tecan" => tecan::Tecan.read(file, channel_opt)?,
        "bmg" => bmg::Bmg.read(file, channel_opt)?,
        "generic" => generic::GenericTabular.read(file, channel_opt)?,
        _ => return Err(format!("Unknown plate reader type: {}.", ptype).into()),
    };

    // Check requested channels exist
    if !channels.is_empty() {
        for ch in channels {
            if !o_dict.contains_key(ch) {
                return Err(
                    format!("Requested channel {} not found in file {}.", ch, file).into(),
                );
            }
        }
    }

    // Apply channel map
    let mut mapped = IndexMap::new();
    for (k, v) in o_dict {
        let new_key = channel_map.get(&k).unwrap_or(&k).clone();
        mapped.insert(new_key, v);
    }

    // Remove all-missing columns
    for (_name, df) in mapped.iter_mut() {
        *df = df.remove_all_missing_columns();
    }

    Ok((mapped, raw_metadata))
}

// ---- Shared utilities for plate reader parsing ----

/// Try to read a file with multiple encodings, return lines.
pub fn read_into_lines(file: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(file)?;

    // Try encodings in order
    let encodings: &[&encoding_rs::Encoding] = &[
        encoding_rs::UTF_16LE,
        encoding_rs::UTF_16BE,
        encoding_rs::UTF_8,
        encoding_rs::WINDOWS_1252,
        encoding_rs::ISO_8859_2, // Close enough to ISO-8859-1 for our purposes
    ];

    // Check for UTF-16 BOM first
    let decoded: String = if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE {
        let (cow, _, _) = encoding_rs::UTF_16LE.decode(&bytes);
        cow.into_owned()
    } else if bytes.len() >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF {
        let (cow, _, _) = encoding_rs::UTF_16BE.decode(&bytes);
        cow.into_owned()
    } else {
        let mut result: Option<String> = None;
        for enc in encodings {
            let (cow, _, had_errors) = enc.decode(&bytes);
            if !had_errors && cow.contains("Time") {
                result = Some(cow.into_owned());
                break;
            }
        }
        if result.is_none() {
            for enc in encodings {
                let (cow, _, had_errors) = enc.decode(&bytes);
                if !had_errors {
                    result = Some(cow.into_owned());
                    break;
                }
            }
        }
        result.ok_or_else(|| {
            format!("Could not determine encoding for file {}", file)
        })?
    };

    let s = decoded.replace("\r\n", "\n").replace('\r', "\n");
    Ok(s.split('\n').map(|line: &str| line.to_string()).collect())
}

/// Calculate the run-length of consecutive 1s starting at position `i`.
pub fn runlength(a: &[i32], i: usize) -> usize {
    if i >= a.len() {
        return 0;
    }
    if a[i] == 1 {
        1 + runlength(a, i + 1)
    } else {
        0
    }
}

/// Read a standard plate reader file format.
/// Returns (data_blocks, raw_metadata) where each block is a Vec<String> of lines.
pub fn read_standard(
    file: &str,
    offset: usize,
) -> Result<(Vec<Vec<String>>, String), Box<dyn std::error::Error>> {
    let f = read_into_lines(file)?;
    let time_re = Regex::new(r"\d{1,2}:\d\d:\d\d")?;

    let contains_time: Vec<i32> = f.iter().map(|j| if time_re.is_match(j) { 1 } else { 0 }).collect();

    let rl: Vec<usize> = (0..contains_time.len())
        .map(|i| runlength(&contains_time, i))
        .collect();

    let max_rl = *rl.iter().max().unwrap_or(&0);
    if max_rl == 0 {
        return Err("No time data found in file.".into());
    }

    let data_locations: Vec<usize> = rl
        .iter()
        .enumerate()
        .filter(|(_, r)| **r == max_rl)
        .map(|(i, _)| i)
        .collect();

    let raw_metadata = if data_locations[0] >= 2 {
        f[0..data_locations[0] - 1].join("\n")
    } else {
        String::new()
    };

    let mut data = Vec::new();
    for &loc in &data_locations {
        let mut table = Vec::new();
        // Push header: find the line containing "Time" before loc
        for j in (0..loc).rev() {
            if f[j].contains("Time") || f[j].contains("time") {
                if j >= offset {
                    table.push(f[j - offset].clone());
                }
                table.push(f[j].clone());
                break;
            }
        }
        // Push data rows
        for j in loc..f.len() {
            if contains_time[j] == 1 {
                table.push(f[j].clone());
            } else {
                break;
            }
        }
        data.push(table);
    }

    Ok((data, raw_metadata))
}

/// Normalize data rows so all have the same number of columns.
pub fn correct_data_length(data: &mut Vec<Vec<String>>, delim: &str) {
    for block in data.iter_mut() {
        let col_lengths: Vec<usize> = block.iter().map(|row| row.split(delim).count()).collect();
        let max_length = *col_lengths.iter().max().unwrap_or(&0);
        for (i, &cur_len) in col_lengths.iter().enumerate() {
            if cur_len < max_length {
                let num_missing = max_length - cur_len;
                block[i].push_str(&delim.repeat(num_missing));
            }
        }
    }
}

/// Parse a time string like "HH:MM:SS" or "HH:MM:SS.mmm" or "H:MM:SS.mmm" into milliseconds.
pub fn parse_time_to_ms(time_str: &str) -> Option<i64> {
    let parts: Vec<&str> = time_str.split(':').collect();
    if parts.len() < 3 {
        return None;
    }
    let hours: i64 = parts[0].trim().parse().ok()?;
    let minutes: i64 = parts[1].trim().parse().ok()?;
    let sec_parts: Vec<&str> = parts[2].split('.').collect();
    let seconds: i64 = sec_parts[0].trim().parse().ok()?;
    let millis: i64 = if sec_parts.len() > 1 {
        let ms_str = sec_parts[1].trim();
        // Pad or truncate to 3 digits
        let padded = format!("{:0<3}", &ms_str[..ms_str.len().min(3)]);
        padded.parse().unwrap_or(0)
    } else {
        0
    };

    Some(hours * 3600 * 1000 + minutes * 60 * 1000 + seconds * 1000 + millis)
}

/// Parse a tab-delimited data block into a DataFrame.
/// First row is headers, subsequent rows are data.
/// Time column is converted to milliseconds.
pub fn parse_data_block_to_df(
    lines: &[String],
    delim: &str,
) -> Result<DataFrame, Box<dyn std::error::Error>> {
    if lines.is_empty() {
        return Err("Empty data block".into());
    }

    // Parse headers
    let headers: Vec<String> = lines[0].split(delim).map(|s| s.trim().to_string()).collect();

    // Initialize columns
    let ncols = headers.len();
    let mut columns: Vec<Vec<crate::dataframe::Value>> = vec![Vec::new(); ncols];

    // Parse data rows
    for line in &lines[1..] {
        let fields: Vec<&str> = line.split(delim).collect();
        for (j, col) in columns.iter_mut().enumerate().take(ncols) {
            let field = fields.get(j).map(|s| s.trim()).unwrap_or("");
            if field.is_empty() {
                col.push(crate::dataframe::Value::Missing);
            } else if let Some(ms) = parse_time_to_ms(field) {
                // Time field
                col.push(crate::dataframe::Value::Int(ms));
            } else if let Ok(v) = field.parse::<f64>() {
                col.push(crate::dataframe::Value::Float(v));
            } else if let Ok(v) = field.parse::<i64>() {
                col.push(crate::dataframe::Value::Int(v));
            } else {
                col.push(crate::dataframe::Value::Str(field.to_string()));
            }
        }
    }

    let mut df = DataFrame::new();
    for (i, header) in headers.iter().enumerate() {
        let name = if i == 0 {
            "time".to_string()
        } else if i == 1 && (header.contains("emperature") || header.contains("emp")) {
            "temperature".to_string()
        } else {
            header.clone()
        };
        df.insert_column(name, columns[i].clone());
    }

    Ok(df)
}

/// Check if a channel should be included based on the filter.
pub fn should_include_channel(channel: &str, channels: Option<&[&str]>) -> bool {
    match channels {
        None => true,
        Some(ch_list) => ch_list.contains(&channel),
    }
}

/// Extract channel name from SpectraMax metadata line.
pub fn extract_spectramax_channel(meta_line: &str) -> (String, String) {
    let parts: Vec<String> = meta_line
        .split('\t')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if parts.len() < 7 {
        return (String::new(), String::new());
    }

    let measurement_type = &parts[5]; // "Fluorescence" or "Absorbance"
    let channel = if measurement_type == "Fluorescence" {
        // For fluorescence: channel is at parts[13] + "_" + parts[17]
        if parts.len() > 17 {
            format!("{}_{}", parts[13], parts[17])
        } else if parts.len() > 13 {
            parts[13].clone()
        } else {
            "unknown".to_string()
        }
    } else {
        // For absorbance: channel is at parts[12]
        if parts.len() > 12 {
            parts[12].clone()
        } else {
            "unknown".to_string()
        }
    };

    (channel, measurement_type.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_time_to_ms() {
        assert_eq!(parse_time_to_ms("00:00:00"), Some(0));
        assert_eq!(parse_time_to_ms("01:00:00"), Some(3600000));
        assert_eq!(parse_time_to_ms("00:10:00"), Some(600000));
        assert_eq!(parse_time_to_ms("00:00:01"), Some(1000));
        assert_eq!(parse_time_to_ms("0:09:04.714"), Some(544714));
        assert_eq!(parse_time_to_ms("0:19:04.314"), Some(1144314));
    }

    #[test]
    fn test_runlength() {
        assert_eq!(runlength(&[1, 1, 1, 0, 1], 0), 3);
        assert_eq!(runlength(&[0, 1, 1], 0), 0);
        assert_eq!(runlength(&[1, 1, 1], 0), 3);
    }

    #[test]
    fn test_read_multipr_unknown_type() {
        let result = read_multipr_file("anyfile", "random_string", &["OD".to_string()], &IndexMap::new());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown plate reader type: random_string"));
    }
}
