//! FCS 3.0 file reader for flow cytometry data.

use crate::utils::format_channel;
use indexmap::IndexMap;
use std::io::{Read, Seek, SeekFrom};

/// Parsed FCS file data.
#[derive(Debug, Clone)]
pub struct FcsFile {
    /// Channel name -> Vec<f64> of event values.
    pub data: IndexMap<String, Vec<f64>>,
    /// All TEXT segment key-value parameters.
    pub params: IndexMap<String, String>,
    /// Per-channel metadata.
    pub channel_meta: IndexMap<String, ChannelMeta>,
}

/// Metadata for a single FCS channel/parameter.
#[derive(Debug, Clone)]
pub struct ChannelMeta {
    pub name: Option<String>,
    pub short_name: Option<String>,
    pub amp_type: Option<String>,
    pub range: Option<String>,
    pub filter: Option<String>,
    pub amp_gain: Option<String>,
    pub ex_wav: Option<String>,
    pub ex_pow: Option<String>,
    pub perc_em: Option<String>,
    pub det_type: Option<String>,
    pub det_volt: Option<String>,
    pub bits: Option<String>,
}

/// Read an FCS file (supports FCS 2.0 and 3.0).
pub fn read_fcs(path: &str) -> Result<FcsFile, Box<dyn std::error::Error>> {
    let mut file = std::fs::File::open(path)?;
    let mut header = [0u8; 58];
    file.read_exact(&mut header)?;

    let version = std::str::from_utf8(&header[0..6])?.trim();
    if !version.starts_with("FCS") {
        return Err(format!("Not an FCS file: {}", version).into());
    }

    // Parse header offsets
    let text_begin: u64 = std::str::from_utf8(&header[10..18])?.trim().parse()?;
    let text_end: u64 = std::str::from_utf8(&header[18..26])?.trim().parse()?;
    let data_begin: u64 = std::str::from_utf8(&header[26..34])?.trim().parse()?;
    let data_end: u64 = std::str::from_utf8(&header[34..42])?.trim().parse()?;

    // Read TEXT segment
    file.seek(SeekFrom::Start(text_begin))?;
    let text_len = (text_end - text_begin + 1) as usize;
    let mut text_buf = vec![0u8; text_len];
    file.read_exact(&mut text_buf)?;
    let text = String::from_utf8_lossy(&text_buf);

    // Parse TEXT segment: first char is delimiter
    let delim = text.chars().next().ok_or("Empty TEXT segment")?;
    let params = parse_text_segment(&text[1..], delim);

    // Get data parameters
    let n_params: usize = params
        .get("$PAR")
        .ok_or("Missing $PAR")?
        .trim()
        .parse()?;
    let n_events: usize = params
        .get("$TOT")
        .ok_or("Missing $TOT")?
        .trim()
        .parse()?;
    let data_type = params
        .get("$DATATYPE")
        .map(|s| s.trim().to_uppercase())
        .unwrap_or_else(|| "F".to_string());
    let byte_order = params
        .get("$BYTEORD")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "1,2,3,4".to_string());
    let is_big_endian = byte_order.starts_with("4,3,2,1");

    // Check for data offsets in TEXT segment (override header 0 values)
    let actual_data_begin = if data_begin == 0 {
        params
            .get("$BEGINDATA")
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(data_begin)
    } else {
        data_begin
    };
    let actual_data_end = if data_end == 0 {
        params
            .get("$ENDDATA")
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(data_end)
    } else {
        data_end
    };

    // Get channel names and bits
    let mut channel_names = Vec::new();
    let mut bits_per_param = Vec::new();
    for p in 1..=n_params {
        let name = params
            .get(&format!("$P{}N", p))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| format!("P{}", p));
        let bits: usize = params
            .get(&format!("$P{}B", p))
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(32);
        channel_names.push(name);
        bits_per_param.push(bits);
    }

    // Read DATA segment
    file.seek(SeekFrom::Start(actual_data_begin))?;
    let data_len = (actual_data_end - actual_data_begin + 1) as usize;
    let mut data_buf = vec![0u8; data_len];
    file.read_exact(&mut data_buf)?;

    // Parse data based on type
    let mut data: IndexMap<String, Vec<f64>> = IndexMap::new();
    for name in &channel_names {
        data.insert(format_channel(name), Vec::with_capacity(n_events));
    }

    match data_type.as_str() {
        "F" => {
            // Float data (32-bit IEEE)
            let bytes_per_event: usize = bits_per_param.iter().map(|b| b / 8).sum();
            for event in 0..n_events {
                let event_offset = event * bytes_per_event;
                let mut param_offset = 0;
                for (p, name) in channel_names.iter().enumerate() {
                    let bits = bits_per_param[p];
                    let byte_count = bits / 8;
                    let start = event_offset + param_offset;
                    let end = start + byte_count;

                    if end > data_buf.len() {
                        break;
                    }

                    let value = if byte_count == 4 {
                        let bytes: [u8; 4] = data_buf[start..end].try_into()?;
                        if is_big_endian {
                            f32::from_be_bytes(bytes) as f64
                        } else {
                            f32::from_le_bytes(bytes) as f64
                        }
                    } else if byte_count == 8 {
                        let bytes: [u8; 8] = data_buf[start..end].try_into()?;
                        if is_big_endian {
                            f64::from_be_bytes(bytes)
                        } else {
                            f64::from_le_bytes(bytes)
                        }
                    } else {
                        0.0
                    };

                    data.get_mut(&format_channel(name)).unwrap().push(value);
                    param_offset += byte_count;
                }
            }
        }
        "I" => {
            // Integer data
            let bytes_per_event: usize = bits_per_param.iter().map(|b| b / 8).sum();
            for event in 0..n_events {
                let event_offset = event * bytes_per_event;
                let mut param_offset = 0;
                for (p, name) in channel_names.iter().enumerate() {
                    let bits = bits_per_param[p];
                    let byte_count = bits / 8;
                    let start = event_offset + param_offset;
                    let end = start + byte_count;

                    if end > data_buf.len() {
                        break;
                    }

                    let value = match byte_count {
                        1 => data_buf[start] as f64,
                        2 => {
                            let bytes: [u8; 2] = data_buf[start..end].try_into()?;
                            if is_big_endian {
                                u16::from_be_bytes(bytes) as f64
                            } else {
                                u16::from_le_bytes(bytes) as f64
                            }
                        }
                        4 => {
                            let bytes: [u8; 4] = data_buf[start..end].try_into()?;
                            if is_big_endian {
                                u32::from_be_bytes(bytes) as f64
                            } else {
                                u32::from_le_bytes(bytes) as f64
                            }
                        }
                        _ => 0.0,
                    };

                    data.get_mut(&format_channel(name)).unwrap().push(value);
                    param_offset += byte_count;
                }
            }
        }
        "D" => {
            // Double data (64-bit IEEE)
            let bytes_per_event: usize = n_params * 8;
            for event in 0..n_events {
                let event_offset = event * bytes_per_event;
                for (p, name) in channel_names.iter().enumerate() {
                    let start = event_offset + p * 8;
                    let end = start + 8;
                    if end > data_buf.len() {
                        break;
                    }
                    let bytes: [u8; 8] = data_buf[start..end].try_into()?;
                    let value = if is_big_endian {
                        f64::from_be_bytes(bytes)
                    } else {
                        f64::from_le_bytes(bytes)
                    };
                    data.get_mut(&format_channel(name)).unwrap().push(value);
                }
            }
        }
        _ => return Err(format!("Unsupported FCS data type: {}", data_type).into()),
    }

    // Build channel metadata
    let mut channel_meta = IndexMap::new();
    for (p, name) in channel_names.iter().enumerate() {
        let idx = p + 1;
        let meta = ChannelMeta {
            name: params.get(&format!("$P{}N", idx)).map(|s| s.trim().to_string()),
            short_name: params.get(&format!("$P{}S", idx)).map(|s| s.trim().to_string()),
            amp_type: params.get(&format!("$P{}E", idx)).map(|s| s.trim().to_string()),
            range: params.get(&format!("$P{}R", idx)).map(|s| s.trim().to_string()),
            filter: params.get(&format!("$P{}F", idx)).map(|s| s.trim().to_string()),
            amp_gain: params.get(&format!("$P{}G", idx)).map(|s| s.trim().to_string()),
            ex_wav: params.get(&format!("$P{}L", idx)).map(|s| s.trim().to_string()),
            ex_pow: params.get(&format!("$P{}O", idx)).map(|s| s.trim().to_string()),
            perc_em: params.get(&format!("$P{}P", idx)).map(|s| s.trim().to_string()),
            det_type: params.get(&format!("$P{}T", idx)).map(|s| s.trim().to_string()),
            det_volt: params.get(&format!("$P{}V", idx)).map(|s| s.trim().to_string()),
            bits: params.get(&format!("$P{}B", idx)).map(|s| s.trim().to_string()),
        };
        channel_meta.insert(format_channel(name), meta);
    }

    Ok(FcsFile {
        data,
        params,
        channel_meta,
    })
}

fn parse_text_segment(text: &str, delim: char) -> IndexMap<String, String> {
    let mut params = IndexMap::new();
    let parts: Vec<&str> = text.split(delim).collect();
    let mut i = 0;
    while i + 1 < parts.len() {
        let key = parts[i].trim();
        let value = parts[i + 1].trim();
        if !key.is_empty() {
            params.insert(key.to_string(), value.to_string());
        }
        i += 2;
    }
    params
}

/// Convert ChannelMeta to a metadata dict suitable for ESM format.
pub fn channel_meta_to_dict(meta: &ChannelMeta) -> IndexMap<String, serde_json::Value> {
    let mut dict = IndexMap::new();
    dict.insert(
        "name".to_string(),
        opt_to_json(&meta.name),
    );
    dict.insert(
        "amp_type".to_string(),
        opt_to_json(&meta.amp_type),
    );
    dict.insert(
        "range".to_string(),
        opt_to_json(&meta.range),
    );
    dict.insert(
        "filter".to_string(),
        opt_to_json(&meta.filter),
    );
    dict.insert(
        "amp_gain".to_string(),
        opt_to_json(&meta.amp_gain),
    );
    dict.insert(
        "ex_wav".to_string(),
        opt_to_json(&meta.ex_wav),
    );
    dict.insert(
        "ex_pow".to_string(),
        opt_to_json(&meta.ex_pow),
    );
    dict.insert(
        "perc_em".to_string(),
        opt_to_json(&meta.perc_em),
    );
    dict.insert(
        "name_s".to_string(),
        opt_to_json(&meta.short_name),
    );
    dict.insert(
        "det_type".to_string(),
        opt_to_json(&meta.det_type),
    );
    dict.insert(
        "det_volt".to_string(),
        opt_to_json(&meta.det_volt),
    );
    dict
}

fn opt_to_json(opt: &Option<String>) -> serde_json::Value {
    match opt {
        Some(s) => serde_json::Value::String(s.clone()),
        None => serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_fcs() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("test")
            .join("inputs")
            .join("small.fcs");
        let path = base.to_str().unwrap();

        let fcs = read_fcs(path).unwrap();

        // Check we got the right number of events
        assert_eq!(fcs.params.get("$TOT").unwrap(), "4");

        // Check channels
        assert!(fcs.data.contains_key("FSC_H"));
        assert!(fcs.data.contains_key("SSC_H"));
        assert!(fcs.data.contains_key("FL1_H"));
        assert!(fcs.data.contains_key("FL1_A"));
        assert!(fcs.data.contains_key("Time"));

        // Each channel should have 4 events
        for (name, values) in &fcs.data {
            assert_eq!(values.len(), 4, "Channel {} has {} events", name, values.len());
        }

        // Check metadata
        assert!(fcs.channel_meta.contains_key("FSC_H"));
        assert_eq!(fcs.channel_meta["SSC_H"].amp_type.as_deref(), Some("2,0.01"));
        assert_eq!(fcs.channel_meta["FL1_H"].amp_gain.as_deref(), Some("4"));
        assert_eq!(fcs.channel_meta["FSC_H"].range.as_deref(), Some("1024"));

        // Check actual data values for FSC-H (linear, no gain)
        let fsc = &fcs.data["FSC_H"];
        // Values should be reasonable (positive, within range)
        for v in fsc {
            assert!(*v >= 0.0, "FSC-H value {} should be non-negative", v);
        }
    }
}
