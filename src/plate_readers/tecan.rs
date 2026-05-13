use super::{should_include_channel, PlateReaderParser};
use crate::dataframe::{DataFrame, Value};
use crate::utils::format_channel;
use calamine::{open_workbook, Data, Reader, Xlsx};
use indexmap::IndexMap;
use std::io::BufReader;

pub struct Tecan;

impl PlateReaderParser for Tecan {
    fn read(
        &self,
        file: &str,
        channels: Option<&[&str]>,
    ) -> Result<(IndexMap<String, DataFrame>, String), Box<dyn std::error::Error>> {
        let mut workbook: Xlsx<BufReader<std::fs::File>> = open_workbook(file)?;

        let range = workbook.worksheet_range("Sheet2")?;

        let rows: Vec<Vec<Data>> = range.rows().map(|r: &[Data]| r.to_vec()).collect();
        let nrows = rows.len();

        // Find which rows have data (non-empty first column)
        let is_data: Vec<i32> = rows
            .iter()
            .map(|row: &Vec<Data>| {
                if row.is_empty() {
                    return 0;
                }
                match &row[0] {
                    Data::Empty => 0,
                    _ => 1,
                }
            })
            .collect();

        let rl: Vec<usize> = (0..is_data.len())
            .map(|i| super::runlength(&is_data, i))
            .collect();

        // Find "Cycle Nr." markers - the channel name is one row above
        let cycle_nr_locations: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter_map(|(i, row): (usize, &Vec<Data>)| {
                if !row.is_empty() {
                    if let Data::String(s) = &row[0] {
                        if s == "Cycle Nr." {
                            return Some(i);
                        }
                    }
                }
                None
            })
            .collect();

        // Build raw metadata from rows before first data location
        let mut raw_metadata = String::new();
        if let Some(&first_loc) = cycle_nr_locations.first() {
            let meta_end = if first_loc > 1 { first_loc - 1 } else { 0 };
            for i in 0..meta_end {
                let row_strs: Vec<String> =
                    rows[i].iter().map(|d: &Data| data_to_string(d)).collect();
                raw_metadata.push_str(&row_strs.join(","));
                raw_metadata.push('\n');
            }
        }
        // Add last row metadata
        if nrows > 0 {
            let row_strs: Vec<String> = rows[nrows - 1]
                .iter()
                .map(|d: &Data| data_to_string(d))
                .collect();
            raw_metadata.push_str(&row_strs.join(","));
        }
        raw_metadata = raw_metadata.replace("missing", "");

        let mut out = IndexMap::new();

        for &cycle_row in &cycle_nr_locations {
            // Channel name is one row above
            let channel_row = if cycle_row > 0 { cycle_row - 1 } else { continue };
            let channel_raw = if !rows[channel_row].is_empty() {
                data_to_string(&rows[channel_row][0])
            } else {
                continue;
            };
            let channel = format_channel(&channel_raw);

            if !should_include_channel(&channel, channels) {
                continue;
            }

            // The block structure in the XLSX is:
            // Row channel_row: [channel_name, ...]
            // Row cycle_row: ["Cycle Nr.", 1, 2, 3, ...] (one per timepoint)
            // Row cycle_row+1: ["Time [s]", t1, t2, t3, ...]
            // Row cycle_row+2: ["Temp. [°C]", temp1, temp2, ...]
            // Row cycle_row+3: ["A1", val1, val2, ...]
            // ...
            // We need to transpose this so columns are wells and rows are timepoints.

            let block_len: usize = if cycle_row < rl.len() {
                rl[cycle_row]
            } else {
                continue;
            };
            let block_end: usize = (cycle_row + block_len).min(nrows);

            // Collect the block rows from cycle_row to block_end
            let block = &rows[cycle_row..block_end];
            if block.len() < 3 {
                continue; // Need at least Cycle Nr., Time, and one data row
            }

            // Number of timepoints = number of columns - 1 (first column is labels)
            let num_timepoints = block[0].len().saturating_sub(1);
            if num_timepoints == 0 {
                continue;
            }

            // Row labels (col 0 of each row): "Cycle Nr.", "Time [s]", "Temp. [°C]", "A1", ...
            let mut df = DataFrame::new();

            // Skip row 0 (Cycle Nr.) - process rows 1+ (Time, Temp, wells)
            for row_idx in 1..block.len() {
                let label = if !block[row_idx].is_empty() {
                    data_to_string(&block[row_idx][0])
                } else {
                    continue;
                };

                let name = if label.contains("Time") {
                    "time".to_string()
                } else if label.contains("Temp") {
                    "temperature".to_string()
                } else {
                    label
                };

                // Collect values from columns 1..num_timepoints+1
                let mut vals: Vec<Value> = Vec::new();
                for col_idx in 1..=num_timepoints {
                    if col_idx < block[row_idx].len() {
                        vals.push(data_to_value(&block[row_idx][col_idx]));
                    } else {
                        vals.push(Value::Missing);
                    }
                }

                // Convert time to milliseconds (Tecan time is in seconds)
                if name == "time" {
                    vals = vals
                        .into_iter()
                        .map(|v: Value| match v {
                            Value::Float(f) => {
                                let ms = (f * 1000.0).round() as i64;
                                Value::Int(ms)
                            }
                            Value::Int(i) => Value::Int(i * 1000),
                            other => other,
                        })
                        .collect();
                }

                df.insert_column(name, vals);
            }

            out.insert(channel, df);
        }

        Ok((out, raw_metadata))
    }
}

fn data_to_string(d: &Data) -> String {
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
        Data::DateTime(dt) => dt.to_string(),
        Data::DateTimeIso(s) => s.clone(),
        Data::DurationIso(s) => s.clone(),
        Data::Error(e) => format!("{:?}", e),
        Data::Empty => String::new(),
    }
}

fn data_to_value(d: &Data) -> Value {
    match d {
        Data::Int(i) => Value::Int(*i),
        Data::Float(f) => Value::Float(*f),
        Data::String(s) => {
            if let Ok(v) = s.parse::<f64>() {
                Value::Float(v)
            } else if let Ok(v) = s.parse::<i64>() {
                Value::Int(v)
            } else {
                Value::Str(s.clone())
            }
        }
        Data::Bool(b) => Value::Bool(*b),
        Data::Empty => Value::Missing,
        _ => Value::Str(data_to_string(d)),
    }
}
