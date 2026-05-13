use super::{read_into_lines, should_include_channel, PlateReaderParser};
use crate::dataframe::{DataFrame, Value};
use crate::utils::format_channel;
use indexmap::IndexMap;

pub struct Bmg;

impl PlateReaderParser for Bmg {
    fn read(
        &self,
        file: &str,
        channels: Option<&[&str]>,
    ) -> Result<(IndexMap<String, DataFrame>, String), Box<dyn std::error::Error>> {
        let f = read_into_lines(file)?;

        // Detect data lines: lines with more than one unique character (not blank/separator)
        let contains_data: Vec<i32> = f
            .iter()
            .map(|j| {
                let trimmed = j.trim();
                if trimmed.is_empty() || trimmed == "sep=," {
                    0
                } else {
                    let chars: std::collections::HashSet<char> = trimmed.chars().collect();
                    if chars.len() > 1 { 1 } else { 0 }
                }
            })
            .collect();

        let rl: Vec<usize> = (0..contains_data.len())
            .map(|i| super::runlength(&contains_data, i))
            .collect();

        let max_rl = *rl.iter().max().unwrap_or(&0);
        if max_rl == 0 {
            return Err("No data found in BMG file.".into());
        }

        let data_locations: Vec<usize> = rl
            .iter()
            .enumerate()
            .filter(|(_, r)| **r == max_rl)
            .map(|(i, _)| i)
            .collect();

        let raw_metadata = if data_locations[0] > 0 {
            f[0..data_locations[0] - 1].join("\n")
        } else {
            String::new()
        };

        // Extract data blocks
        let mut data_blocks: Vec<Vec<String>> = Vec::new();
        for &loc in &data_locations {
            let mut table = Vec::new();
            table.push(f[loc].clone());
            // Skip one line after header (blank or separator), then collect data
            let start = loc + 2;
            for j in start..f.len() {
                if contains_data[j] == 1 {
                    table.push(f[j].clone());
                } else {
                    break;
                }
            }
            data_blocks.push(table);
        }

        let mut out = IndexMap::new();
        for block in &data_blocks {
            if block.is_empty() {
                continue;
            }

            // Channel name is the second field of the first line
            let first_fields: Vec<&str> = block[0].split(',').collect();
            let channel_raw = first_fields.get(1).map(|s| s.trim()).unwrap_or("unknown");
            let channel = format_channel(channel_raw);

            if !should_include_channel(&channel, channels) {
                continue;
            }

            // Transpose data: each row after header is a well's time series
            let data_rows: Vec<Vec<&str>> = block[1..]
                .iter()
                .map(|row| row.split(',').map(|s| s.trim()).collect::<Vec<_>>())
                .collect();

            if data_rows.is_empty() {
                continue;
            }

            // Transpose: rows become columns
            let nrows = data_rows.iter().map(|r| r.len()).max().unwrap_or(0);

            let mut transposed: Vec<String> = Vec::new();
            for row_idx in 0..nrows {
                let mut new_row = Vec::new();
                for col in &data_rows {
                    new_row.push(col.get(row_idx).copied().unwrap_or(""));
                }
                transposed.push(new_row.join(","));
            }

            if transposed.is_empty() {
                continue;
            }

            let headers: Vec<String> = transposed[0]
                .split(',')
                .map(|s: &str| s.trim().to_string())
                .collect();

            let mut df = DataFrame::new();
            let num_data_rows = transposed.len() - 1;

            // Time column (first column of data rows)
            let mut time_vals = Vec::new();
            for row in &transposed[1..] {
                let fields: Vec<&str> = row.split(',').collect();
                let time_str = fields.first().map(|s| s.trim()).unwrap_or("0");
                let time_f: f64 = time_str.parse().unwrap_or(0.0);
                // Convert seconds to milliseconds, handling floating point
                let time_ms = (time_f * 1000.0).round() as i64;
                time_vals.push(Value::Int(time_ms));
            }
            df.insert_column("time", time_vals);

            // Temperature column (NaN - BMG doesn't record temperature in this format)
            df.insert_column(
                "temperature",
                vec![Value::Float(f64::NAN); num_data_rows],
            );

            // Data columns (wells)
            for (col_idx, header) in headers.iter().enumerate().skip(1) {
                let mut vals = Vec::new();
                for row in &transposed[1..] {
                    let fields: Vec<&str> = row.split(',').collect();
                    let field = fields.get(col_idx).map(|s| s.trim()).unwrap_or("");
                    if field.is_empty() {
                        vals.push(Value::Missing);
                    } else if let Ok(v) = field.parse::<f64>() {
                        vals.push(Value::Float(v));
                    } else if let Ok(v) = field.parse::<i64>() {
                        vals.push(Value::Int(v));
                    } else {
                        vals.push(Value::Missing);
                    }
                }
                df.insert_column(header.clone(), vals);
            }

            out.insert(channel, df);
        }

        Ok((out, raw_metadata))
    }
}
