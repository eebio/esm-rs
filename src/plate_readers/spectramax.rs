use super::{
    correct_data_length, extract_spectramax_channel, parse_time_to_ms, read_standard,
    should_include_channel, PlateReaderParser,
};
use crate::dataframe::{DataFrame, Value};
use crate::utils::format_channel;
use indexmap::IndexMap;

pub struct SpectraMax;

impl PlateReaderParser for SpectraMax {
    fn read(
        &self,
        file: &str,
        channels: Option<&[&str]>,
    ) -> Result<(IndexMap<String, DataFrame>, String), Box<dyn std::error::Error>> {
        let (mut data, _) = read_standard(file, 1)?;
        correct_data_length(&mut data, "\t");

        let mut out = IndexMap::new();
        let mut raw_metadata = String::new();

        // Process channel metadata from the first line of each block
        let mut all_channels: Vec<(String, String)> = Vec::new();
        for block in &data {
            if block.is_empty() {
                continue;
            }
            raw_metadata.push_str(&block[0]);
            raw_metadata.push('\n');
            let (channel, mtype) = extract_spectramax_channel(&block[0]);
            all_channels.push((channel, mtype));
        }

        // Handle duplicate channel names
        let mut channel_counts: IndexMap<String, usize> = IndexMap::new();
        for i in 0..all_channels.len() {
            let channel = all_channels[i].0.clone();
            let mtype = all_channels[i].1.clone();
            let count = channel_counts.entry(channel.clone()).or_insert(0);
            *count += 1;

            if *count > 1 {
                if mtype == "Absorbance" {
                    let wavelengths: Vec<&str> = channel.split(' ').collect();
                    if i < wavelengths.len() {
                        all_channels[i].0 = wavelengths[i].to_string();
                    }
                    if *count == 2 {
                        if let Some(j) = all_channels[..i]
                            .iter()
                            .position(|(c, _)| *c == channel)
                        {
                            all_channels[j].0 = wavelengths[0].to_string();
                        }
                    }
                } else {
                    all_channels[i].0 = format!("{}_{}", channel, count);
                    if *count == 2 {
                        if let Some(j) = all_channels[..i]
                            .iter()
                            .position(|(c, _)| *c == channel)
                        {
                            all_channels[j].0 = format!("{}_1", channel);
                        }
                    }
                }
            }
        }

        // Process each data block
        for (i, block) in data.iter().enumerate() {
            if i >= all_channels.len() || block.len() < 2 {
                continue;
            }

            let channel = format_channel(&all_channels[i].0);
            if !should_include_channel(&channel, channels) {
                continue;
            }

            // Replace #SAT with NaN in data rows
            let mut cleaned_block: Vec<String> = block[1..].to_vec();
            for line in &mut cleaned_block {
                *line = line.replace("#SAT", "NaN");
            }

            // Parse the block: first line is headers, rest is data
            let df = parse_spectramax_block(&cleaned_block)?;
            out.insert(channel, df);
        }

        Ok((out, raw_metadata))
    }
}

fn parse_spectramax_block(
    lines: &[String],
) -> Result<DataFrame, Box<dyn std::error::Error>> {
    if lines.is_empty() {
        return Err("Empty data block".into());
    }

    // Parse headers
    let headers: Vec<String> = lines[0]
        .split('\t')
        .map(|s| s.trim().to_string())
        .collect();

    let ncols = headers.len();
    let mut columns: Vec<Vec<Value>> = vec![Vec::new(); ncols];

    // Parse data rows
    for line in &lines[1..] {
        let fields: Vec<&str> = line.split('\t').collect();
        for (j, col) in columns.iter_mut().enumerate().take(ncols) {
            let field = fields.get(j).map(|s| s.trim()).unwrap_or("");
            if field.is_empty() {
                col.push(Value::Missing);
            } else if let Some(ms) = parse_time_to_ms(field) {
                col.push(Value::Int(ms));
            } else if field == "NaN" {
                col.push(Value::Float(f64::NAN));
            } else if let Ok(v) = field.parse::<f64>() {
                col.push(Value::Float(v));
            } else if let Ok(v) = field.parse::<i64>() {
                col.push(Value::Int(v));
            } else {
                col.push(Value::Str(field.to_string()));
            }
        }
    }

    let mut df = DataFrame::new();
    for (i, header) in headers.iter().enumerate() {
        let name = if i == 0 {
            "time".to_string()
        } else if i == 1 && header.contains("emperature") {
            "temperature".to_string()
        } else {
            header.clone()
        };
        df.insert_column(name, columns[i].clone());
    }

    Ok(df)
}
