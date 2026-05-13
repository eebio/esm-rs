use super::{
    parse_time_to_ms, read_into_lines, read_standard, should_include_channel, PlateReaderParser,
};
use crate::dataframe::{DataFrame, Value};
use crate::utils::format_channel;
use indexmap::IndexMap;

pub struct BioTek;

impl PlateReaderParser for BioTek {
    fn read(
        &self,
        file: &str,
        channels: Option<&[&str]>,
    ) -> Result<(IndexMap<String, DataFrame>, String), Box<dyn std::error::Error>> {
        let (data, mut raw_metadata) = read_standard(file, 2)?;

        // Add possible extra metadata from end of file (Results section)
        let f = read_into_lines(file)?;
        let mut results_found = false;
        for line in &f {
            if line.contains("Results") {
                results_found = true;
            }
            if results_found {
                raw_metadata.push('\n');
                raw_metadata.push_str(line);
            }
        }

        let mut out = IndexMap::new();
        for block in &data {
            if block.len() < 3 {
                continue;
            }

            let channel = format_channel(&block[0]);
            if !should_include_channel(&channel, channels) {
                continue;
            }

            // Remove channel name from the header row (row index 1)
            let mut header_line = block[1].replace(&format!(" {}", &block[0]), "");
            // Also try without space prefix
            header_line = header_line.replace(&block[0], "");

            // Replace OVRFLW with empty in data rows
            let mut cleaned: Vec<String> = Vec::new();
            cleaned.push(header_line);
            for line in &block[2..] {
                cleaned.push(line.replace("OVRFLW", ""));
            }

            // Detect delimiter (tab or comma)
            let delim = if cleaned[0].contains('\t') {
                "\t"
            } else {
                ","
            };

            let df = parse_biotek_block(&cleaned, delim)?;
            out.insert(channel, df);
        }

        Ok((out, raw_metadata))
    }
}

fn parse_biotek_block(
    lines: &[String],
    delim: &str,
) -> Result<DataFrame, Box<dyn std::error::Error>> {
    if lines.is_empty() {
        return Err("Empty data block".into());
    }

    let headers: Vec<String> = lines[0]
        .split(delim)
        .map(|s| s.trim().to_string())
        .collect();

    let ncols = headers.len();
    let mut columns: Vec<Vec<Value>> = vec![Vec::new(); ncols];

    for line in &lines[1..] {
        let fields: Vec<&str> = line.split(delim).collect();
        for (j, col) in columns.iter_mut().enumerate().take(ncols) {
            let field = fields.get(j).map(|s| s.trim()).unwrap_or("");
            if field.is_empty() {
                col.push(Value::Missing);
            } else if let Some(ms) = parse_time_to_ms(field) {
                col.push(Value::Int(ms));
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
        } else if i == 1 && (header.contains("emperature") || header.contains("emp") || header.starts_with('T')) {
            "temperature".to_string()
        } else {
            header.clone()
        };
        df.insert_column(name, columns[i].clone());
    }

    Ok(df)
}
