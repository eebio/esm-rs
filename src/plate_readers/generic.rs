use super::{parse_time_to_ms, should_include_channel, PlateReaderParser};
use crate::dataframe::{DataFrame, Value};
use crate::utils::format_channel;
use indexmap::IndexMap;
use std::path::Path;

pub struct GenericTabular;

impl PlateReaderParser for GenericTabular {
    fn read(
        &self,
        file: &str,
        channels: Option<&[&str]>,
    ) -> Result<(IndexMap<String, DataFrame>, String), Box<dyn std::error::Error>> {
        let dir = Path::new(file);
        if !dir.is_dir() {
            return Err(format!("{} is not a directory", file).into());
        }

        let mut out = IndexMap::new();

        let mut entries: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let path = entry.path();
            let filename = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let channel = format_channel(filename);

            if !should_include_channel(&channel, channels) {
                continue;
            }

            // Read the file
            let content = std::fs::read_to_string(&path)?;
            let lines: Vec<&str> = content.lines().collect();
            if lines.is_empty() {
                continue;
            }

            // Detect delimiter
            let delim = if lines[0].contains('\t') {
                '\t'
            } else {
                ','
            };

            // Parse headers
            let headers: Vec<String> = lines[0]
                .split(delim)
                .map(|s| s.trim().to_string())
                .collect();

            let ncols = headers.len();
            let mut columns: Vec<Vec<Value>> = vec![Vec::new(); ncols];

            // Parse data
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
                let name = if header.to_lowercase() == "time" {
                    "time".to_string()
                } else if header.to_lowercase() == "temperature" {
                    "temperature".to_string()
                } else {
                    header.clone()
                };
                df.insert_column(name, columns[i].clone());
            }

            out.insert(channel, df);
        }

        Ok((out, String::new()))
    }
}
