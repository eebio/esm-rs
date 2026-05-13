//! View production and CSV export.

use indexmap::IndexMap;
use std::io::Write;

use crate::dataframe::DataFrame;
use crate::esm_files;
use crate::esm_types::EsmZones;
use crate::expression::{evaluate_expression, ExprResult};
use log::info;

/// Produce views from ESM data.
///
/// `to_out` specifies which views to produce; if empty, all views are produced.
pub fn produce_views(
    es: &EsmZones,
    trans_map: &IndexMap<String, String>,
    to_out: &[&str],
) -> Result<IndexMap<String, ViewResult>, String> {
    let view_keys: Vec<String> = if to_out.is_empty() {
        es.views.keys().cloned().collect()
    } else {
        to_out.iter().map(|s| s.to_string()).collect()
    };

    let mut v_out = IndexMap::new();

    for view_name in &view_keys {
        info!("Producing view {}.", view_name);
        let view_data: &Vec<String> = es
            .views
            .get(view_name)
            .and_then(|v: &IndexMap<String, Vec<String>>| v.get("data"))
            .ok_or_else(|| format!("View {} not found", view_name))?;

        let mut results: Vec<ExprResult> = Vec::new();

        for item in view_data {
            // Check if it's a transformation
            if trans_map.contains_key(item) {
                let result = evaluate_expression(item, es, trans_map)?;
                results.push(result);
            } else if esm_files::group_names(es).contains(&item.to_string()) {
                // It's a group
                let formed = esm_files::form_df(es, &esm_files::filter_row(es, item));
                results.push(ExprResult::Frame(formed));
            } else {
                // Check if it's a sample
                let sample_names = esm_files::sample_names(es);
                let sample_ids: Vec<String> = sample_names
                    .iter()
                    .map(|n| esm_files::get_sample_id(n).to_string())
                    .collect();
                if sample_names.contains(&item.to_string()) || sample_ids.contains(&item.to_string()) {
                    let name_col = es.samples.column("name").unwrap();
                    let matching: Vec<usize> = name_col
                        .iter()
                        .enumerate()
                        .filter_map(|(i, v)| {
                            if let crate::dataframe::Value::Str(n) = v {
                                if n == item || esm_files::get_sample_id(n) == item.as_str() {
                                    Some(i)
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect();
                    let df = esm_files::form_df(es, &matching);
                    results.push(ExprResult::Frame(df));
                } else {
                    return Err(format!(
                        "View {} = {} is not a sample, group or transformation",
                        view_name, item
                    ));
                }
            }
        }

        // Combine results
        let view_result = combine_results(results);
        v_out.insert(view_name.clone(), view_result);
    }

    info!("Views produced.");
    Ok(v_out)
}

/// Combined result of a view.
#[derive(Debug, Clone)]
pub enum ViewResult {
    Frame(DataFrame),
    Matrix(Vec<Vec<f64>>),
    Scalar(f64),
}

fn combine_results(results: Vec<ExprResult>) -> ViewResult {
    if results.is_empty() {
        return ViewResult::Frame(DataFrame::new());
    }

    // Check if all results are numbers/matrices
    let all_numeric = results.iter().all(|r| matches!(r, ExprResult::Number(_) | ExprResult::Int(_)));
    let any_matrix = results.iter().any(|r| matches!(r, ExprResult::Matrix(_)));

    if all_numeric {
        if results.len() == 1 {
            return ViewResult::Scalar(results[0].as_number().unwrap());
        }
        // Multiple numbers - combine as a row
        let vals: Vec<f64> = results.iter().filter_map(|r| r.as_number()).collect();
        let matrix = vec![vals];
        return ViewResult::Matrix(matrix);
    }

    if any_matrix {
        // Combine matrices horizontally
        let mut combined_rows: Vec<Vec<f64>> = Vec::new();
        for result in &results {
            match result {
                ExprResult::Matrix(m) => {
                    if combined_rows.is_empty() {
                        combined_rows = m.clone();
                    } else {
                        for (i, row) in m.iter().enumerate() {
                            if i < combined_rows.len() {
                                combined_rows[i].extend(row);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        return ViewResult::Matrix(combined_rows);
    }

    // Combine DataFrames horizontally
    let mut combined = DataFrame::new();
    for result in &results {
        if let ExprResult::Frame(df) = result {
            combined = if combined.ncol() == 0 {
                df.clone()
            } else {
                combined.hcat_unique(df)
            };
        }
    }
    ViewResult::Frame(combined)
}

/// Write views to CSV files.
pub fn view_to_csv(
    es: &EsmZones,
    trans_map: &IndexMap<String, String>,
    outdir: &str,
    to_out: &[&str],
) -> Result<(), String> {
    let views = produce_views(es, trans_map, to_out)?;

    for (name, view) in &views {
        let path = if outdir.is_empty() {
            format!("{}.csv", name)
        } else {
            format!("{}/{}.csv", outdir, name)
        };
        info!("Writing view: {} to {}", name, path);

        match view {
            ViewResult::Frame(df) => {
                write_df_csv(df, &path).map_err(|e| format!("Failed to write CSV: {}", e))?;
            }
            ViewResult::Matrix(m) => {
                write_matrix_csv(m, &path).map_err(|e| format!("Failed to write CSV: {}", e))?;
            }
            ViewResult::Scalar(v) => {
                let mut f =
                    std::fs::File::create(&path).map_err(|e| format!("Failed to create file: {}", e))?;
                writeln!(f, "{}", v).map_err(|e| format!("Failed to write: {}", e))?;
            }
        }
    }

    info!("Views written successfully.");
    Ok(())
}

fn write_df_csv(df: &DataFrame, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;
    let names = df.names();
    wtr.write_record(&names)?;

    for i in 0..df.nrow() {
        let row: Vec<String> = names
            .iter()
            .map(|name| df.column(name).unwrap()[i].to_string())
            .collect();
        wtr.write_record(&row)?;
    }
    wtr.flush()?;
    Ok(())
}

fn write_matrix_csv(matrix: &[Vec<f64>], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut wtr = csv::Writer::from_path(path)?;
    for row in matrix {
        let strings: Vec<String> = row.iter().map(|v| v.to_string()).collect();
        wtr.write_record(&strings)?;
    }
    wtr.flush()?;
    Ok(())
}
