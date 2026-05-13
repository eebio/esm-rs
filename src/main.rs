//! ESM CLI - Standardised data format and supporting tools for reproducible data processing
//! in engineering biology.

use clap::{Parser, Subcommand};
use esm::{esm_files, read_data, views};
use log::info;
use std::process;

#[derive(Parser)]
#[command(name = "esm")]
#[command(about = "Standardised data format and tools for engineering biology data processing")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Translate a completed .xlsx template file to a .esm file
    Translate {
        /// The completed .xlsx template file to read
        input: String,
        /// The filepath/destination for the .esm file
        output: String,
    },
    /// Produce and save views from a .esm file
    Views {
        /// The .esm file to read
        esm_file: String,
        /// Specific view to produce (all views if not specified)
        #[arg(short, long)]
        view: Option<String>,
        /// Directory to save output(s) to
        #[arg(short, long, default_value = ".")]
        output_dir: String,
    },
    /// Summarise a data file (.esm, plate reader, .fcs, etc.)
    Summarise {
        /// The data file to summarise
        file: String,
        /// Data file type: auto, esm, spectramax, biotek, tecan, bmg, generic, fcs
        #[arg(short, long, default_value = "auto")]
        r#type: String,
        /// Produce plots of the data (PDF). Not available for --type=esm only insofar as it has no time axis; still emits per-sample timeseries.
        #[arg(short, long)]
        plot: bool,
        /// Save the data as CSV files. Not available for --type=esm.
        #[arg(short, long)]
        csv: bool,
    },
    /// Produce a template xlsx file for data entry into the ESM
    Template {
        /// Path to create the template at
        #[arg(short, long, default_value = "ESM.xlsx")]
        output_path: String,
    },
}

fn main() {
    env_logger::init();
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Translate { input, output } => cmd_translate(&input, &output),
        Commands::Views {
            esm_file,
            view,
            output_dir,
        } => cmd_views(&esm_file, view.as_deref(), &output_dir),
        Commands::Summarise {
            file,
            r#type,
            plot,
            csv,
        } => cmd_summarise(&file, &r#type, plot, csv),
        Commands::Template { output_path } => cmd_template(&output_path),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

const TEMPLATE_XLSX: &[u8] = include_bytes!("ESM.xlsx");

fn cmd_template(output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::write(output_path, TEMPLATE_XLSX)?;
    println!("Template written to {}", output_path);
    Ok(())
}

fn cmd_translate(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("Translating {} to {}", input, output);
    let data = read_data::read_data(input)?;
    read_data::write_esm_data(&data, output)?;
    println!("ESM file written to {}", output);
    Ok(())
}

fn cmd_views(
    esm_file: &str,
    view: Option<&str>,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let es = esm_files::read_esm(esm_file)?;
    info!("Producing views.");

    let view_names: Vec<&str> = match view {
        Some(v) => vec![v],
        None => Vec::new(),
    };

    // Build transformation map
    let trans_map: indexmap::IndexMap<String, String> = es
        .transformations
        .iter()
        .filter_map(|(k, v)| v.get("equation").map(|eq: &String| (k.clone(), eq.clone())))
        .collect();

    let view_list: Vec<&str> = if view_names.is_empty() {
        es.views.keys().map(|s| s.as_str()).collect()
    } else {
        view_names
    };

    views::view_to_csv(&es, &trans_map, output_dir, &view_list)?;
    println!("Views saved to {}", output_dir);
    Ok(())
}

fn cmd_summarise(
    file: &str,
    file_type: &str,
    plot: bool,
    csv: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let actual_type = if file_type == "auto" {
        // Infer from extension
        if file.ends_with(".esm") {
            "esm"
        } else if file.ends_with(".fcs") {
            "fcs"
        } else if std::path::Path::new(file).is_dir() {
            "generic"
        } else {
            return Err(format!(
                "Cannot infer file type from '{}'. Use --type to specify.",
                file
            )
            .into());
        }
    } else {
        file_type
    };

    match actual_type {
        "esm" => {
            if csv {
                return Err("--csv is not supported for esm files".into());
            }
            summarise_esm(file, plot)
        }
        "fcs" => summarise_fcs(file, plot, csv),
        "spectramax" | "biotek" | "tecan" | "bmg" | "generic" => {
            summarise_plate_reader(file, actual_type, plot, csv)
        }
        _ => Err(format!("Unsupported file type: {}", actual_type).into()),
    }
}

fn summarise_esm(file: &str, plot: bool) -> Result<(), Box<dyn std::error::Error>> {
    let es = esm_files::read_esm(file)?;

    let sample_names = esm_files::sample_names(&es);
    let group_names = esm_files::group_names(&es);

    // Count timeseries vs population samples
    let type_col = es.samples.column("type").unwrap();
    let mut timeseries_count = 0;
    let mut population_count = 0;
    let mut channels: std::collections::HashSet<String> = std::collections::HashSet::new();

    let channel_col = es.samples.column("channel").unwrap();
    for (t, ch) in type_col.iter().zip(channel_col.iter()) {
        if let esm::Value::Str(s) = t {
            if s == "timeseries" {
                timeseries_count += 1;
            } else {
                population_count += 1;
            }
        }
        if let esm::Value::Str(c) = ch {
            channels.insert(c.clone());
        }
    }

    println!("ESM Summary: {}", file);
    println!(
        "  Samples: {} ({} timeseries, {} population)",
        sample_names.len(),
        timeseries_count,
        population_count
    );
    println!("  Channels: {:?}", channels);
    println!("  Groups: {:?}", group_names);
    println!(
        "  Transformations: {:?}",
        es.transformations.keys().collect::<Vec<_>>()
    );
    println!("  Views: {:?}", es.views.keys().collect::<Vec<_>>());

    if plot {
        esm::summarise::plot_esm(file, &es)?;
    }

    Ok(())
}

fn summarise_fcs(file: &str, plot: bool, csv: bool) -> Result<(), Box<dyn std::error::Error>> {
    let fcs = esm::fcs::read_fcs(file)?;

    let n_events = fcs.data.values().next().map(|v| v.len()).unwrap_or(0);

    println!("FCS Summary: {}", file);
    println!("  Events: {}", n_events);
    println!("  Channels: {}", fcs.data.len());
    for (name, values) in &fcs.data {
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("    {}: range [{:.2}, {:.2}]", name, min, max);
    }

    if csv {
        esm::summarise::export_fcs_csv(file, &fcs)?;
    }
    if plot {
        esm::summarise::plot_fcs(file, &fcs)?;
    }

    Ok(())
}

fn summarise_plate_reader(
    file: &str,
    ptype: &str,
    plot: bool,
    csv: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use esm::plate_readers::PlateReaderParser;

    let (data, _metadata) = match ptype {
        "spectramax" => esm::plate_readers::spectramax::SpectraMax.read(file, None)?,
        "biotek" => esm::plate_readers::biotek::BioTek.read(file, None)?,
        "tecan" => esm::plate_readers::tecan::Tecan.read(file, None)?,
        "bmg" => esm::plate_readers::bmg::Bmg.read(file, None)?,
        "generic" => esm::plate_readers::generic::GenericTabular.read(file, None)?,
        _ => return Err(format!("Unknown plate reader type: {}", ptype).into()),
    };

    println!("Plate Reader Summary: {} ({})", file, ptype);
    println!("  Channels: {}", data.len());
    for (channel, df) in &data {
        let n_wells = df.ncol() - 2; // Subtract time and temperature
        println!(
            "    {}: {} timepoints, {} wells",
            channel,
            df.nrow(),
            n_wells
        );
    }

    if csv {
        esm::summarise::export_plate_reader_csv(file, &data)?;
    }
    if plot {
        esm::summarise::plot_plate_reader(file, &data)?;
    }

    Ok(())
}
