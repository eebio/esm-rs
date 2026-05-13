//! CSV export and PDF plotting for the `summarise` CLI command.
//!
//! Mirrors the behaviour of `src/summarise.jl`:
//! - `--csv` writes per-channel CSV files for plate readers, and a single CSV for FCS.
//! - `--plot` writes a multi-page PDF (one per input file) with plots of the data.

use std::error::Error;

use indexmap::IndexMap;
use plotters::prelude::*;
use printpdf::{Mm, Op, PdfDocument, PdfPage, PdfSaveOptions, RawImage, XObjectTransform};

use crate::dataframe::{DataFrame, Value};
use crate::esm_files::get_sample_values;
use crate::esm_types::EsmZones;
use crate::fcs::FcsFile;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const PLOT_DPI: f32 = 150.0;

// ---------------------------------------------------------------------------
// CSV export
// ---------------------------------------------------------------------------

pub fn export_plate_reader_csv(
    file: &str,
    data: &IndexMap<String, DataFrame>,
) -> Result<()> {
    for (channel, df) in data {
        let out = format!("{}_{}.csv", file, channel);
        write_dataframe_csv(df, &out)?;
        log::info!("Wrote {}", out);
    }
    Ok(())
}

pub fn export_fcs_csv(file: &str, fcs: &FcsFile) -> Result<()> {
    let out = format!("{}.csv", file);
    let mut wtr = csv::Writer::from_path(&out)?;
    let cols: Vec<&str> = fcs.data.keys().map(|s| s.as_str()).collect();
    wtr.write_record(&cols)?;
    let nrows = fcs.data.values().next().map_or(0, |v| v.len());
    for i in 0..nrows {
        let row: Vec<String> =
            fcs.data.values().map(|v| v[i].to_string()).collect();
        wtr.write_record(&row)?;
    }
    wtr.flush()?;
    log::info!("Wrote {}", out);
    Ok(())
}

fn write_dataframe_csv(df: &DataFrame, path: &str) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    let names = df.names();
    wtr.write_record(&names)?;
    for i in 0..df.nrow() {
        let row: Vec<String> = df
            .columns
            .values()
            .map(|col| match col.get(i) {
                Some(Value::Missing) => String::new(),
                Some(v) => v.to_string(),
                None => String::new(),
            })
            .collect();
        wtr.write_record(&row)?;
    }
    wtr.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// PDF plotting
// ---------------------------------------------------------------------------

/// A rendered plot: PNG bytes and pixel dimensions.
struct Page {
    width: u32,
    height: u32,
    png: Vec<u8>,
}

/// Render an RGB drawing into a PNG byte buffer.
fn render_png<F>(width: u32, height: u32, draw: F) -> Result<Page>
where
    F: FnOnce(
        &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    ) -> std::result::Result<(), Box<dyn Error>>,
{
    let mut buf = vec![0u8; (width * height * 3) as usize];
    {
        let backend = BitMapBackend::with_buffer(&mut buf, (width, height));
        let area = backend.into_drawing_area();
        area.fill(&WHITE)?;
        draw(&area)?;
        area.present()?;
    }
    let img = image::RgbImage::from_raw(width, height, buf)
        .ok_or("png buffer size mismatch")?;
    let mut png_bytes: Vec<u8> = Vec::new();
    image::DynamicImage::ImageRgb8(img).write_to(
        &mut std::io::Cursor::new(&mut png_bytes),
        image::ImageFormat::Png,
    )?;
    Ok(Page {
        width,
        height,
        png: png_bytes,
    })
}

/// Combine rendered pages into a single PDF written to `output`.
fn pages_to_pdf(pages: Vec<Page>, output: &str, title: &str) -> Result<()> {
    if pages.is_empty() {
        log::info!("No plots to write for {}", output);
        return Ok(());
    }
    let mut doc = PdfDocument::new(title);
    let mut pdf_pages = Vec::new();
    for page in pages {
        let mut warnings = Vec::new();
        let raw = RawImage::decode_from_bytes(&page.png, &mut warnings)
            .map_err(|e| format!("decode PNG: {:?}", e))?;
        let id = doc.add_image(&raw);
        let mm_w = page.width as f32 / PLOT_DPI * 25.4;
        let mm_h = page.height as f32 / PLOT_DPI * 25.4;
        let ops = vec![Op::UseXobject {
            id,
            transform: XObjectTransform {
                dpi: Some(PLOT_DPI),
                ..Default::default()
            },
        }];
        pdf_pages.push(PdfPage::new(Mm(mm_w), Mm(mm_h), ops));
    }
    let bytes = doc
        .with_pages(pdf_pages)
        .save(&PdfSaveOptions::default(), &mut Vec::new());
    std::fs::write(output, bytes)?;
    log::info!("Wrote {}", output);
    Ok(())
}

// ---- ESM plotting ----

pub fn plot_esm(file: &str, es: &EsmZones) -> Result<()> {
    let type_col = es
        .samples
        .column("type")
        .ok_or("samples missing 'type' column")?;
    let name_col = es
        .samples
        .column("name")
        .ok_or("samples missing 'name' column")?;

    let mut pages: Vec<Page> = Vec::new();
    for i in 0..es.samples.nrow() {
        let is_timeseries = matches!(type_col.get(i), Some(Value::Str(s)) if s == "timeseries");
        if !is_timeseries {
            continue;
        }
        let name = name_col
            .get(i)
            .map(|v| v.to_string())
            .unwrap_or_else(|| format!("sample_{}", i));
        let values = get_sample_values(es, i);
        if values.is_empty() {
            continue;
        }
        let page = render_timeseries_page(&name, &values)?;
        pages.push(page);
    }
    pages_to_pdf(pages, &format!("{}.pdf", file), "ESM summary")
}

fn render_timeseries_page(name: &str, values: &[f64]) -> Result<Page> {
    let (y_min, y_max) = finite_range(values, 0.0..=1.0);
    let title = format!("Timeseries for {}", name);
    render_png(1200, 800, move |area| {
        let mut chart = ChartBuilder::on(area)
            .caption(&title, ("sans-serif", 28))
            .margin(20)
            .x_label_area_size(50)
            .y_label_area_size(70)
            .build_cartesian_2d(0f64..(values.len().max(1) as f64), y_min..y_max)?;
        chart
            .configure_mesh()
            .x_desc("Index")
            .y_desc("Value")
            .draw()?;
        chart.draw_series(LineSeries::new(
            values.iter().enumerate().map(|(i, v)| (i as f64, *v)),
            &BLACK,
        ))?;
        Ok(())
    })
}

// ---- Plate reader plotting ----

pub fn plot_plate_reader(
    file: &str,
    data: &IndexMap<String, DataFrame>,
) -> Result<()> {
    let mut pages: Vec<Page> = Vec::new();
    for (channel, df) in data {
        let well_cols = well_columns(df);
        if well_cols.is_empty() {
            continue;
        }
        let time_minutes: Vec<f64> = df
            .columns
            .values()
            .next()
            .map(|c| c.iter().filter_map(|v| v.as_f64()).map(|v| v / 60000.0).collect())
            .unwrap_or_default();

        let mut all_values: Vec<f64> = Vec::new();
        for (_, vals) in &well_cols {
            all_values.extend(vals.iter().copied().filter(|v| v.is_finite()));
        }
        let data_max = all_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let max_lin = data_max.max(1.0) * 1.05;
        let mut max_log = 10f64.powf((data_max * 1.05).max(1.0).log10().ceil());
        let min_log = 10f64.powf(
            all_values
                .iter()
                .cloned()
                .filter(|v| *v > 0.0)
                .fold(f64::INFINITY, f64::min)
                .max(1e-9)
                .log10()
                .floor(),
        );
        // plotters' LogCoord::key_points generation can spin on single-decade
        // ranges; widen the range to at least two decades so it terminates.
        if !(max_log > min_log * 10.0) {
            max_log = min_log * 100.0;
        }

        for (scale_name, log_scale, y_min, y_max) in [
            ("Linear", false, 0.0_f64, max_lin),
            ("Log10", true, min_log, max_log),
        ] {
            pages.push(render_multipanel(
                channel,
                scale_name,
                log_scale,
                &time_minutes,
                &well_cols,
                y_min,
                y_max,
            )?);
            pages.push(render_overlay(
                channel,
                scale_name,
                log_scale,
                &time_minutes,
                &well_cols,
                y_min,
                y_max,
            )?);
        }
    }
    pages_to_pdf(pages, &format!("{}.pdf", file), "Plate reader summary")
}

/// Return (well_name, finite values) pairs, skipping the leading time and
/// temperature columns plus any anonymous trailing column produced by parsers
/// that read a trailing tab/comma as an empty well.
fn well_columns(df: &DataFrame) -> Vec<(String, Vec<f64>)> {
    df.columns
        .iter()
        .enumerate()
        .filter_map(|(idx, (name, col))| {
            if idx < 2 || name.trim().is_empty() {
                return None;
            }
            let vals: Vec<f64> = col.iter().filter_map(|v| v.as_f64()).collect();
            if vals.iter().all(|v| !v.is_finite()) {
                return None;
            }
            Some((name.clone(), vals))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn render_multipanel(
    channel: &str,
    scale_name: &str,
    log_scale: bool,
    time: &[f64],
    wells: &[(String, Vec<f64>)],
    y_min: f64,
    y_max: f64,
) -> Result<Page> {
    let ncols = wells.len().min(12).max(1);
    let nrows = wells.len().div_ceil(12).max(1);
    let width = (150 * ncols) as u32;
    let height = (150 * nrows) as u32 + 80;
    let title =
        format!("Multipanel — Channel {} ({} scale)", channel, scale_name);
    let x_max = time.last().copied().unwrap_or(1.0).max(1.0);
    render_png(width, height, move |area| {
        let (header, body) = area.split_vertically(60);
        header.titled(&title, ("sans-serif", 22).into_font())?;
        let sub_areas = body.split_evenly((nrows, ncols));
        for (idx, sub) in sub_areas.iter().enumerate() {
            if idx >= wells.len() {
                continue;
            }
            let (name, vals) = &wells[idx];
            let mut cb = ChartBuilder::on(sub);
            cb.margin(2)
                .x_label_area_size(20)
                .y_label_area_size(28)
                .caption(name, ("sans-serif", 12));
            draw_well_chart(&mut cb, log_scale, time, vals, 0.0..x_max, y_min..y_max)?;
        }
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn render_overlay(
    channel: &str,
    scale_name: &str,
    log_scale: bool,
    time: &[f64],
    wells: &[(String, Vec<f64>)],
    y_min: f64,
    y_max: f64,
) -> Result<Page> {
    let title = format!("Overlay — Channel {} ({} scale)", channel, scale_name);
    let x_max = time.last().copied().unwrap_or(1.0).max(1.0);
    render_png(1200, 800, move |area| {
        let mut cb = ChartBuilder::on(area);
        cb.caption(&title, ("sans-serif", 24))
            .margin(20)
            .x_label_area_size(45)
            .y_label_area_size(70);
        if log_scale {
            let mut chart = cb.build_cartesian_2d(
                0f64..x_max,
                (y_min..y_max).log_scale(),
            )?;
            chart
                .configure_mesh()
                .x_desc("Time (min)")
                .y_desc("Value")
                .draw()?;
            for (_, vals) in wells {
                let pts: Vec<(f64, f64)> = time
                    .iter()
                    .zip(vals.iter())
                    .filter(|(_, v)| v.is_finite() && **v > 0.0)
                    .map(|(t, v)| (*t, *v))
                    .collect();
                chart.draw_series(LineSeries::new(
                    pts,
                    BLACK.mix(0.4).stroke_width(2),
                ))?;
            }
        } else {
            let mut chart = cb.build_cartesian_2d(0f64..x_max, y_min..y_max)?;
            chart
                .configure_mesh()
                .x_desc("Time (min)")
                .y_desc("Value")
                .draw()?;
            for (_, vals) in wells {
                let pts: Vec<(f64, f64)> = time
                    .iter()
                    .zip(vals.iter())
                    .filter(|(_, v)| v.is_finite())
                    .map(|(t, v)| (*t, *v))
                    .collect();
                chart.draw_series(LineSeries::new(
                    pts,
                    BLACK.mix(0.4).stroke_width(2),
                ))?;
            }
        }
        Ok(())
    })
}

fn draw_well_chart(
    cb: &mut ChartBuilder<BitMapBackend>,
    log_scale: bool,
    time: &[f64],
    vals: &[f64],
    x_range: std::ops::Range<f64>,
    y_range: std::ops::Range<f64>,
) -> std::result::Result<(), Box<dyn Error>> {
    if log_scale {
        let mut chart = cb.build_cartesian_2d(x_range, y_range.log_scale())?;
        chart
            .configure_mesh()
            .disable_mesh()
            .label_style(("sans-serif", 9))
            .draw()?;
        let pts: Vec<(f64, f64)> = time
            .iter()
            .zip(vals.iter())
            .filter(|(_, v)| v.is_finite() && **v > 0.0)
            .map(|(t, v)| (*t, *v))
            .collect();
        chart.draw_series(LineSeries::new(pts, BLACK.stroke_width(2)))?;
    } else {
        let mut chart = cb.build_cartesian_2d(x_range, y_range)?;
        chart
            .configure_mesh()
            .disable_mesh()
            .label_style(("sans-serif", 9))
            .draw()?;
        let pts: Vec<(f64, f64)> = time
            .iter()
            .zip(vals.iter())
            .filter(|(_, v)| v.is_finite())
            .map(|(t, v)| (*t, *v))
            .collect();
        chart.draw_series(LineSeries::new(pts, BLACK.stroke_width(2)))?;
    }
    Ok(())
}

// ---- FCS plotting ----

pub fn plot_fcs(file: &str, fcs: &FcsFile) -> Result<()> {
    let time_key = fcs
        .data
        .keys()
        .find(|k| k.eq_ignore_ascii_case("Time"))
        .cloned();
    let channels: Vec<String> = fcs
        .data
        .keys()
        .filter(|k| Some(*k) != time_key.as_ref())
        .cloned()
        .collect();

    let mut pages: Vec<Page> = Vec::new();
    for c in &channels {
        let vals = &fcs.data[c];
        pages.push(render_histogram(c, vals)?);
        if let Some(tk) = &time_key {
            let t = &fcs.data[tk];
            pages.push(render_time_scatter(c, t, vals)?);
        }
    }
    for i in 0..channels.len() {
        for j in (i + 1)..channels.len() {
            let c1 = &channels[i];
            let c2 = &channels[j];
            pages.push(render_2d_histogram(
                c1,
                &fcs.data[c1],
                c2,
                &fcs.data[c2],
            )?);
        }
    }
    pages_to_pdf(pages, &format!("{}.pdf", file), "FCS summary")
}

fn render_histogram(name: &str, values: &[f64]) -> Result<Page> {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return render_png(800, 600, |_| Ok(()));
    }
    let v_min = finite.iter().cloned().fold(f64::INFINITY, f64::min);
    let v_max = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let nbins = 50usize;
    let width = ((v_max - v_min) / nbins as f64).max(f64::EPSILON);
    let mut bins = vec![0u64; nbins];
    for v in &finite {
        let idx = (((*v - v_min) / width) as usize).min(nbins - 1);
        bins[idx] += 1;
    }
    let count_max = *bins.iter().max().unwrap_or(&1) as f64;
    let title = format!("{} (histogram)", name);
    let name = name.to_string();
    render_png(1000, 700, move |area| {
        let mut chart = ChartBuilder::on(area)
            .caption(&title, ("sans-serif", 24))
            .margin(20)
            .x_label_area_size(45)
            .y_label_area_size(60)
            .build_cartesian_2d(v_min..v_max, 0f64..(count_max * 1.05))?;
        chart
            .configure_mesh()
            .x_desc(&name)
            .y_desc("Count")
            .draw()?;
        chart.draw_series(bins.iter().enumerate().map(|(i, &n)| {
            let x0 = v_min + i as f64 * width;
            let x1 = x0 + width;
            Rectangle::new(
                [(x0, 0.0), (x1, n as f64)],
                RGBColor(96, 125, 139).filled(),
            )
        }))?;
        Ok(())
    })
}

fn render_time_scatter(name: &str, time: &[f64], values: &[f64]) -> Result<Page> {
    let pts: Vec<(f64, f64)> = time
        .iter()
        .zip(values.iter())
        .filter(|(t, v)| t.is_finite() && v.is_finite())
        .map(|(t, v)| (*t, *v))
        .collect();
    if pts.is_empty() {
        return render_png(800, 600, |_| Ok(()));
    }
    let t_min = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let t_max = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let v_min = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let v_max = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let title = format!("{} vs Time", name);
    let name = name.to_string();
    render_png(1000, 700, move |area| {
        let mut chart = ChartBuilder::on(area)
            .caption(&title, ("sans-serif", 24))
            .margin(20)
            .x_label_area_size(45)
            .y_label_area_size(60)
            .build_cartesian_2d(t_min..t_max, v_min..v_max)?;
        chart
            .configure_mesh()
            .x_desc("Time")
            .y_desc(&name)
            .draw()?;
        chart.draw_series(
            pts.iter()
                .map(|(t, v)| Circle::new((*t, *v), 2, BLACK.mix(0.4).filled())),
        )?;
        Ok(())
    })
}

fn render_2d_histogram(
    name_x: &str,
    xs: &[f64],
    name_y: &str,
    ys: &[f64],
) -> Result<Page> {
    let pts: Vec<(f64, f64)> = xs
        .iter()
        .zip(ys.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (*x, *y))
        .collect();
    if pts.is_empty() {
        return render_png(800, 600, |_| Ok(()));
    }
    let x_min = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let x_max = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let y_min = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let y_max = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let nbins = 64usize;
    let dx = ((x_max - x_min) / nbins as f64).max(f64::EPSILON);
    let dy = ((y_max - y_min) / nbins as f64).max(f64::EPSILON);
    let mut grid = vec![0u32; nbins * nbins];
    for (x, y) in &pts {
        let i = (((*x - x_min) / dx) as usize).min(nbins - 1);
        let j = (((*y - y_min) / dy) as usize).min(nbins - 1);
        grid[j * nbins + i] += 1;
    }
    let max_count = *grid.iter().max().unwrap_or(&1) as f64;
    let title = format!("{} vs {} (2D histogram)", name_x, name_y);
    let name_x = name_x.to_string();
    let name_y = name_y.to_string();
    render_png(900, 800, move |area| {
        let mut chart = ChartBuilder::on(area)
            .caption(&title, ("sans-serif", 22))
            .margin(20)
            .x_label_area_size(45)
            .y_label_area_size(60)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
        chart
            .configure_mesh()
            .x_desc(&name_x)
            .y_desc(&name_y)
            .draw()?;
        let cells = (0..nbins).flat_map(|j| (0..nbins).map(move |i| (i, j)));
        chart.draw_series(cells.map(|(i, j)| {
            let count = grid[j * nbins + i] as f64;
            let intensity = if max_count > 0.0 {
                (count / max_count).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let shade = (255.0 * (1.0 - intensity)) as u8;
            let x0 = x_min + i as f64 * dx;
            let x1 = x0 + dx;
            let y0 = y_min + j as f64 * dy;
            let y1 = y0 + dy;
            Rectangle::new(
                [(x0, y0), (x1, y1)],
                RGBColor(shade, shade, 255).filled(),
            )
        }))?;
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn finite_range(
    values: &[f64],
    default: std::ops::RangeInclusive<f64>,
) -> (f64, f64) {
    let finite: Vec<f64> =
        values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (*default.start(), *default.end());
    }
    let mn = finite.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (mx - mn).abs() < f64::EPSILON {
        (mn - 1.0, mx + 1.0)
    } else {
        let pad = (mx - mn) * 0.05;
        (mn - pad, mx + pad)
    }
}
