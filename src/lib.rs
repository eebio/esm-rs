//! ESM - Standardised data format and supporting tools for reproducible data processing
//! in engineering biology.
//!
//! This is a Rust port of the Julia ESM package, providing high-performance data
//! processing for plate reader and flow cytometry data.

pub mod calibrate;
pub mod dataframe;
pub mod esm_files;
pub mod esm_types;
pub mod expression;
pub mod fcs;
pub mod fit_ellipse;
pub mod flow;
pub mod fluorescence;
pub mod growth_rate;
pub mod plate_readers;
pub mod read_data;
pub mod summarise;
pub mod utils;
pub mod views;

// Re-export commonly used types
pub use dataframe::{DataFrame, Value};
pub use esm_files::{read_esm, read_esm_from_str};
pub use esm_types::EsmZones;
pub use utils::{expand_group, expand_groups, format_channel};
