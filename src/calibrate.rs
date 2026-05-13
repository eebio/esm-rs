use crate::dataframe::DataFrame;

/// Trait for calibration methods.
pub trait CalibrationMethod {
    fn calibrate(&self, data: &DataFrame, time_col: &DataFrame) -> DataFrame;
}

/// Apply calibration with an offset added afterwards.
pub fn calibrate_with_offset(
    data: &DataFrame,
    time_col: &DataFrame,
    method: &dyn CalibrationMethod,
    offset: f64,
) -> DataFrame {
    let calibrated = method.calibrate(data, time_col);
    calibrated.add_scalar(offset)
}

/// Calibrate using the raw method (no offset).
pub fn calibrate(
    data: &DataFrame,
    time_col: &DataFrame,
    method: &dyn CalibrationMethod,
) -> DataFrame {
    method.calibrate(data, time_col)
}

/// Timeseries blank subtraction: subtract the mean of blank wells at each timepoint.
pub struct TimeseriesBlank {
    pub blanks: DataFrame,
    pub time_col: Option<DataFrame>,
}

impl CalibrationMethod for TimeseriesBlank {
    fn calibrate(&self, data: &DataFrame, time_col: &DataFrame) -> DataFrame {
        let blanks = &self.blanks;
        let averaged_blanks = blanks.colmean();

        let blank_time_col = self.time_col.as_ref().unwrap_or(time_col);

        // Check if interpolation is needed
        let data_times = time_col.column_f64(time_col.names()[0]).unwrap();
        let blank_times = blank_time_col
            .column_f64(blank_time_col.names()[0])
            .unwrap();

        let interp_blanks = if data_times != blank_times {
            linear_interpolate(&blank_times, &averaged_blanks, &data_times)
        } else {
            averaged_blanks
        };

        data.sub_vec(&interp_blanks)
    }
}

/// Smoothed timeseries blank: fit a linear model to blanks and subtract.
pub struct SmoothedTimeseriesBlank {
    pub blanks: DataFrame,
    pub time_col: Option<DataFrame>,
}

impl CalibrationMethod for SmoothedTimeseriesBlank {
    fn calibrate(&self, data: &DataFrame, time_col: &DataFrame) -> DataFrame {
        let blanks = &self.blanks;
        let averaged_blanks = blanks.colmean();

        let blank_time_col = self.time_col.as_ref().unwrap_or(time_col);
        let blank_times = blank_time_col
            .column_f64(blank_time_col.names()[0])
            .unwrap();

        // Fit linear model: od = a + b * time
        let (intercept, slope) = linear_regression(&blank_times, &averaged_blanks);

        // Predict at data time points
        let data_times = time_col.column_f64(time_col.names()[0]).unwrap();
        let smoothed: Vec<f64> = data_times
            .iter()
            .map(|&t| intercept + slope * t)
            .collect();

        data.sub_vec(&smoothed)
    }
}

/// Mean blank subtraction: subtract the overall mean of blank wells.
pub struct MeanBlank {
    pub blanks: DataFrame,
}

impl CalibrationMethod for MeanBlank {
    fn calibrate(&self, data: &DataFrame, _time_col: &DataFrame) -> DataFrame {
        let averaged = self.blanks.colmean();
        let mean = averaged.iter().sum::<f64>() / averaged.len() as f64;
        data.sub_scalar(mean)
    }
}

/// Minimum blank subtraction: subtract the minimum value across all blank wells.
pub struct MinBlank {
    pub blanks: DataFrame,
}

impl CalibrationMethod for MinBlank {
    fn calibrate(&self, data: &DataFrame, _time_col: &DataFrame) -> DataFrame {
        let min_val = self
            .blanks
            .columns
            .values()
            .flat_map(|col| col.iter().filter_map(|v| v.as_f64()))
            .fold(f64::INFINITY, f64::min);
        data.sub_scalar(min_val)
    }
}

/// Minimum data subtraction: subtract per-column minimum from data.
pub struct MinData;

impl CalibrationMethod for MinData {
    fn calibrate(&self, data: &DataFrame, _time_col: &DataFrame) -> DataFrame {
        let mut result = data.clone();
        for (name, col) in &data.columns {
            let min_val = col
                .iter()
                .filter_map(|v| v.as_f64())
                .fold(f64::INFINITY, f64::min);
            if min_val.is_finite() {
                let new_col: Vec<crate::dataframe::Value> = col
                    .iter()
                    .map(|v| match v.as_f64() {
                        Some(f) => crate::dataframe::Value::Float(f - min_val),
                        None => v.clone(),
                    })
                    .collect();
                result.columns.insert(name.clone(), new_col);
            }
        }
        result
    }
}

/// Start data subtraction: subtract the first row value from each column.
pub struct StartData;

impl CalibrationMethod for StartData {
    fn calibrate(&self, data: &DataFrame, _time_col: &DataFrame) -> DataFrame {
        let mut result = data.clone();
        for (name, col) in &data.columns {
            if let Some(first_val) = col.first().and_then(|v| v.as_f64()) {
                let new_col: Vec<crate::dataframe::Value> = col
                    .iter()
                    .map(|v| match v.as_f64() {
                        Some(f) => crate::dataframe::Value::Float(f - first_val),
                        None => v.clone(),
                    })
                    .collect();
                result.columns.insert(name.clone(), new_col);
            }
        }
        result
    }
}

/// Simple linear interpolation.
fn linear_interpolate(x_known: &[f64], y_known: &[f64], x_new: &[f64]) -> Vec<f64> {
    x_new
        .iter()
        .map(|&xi| {
            if xi <= x_known[0] {
                return y_known[0]; // Constant extrapolation
            }
            if xi >= *x_known.last().unwrap() {
                return *y_known.last().unwrap(); // Constant extrapolation
            }
            // Find the two surrounding known points
            let mut j = 0;
            while j < x_known.len() - 1 && x_known[j + 1] < xi {
                j += 1;
            }
            let t = (xi - x_known[j]) / (x_known[j + 1] - x_known[j]);
            y_known[j] + t * (y_known[j + 1] - y_known[j])
        })
        .collect()
}

/// Simple linear regression: returns (intercept, slope).
fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(&xi, &yi)| xi * yi).sum();

    let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let intercept = (sy - slope * sx) / n;

    (intercept, slope)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::Value;

    fn make_data() -> (DataFrame, DataFrame) {
        let mut data = DataFrame::new();
        data.insert_f64_column("A", vec![0.5, 0.65, 0.79, 0.83, 0.95]);
        data.insert_f64_column("B", vec![1.11, 1.05, 1.23, 1.36, 1.44]);
        let mut time = DataFrame::new();
        time.insert_column(
            "Time",
            vec![
                Value::Float(0.0),
                Value::Float(600000.0),
                Value::Float(1200000.0),
                Value::Float(1800000.0),
                Value::Float(2400000.0),
            ],
        );
        (data, time)
    }

    fn make_blanks() -> DataFrame {
        let mut blanks = DataFrame::new();
        blanks.insert_f64_column("C", vec![0.1, 0.15, 0.2, 0.17, 0.08]);
        blanks.insert_f64_column("D", vec![0.21, 0.26, 0.22, 0.23, 0.2]);
        blanks
    }

    #[test]
    fn test_timeseries_blank() {
        let (data, time) = make_data();
        let blanks = make_blanks();
        let data_copy = data.clone();

        let method = TimeseriesBlank {
            blanks,
            time_col: None,
        };
        let result = calibrate(&data, &time, &method);

        // averaged_blanks = [(0.1+0.21)/2, (0.15+0.26)/2, (0.2+0.22)/2, (0.17+0.23)/2, (0.08+0.2)/2]
        //                 = [0.155, 0.205, 0.21, 0.2, 0.14]
        let expected_a = vec![
            0.5 - 0.155,
            0.65 - 0.205,
            0.79 - 0.21,
            0.83 - 0.2,
            0.95 - 0.14,
        ];
        let result_a = result.column_f64("A").unwrap();
        for (a, b) in result_a.iter().zip(expected_a.iter()) {
            assert!((a - b).abs() < 1e-10, "{} != {}", a, b);
        }

        // Check data not mutated
        assert_eq!(data, data_copy);
    }

    #[test]
    fn test_timeseries_blank_interpolated() {
        let (data, time) = make_data();
        let mut new_blanks = DataFrame::new();
        new_blanks.insert_f64_column("C", vec![0.12, 0.14, 0.19]);
        new_blanks.insert_f64_column("D", vec![0.22, 0.25, 0.21]);
        let mut blank_time = DataFrame::new();
        blank_time.insert_f64_column("Time", vec![300000.0, 1500000.0, 2100000.0]);

        let method = TimeseriesBlank {
            blanks: new_blanks,
            time_col: Some(blank_time),
        };
        let result = calibrate(&data, &time, &method);

        // averaged = [(0.12+0.22)/2, (0.14+0.25)/2, (0.19+0.21)/2] = [0.17, 0.195, 0.20]
        // at blank_times [300000, 1500000, 2100000]
        // interpolated to data_times [0, 600000, 1200000, 1800000, 2400000]
        let expected_a = vec![
            0.5 - 0.17,
            0.65 - (0.75 * 0.17 + 0.25 * 0.195),
            0.79 - (0.25 * 0.17 + 0.75 * 0.195),
            0.83 - (0.5 * 0.195 + 0.5 * 0.20),
            0.95 - 0.20,
        ];
        let result_a = result.column_f64("A").unwrap();
        for (a, b) in result_a.iter().zip(expected_a.iter()) {
            assert!((a - b).abs() < 1e-10, "{} != {}", a, b);
        }
    }

    #[test]
    fn test_mean_blank() {
        let (data, time) = make_data();
        let blanks = make_blanks();
        let data_copy = data.clone();

        let method = MeanBlank { blanks };
        let result = calibrate(&data, &time, &method);

        // mean of colmean = mean of [0.155, 0.205, 0.21, 0.2, 0.14] = 0.182
        let mean = 0.182;
        let result_a = result.column_f64("A").unwrap();
        for (i, &v) in result_a.iter().enumerate() {
            let expected = data.column_f64("A").unwrap()[i] - mean;
            assert!((v - expected).abs() < 1e-10);
        }
        assert_eq!(data, data_copy);
    }

    #[test]
    fn test_min_blank() {
        let (data, time) = make_data();
        let blanks = make_blanks();
        let data_copy = data.clone();

        let method = MinBlank { blanks };
        let result = calibrate(&data, &time, &method);

        let min = 0.08;
        let result_a = result.column_f64("A").unwrap();
        for (i, &v) in result_a.iter().enumerate() {
            let expected = data.column_f64("A").unwrap()[i] - min;
            assert!((v - expected).abs() < 1e-10);
        }
        assert_eq!(data, data_copy);
    }

    #[test]
    fn test_min_data() {
        let (data, time) = make_data();
        let data_copy = data.clone();

        let method = MinData;
        let result = calibrate(&data, &time, &method);

        let result_a = result.column_f64("A").unwrap();
        assert!((result_a[0] - 0.0).abs() < 1e-10);
        assert!((result_a[1] - 0.15).abs() < 1e-10);

        let result_b = result.column_f64("B").unwrap();
        assert!((result_b[0] - (1.11 - 1.05)).abs() < 1e-10);
        assert!((result_b[1] - 0.0).abs() < 1e-10);

        assert_eq!(data, data_copy);
    }

    #[test]
    fn test_start_data() {
        let (data, time) = make_data();
        let data_copy = data.clone();

        let method = StartData;
        let result = calibrate(&data, &time, &method);

        let result_a = result.column_f64("A").unwrap();
        assert!((result_a[0] - 0.0).abs() < 1e-10);
        assert!((result_a[1] - 0.15).abs() < 1e-10);

        let result_b = result.column_f64("B").unwrap();
        assert!((result_b[0] - 0.0).abs() < 1e-10);
        assert!((result_b[1] - (1.05 - 1.11)).abs() < 1e-10);

        assert_eq!(data, data_copy);
    }

    #[test]
    fn test_start_data_with_offset() {
        let (data, time) = make_data();
        let method = StartData;
        let result = calibrate_with_offset(&data, &time, &method, 0.1);

        let result_a = result.column_f64("A").unwrap();
        assert!((result_a[0] - 0.1).abs() < 1e-10); // 0.5 - 0.5 + 0.1
        assert!((result_a[1] - 0.25).abs() < 1e-10); // 0.65 - 0.5 + 0.1
    }

    #[test]
    fn test_smoothed_timeseries_blank() {
        let (data, time) = make_data();
        let blanks = make_blanks();
        let data_copy = data.clone();

        let method = SmoothedTimeseriesBlank {
            blanks,
            time_col: None,
        };
        let result = calibrate(&data, &time, &method);

        // Should be less than original
        let result_a = result.column_f64("A").unwrap();
        let data_a = data.column_f64("A").unwrap();
        for (r, d) in result_a.iter().zip(data_a.iter()) {
            assert!(r < d);
        }

        // Differences should be linear (constant diff of diffs)
        let diffs: Vec<f64> = result_a
            .iter()
            .zip(data_a.iter())
            .map(|(r, d)| r - d)
            .collect();
        let diff_of_diffs: Vec<f64> = diffs.windows(2).map(|w| w[1] - w[0]).collect();
        for d in diff_of_diffs.windows(2) {
            assert!((d[0] - d[1]).abs() < 1e-10);
        }

        assert_eq!(data, data_copy);
    }
}
