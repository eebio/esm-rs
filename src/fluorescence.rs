use crate::dataframe::DataFrame;
use crate::growth_rate::{self, GrowthRateMethod, Recalibrate};
use crate::utils::index_between_vals;
use log::warn;

/// Trait for fluorescence methods.
pub trait FluorescenceMethod {
    fn compute(
        &self,
        data_fl: &DataFrame,
        time_fl: &DataFrame,
        data_od: &DataFrame,
        time_od: &DataFrame,
    ) -> DataFrame;
}

/// Fluorescence per cell at a specific time point.
pub struct RatioAtTime {
    pub time: f64, // in minutes
}

impl FluorescenceMethod for RatioAtTime {
    fn compute(
        &self,
        data_fl: &DataFrame,
        time_fl: &DataFrame,
        data_od: &DataFrame,
        time_od: &DataFrame,
    ) -> DataFrame {
        let mut result = DataFrame::new();

        let fl_times = time_fl.column_f64(time_fl.names()[0]).unwrap();
        let od_times = time_od.column_f64(time_od.names()[0]).unwrap();

        for col_name in data_od.names() {
            if self.time.is_nan() {
                warn!("Invalid time of {} specified for fluorescence per cell. Returning NaN for column {}.", self.time, col_name);
                result.insert_f64_column(col_name, vec![f64::NAN]);
                continue;
            }

            let fl_val = at_time_scalar(
                &data_fl.column_f64(col_name).unwrap_or_default(),
                &fl_times,
                self.time,
            );
            let od_val = at_time_scalar(
                &data_od.column_f64(col_name).unwrap_or_default(),
                &od_times,
                self.time,
            );

            match (fl_val, od_val) {
                (Some(fl), Some(od)) => {
                    result.insert_f64_column(col_name, vec![fl / od]);
                }
                _ => {
                    warn!("Invalid time of {} specified for fluorescence per cell. Returning NaN for column {}.", self.time, col_name);
                    result.insert_f64_column(col_name, vec![f64::NAN]);
                }
            }
        }

        result
    }
}

/// Fluorescence per cell at the time of maximum growth rate.
pub struct RatioAtMaxGrowth {
    pub method: Box<dyn GrowthRateMethod>,
    pub recalibrate: Recalibrate,
    pub offset: f64,
}

impl FluorescenceMethod for RatioAtMaxGrowth {
    fn compute(
        &self,
        data_fl: &DataFrame,
        time_fl: &DataFrame,
        data_od: &DataFrame,
        time_od: &DataFrame,
    ) -> DataFrame {
        let time_max = growth_rate::time_to_max_growth(
            data_od,
            time_od,
            self.method.as_ref(),
            self.recalibrate,
            self.offset,
        );

        let mut result = DataFrame::new();
        for col_name in data_od.names() {
            let t = time_max.column_f64(col_name).unwrap()[0];
            let ratio = RatioAtTime { time: t };
            let col_result = ratio.compute(data_fl, time_fl, data_od, time_od);
            if let Some(vals) = col_result.column_f64(col_name) {
                result.insert_f64_column(col_name, vals);
            }
        }
        result
    }
}

/// Helper: get scalar value at a time point.
fn at_time_scalar(data: &[f64], times: &[f64], time_min: f64) -> Option<f64> {
    let time_ms = time_min * 60000.0;
    let (_, last) = index_between_vals(times, 0.0, time_ms);
    last.map(|idx| data[idx])
}

/// Convenience function to compute fluorescence.
pub fn fluorescence(
    data_fl: &DataFrame,
    time_fl: &DataFrame,
    data_od: &DataFrame,
    time_od: &DataFrame,
    method: &dyn FluorescenceMethod,
) -> DataFrame {
    method.compute(data_fl, time_fl, data_od, time_od)
}
