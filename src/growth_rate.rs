use crate::calibrate::{self, MinData};
use crate::dataframe::DataFrame;
use crate::utils::index_between_vals;
use log::warn;

/// Trait for growth rate methods.
pub trait GrowthRateMethod {
    fn compute(
        &self,
        df: &DataFrame,
        time_col: &DataFrame,
    ) -> GrowthResult;
}

/// Results from a growth rate computation.
#[derive(Debug, Clone)]
pub struct GrowthResult {
    pub growth_rate: f64,
    pub time_to_max_growth: f64,
    pub od_at_max_growth: f64,
    pub max_od: f64,
}

/// Recalibration mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Recalibrate {
    /// Only recalibrate if negative values are present.
    Negative,
    /// Always recalibrate.
    Always,
    /// Never recalibrate (filter out non-positive values instead).
    Never,
}

/// Calculate growth rate for each column in the DataFrame.
pub fn growth_rate(
    df: &DataFrame,
    time_col: &DataFrame,
    method: &dyn GrowthRateMethod,
    recalibrate: Recalibrate,
    offset: f64,
) -> DataFrame {
    let mut result = DataFrame::new();
    for name in df.names() {
        let single = df.select(&[name]);
        let (od, times, _) = perform_recalibration(&single, time_col, recalibrate, offset);
        if od.nrow() == 0 {
            warn!("No positive values found for column {} after filtering. Returning NaN.", name);
            result.insert_f64_column(name, vec![f64::NAN]);
            continue;
        }
        let gr = method.compute(&od, &times);
        result.insert_f64_column(name, vec![gr.growth_rate]);
    }
    result
}

/// Calculate doubling time.
pub fn doubling_time(
    df: &DataFrame,
    time_col: &DataFrame,
    method: &dyn GrowthRateMethod,
    recalibrate: Recalibrate,
    offset: f64,
) -> DataFrame {
    let gr = growth_rate(df, time_col, method, recalibrate, offset);
    let mut result = DataFrame::new();
    for name in gr.names() {
        let vals = gr.column_f64(name).unwrap();
        let dt: Vec<f64> = vals.iter().map(|&v| 2.0_f64.ln() / v).collect();
        result.insert_f64_column(name, dt);
    }
    result
}

/// Calculate maximum OD.
pub fn max_od(
    df: &DataFrame,
    time_col: &DataFrame,
    method: &dyn GrowthRateMethod,
    recalibrate: Recalibrate,
    offset: f64,
) -> DataFrame {
    let mut result = DataFrame::new();
    for name in df.names() {
        let single = df.select(&[name]);
        let (od, times, recalibrant) =
            perform_recalibration(&single, time_col, recalibrate, offset);
        if od.nrow() == 0 {
            warn!("No positive values found for column {} after filtering. Returning NaN.", name);
            result.insert_f64_column(name, vec![f64::NAN]);
            continue;
        }
        let gr = method.compute(&od, &times);
        result.insert_f64_column(name, vec![gr.max_od + recalibrant]);
    }
    result
}

/// Calculate time to maximum growth.
pub fn time_to_max_growth(
    df: &DataFrame,
    time_col: &DataFrame,
    method: &dyn GrowthRateMethod,
    recalibrate: Recalibrate,
    offset: f64,
) -> DataFrame {
    let mut result = DataFrame::new();
    for name in df.names() {
        let single = df.select(&[name]);
        let (od, times, _) = perform_recalibration(&single, time_col, recalibrate, offset);
        if od.nrow() == 0 {
            warn!("No positive values found for column {} after filtering. Returning NaN.", name);
            result.insert_f64_column(name, vec![f64::NAN]);
            continue;
        }
        let gr = method.compute(&od, &times);
        result.insert_f64_column(name, vec![gr.time_to_max_growth]);
    }
    result
}

/// Calculate OD at maximum growth.
pub fn od_at_max_growth(
    df: &DataFrame,
    time_col: &DataFrame,
    method: &dyn GrowthRateMethod,
    recalibrate: Recalibrate,
    offset: f64,
) -> DataFrame {
    let mut result = DataFrame::new();
    for name in df.names() {
        let single = df.select(&[name]);
        let (od, times, recalibrant) =
            perform_recalibration(&single, time_col, recalibrate, offset);
        if od.nrow() == 0 {
            warn!("No positive values found for column {} after filtering. Returning NaN.", name);
            result.insert_f64_column(name, vec![f64::NAN]);
            continue;
        }
        let gr = method.compute(&od, &times);
        result.insert_f64_column(name, vec![gr.od_at_max_growth + recalibrant]);
    }
    result
}

/// Calculate lag time.
pub fn lag_time(
    df: &DataFrame,
    time_col: &DataFrame,
    method: &dyn GrowthRateMethod,
    recalibrate: Recalibrate,
    offset: f64,
) -> DataFrame {
    let mut result = DataFrame::new();
    for name in df.names() {
        let single = df.select(&[name]);
        let (od, times, recalibrant) =
            perform_recalibration(&single, time_col, recalibrate, offset);
        if od.nrow() == 0 {
            warn!("No positive values found for column {} after filtering. Returning NaN.", name);
            result.insert_f64_column(name, vec![f64::NAN]);
            continue;
        }
        let gr = method.compute(&od, &times);
        let od_vals = od.column_f64(od.names()[0]).unwrap();
        let lt = compute_lagtime(
            gr.time_to_max_growth,
            gr.growth_rate,
            gr.od_at_max_growth + recalibrant,
            od_vals[0] + recalibrant,
        );
        result.insert_f64_column(name, vec![lt]);
    }
    result
}

fn compute_lagtime(time_at_max: f64, growth_rate: f64, od_at_max: f64, od_at_start: f64) -> f64 {
    time_at_max - (1.0 / growth_rate) * (od_at_max / od_at_start).ln()
}

fn perform_recalibration(
    df: &DataFrame,
    time_col: &DataFrame,
    recalibrate: Recalibrate,
    offset: f64,
) -> (DataFrame, DataFrame, f64) {
    let col_name = df.names()[0];
    let vals = df.column_f64(col_name).unwrap();

    match recalibrate {
        Recalibrate::Negative if vals.iter().any(|&v| v <= 0.0) => {
            let recalibrant = vals.iter().cloned().fold(f64::INFINITY, f64::min) - offset;
            let calibrated = calibrate::calibrate_with_offset(df, time_col, &MinData, offset);
            (calibrated, time_col.clone(), recalibrant)
        }
        Recalibrate::Always => {
            let recalibrant = vals.iter().cloned().fold(f64::INFINITY, f64::min) - offset;
            let calibrated = calibrate::calibrate_with_offset(df, time_col, &MinData, offset);
            (calibrated, time_col.clone(), recalibrant)
        }
        Recalibrate::Never => {
            // Filter out non-positive values
            let mask: Vec<bool> = vals.iter().map(|&v| v > 0.0).collect();
            let filtered = df.filter(&mask);
            let filtered_time = time_col.filter(&mask);
            (filtered, filtered_time, 0.0)
        }
        _ => (df.clone(), time_col.clone(), 0.0),
    }
}

// Helper to get value at a specific time
fn at_time_val(data: &[f64], times: &[f64], time_point_ms: f64) -> Option<f64> {
    let (_, last) = index_between_vals(times, 0.0, time_point_ms);
    last.map(|idx| data[idx])
}

/// Helper to compute between_times for a single column
fn between_times_indices(times: &[f64], mint_ms: f64, maxt_ms: f64) -> (Option<usize>, Option<usize>) {
    index_between_vals(times, mint_ms, maxt_ms)
}

// ---- Growth Rate Methods ----

/// Endpoints method: two-point growth calculation.
pub struct Endpoints {
    pub start_time: f64, // in minutes
    pub end_time: f64,
}

impl GrowthRateMethod for Endpoints {
    fn compute(&self, df: &DataFrame, time_col: &DataFrame) -> GrowthResult {
        let col_name = df.names()[0];
        let data = df.column_f64(col_name).unwrap();
        let times = time_col.column_f64(time_col.names()[0]).unwrap();

        let start_od = at_time_val(&data, &times, self.start_time * 60000.0).unwrap_or(f64::NAN);
        let end_od = at_time_val(&data, &times, self.end_time * 60000.0).unwrap_or(f64::NAN);
        let start_t = at_time_val(&times, &times, self.start_time * 60000.0).unwrap_or(0.0) / 60000.0;
        let end_t = at_time_val(&times, &times, self.end_time * 60000.0).unwrap_or(0.0) / 60000.0;

        let gr = (end_od.ln() - start_od.ln()) / (end_t - start_t);

        GrowthResult {
            growth_rate: gr,
            time_to_max_growth: (start_t + end_t) / 2.0,
            od_at_max_growth: ((start_od.ln() + end_od.ln()) / 2.0).exp(),
            max_od: data.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

/// Moving window method.
pub struct MovingWindow {
    pub window_size: usize,
    pub method: MovingWindowMethod,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MovingWindowMethod {
    Endpoints,
    LinearOnLog,
}

impl Default for MovingWindow {
    fn default() -> Self {
        MovingWindow {
            window_size: 10,
            method: MovingWindowMethod::Endpoints,
        }
    }
}

impl GrowthRateMethod for MovingWindow {
    fn compute(&self, df: &DataFrame, time_col: &DataFrame) -> GrowthResult {
        let col_name = df.names()[0];
        let data = df.column_f64(col_name).unwrap();
        let times = time_col.column_f64(time_col.names()[0]).unwrap();
        let n = data.len();

        let mut max_rate = f64::NEG_INFINITY;
        let mut best_time = f64::NAN;
        let mut best_od = f64::NAN;

        for j in 0..(n - self.window_size) {
            let start_t = times[j] / 60000.0;
            let end_t = times[j + self.window_size - 1] / 60000.0;

            let sub_method: Box<dyn GrowthRateMethod> = match self.method {
                MovingWindowMethod::Endpoints => Box::new(Endpoints {
                    start_time: start_t,
                    end_time: end_t,
                }),
                MovingWindowMethod::LinearOnLog => Box::new(LinearOnLog {
                    start_time: start_t,
                    end_time: end_t,
                }),
            };

            let result = sub_method.compute(df, time_col);

            if result.growth_rate > max_rate && !result.growth_rate.is_infinite() {
                max_rate = result.growth_rate;
                best_time = (start_t + end_t) / 2.0;

                let start_od = at_time_val(&data, &times, start_t * 60000.0).unwrap_or(1.0);
                let end_od = at_time_val(&data, &times, end_t * 60000.0).unwrap_or(1.0);
                best_od = ((start_od.ln() + end_od.ln()) / 2.0).exp();
            }
        }

        GrowthResult {
            growth_rate: max_rate,
            time_to_max_growth: best_time,
            od_at_max_growth: best_od,
            max_od: data.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

/// Linear regression on log-transformed data.
pub struct LinearOnLog {
    pub start_time: f64,
    pub end_time: f64,
}

impl GrowthRateMethod for LinearOnLog {
    fn compute(&self, df: &DataFrame, time_col: &DataFrame) -> GrowthResult {
        let col_name = df.names()[0];
        let data = df.column_f64(col_name).unwrap();
        let times = time_col.column_f64(time_col.names()[0]).unwrap();
        let n = data.len();

        if n < 2 {
            warn!("Not enough data points ({}) after log scaling and removing <= 0 values.", n);
            return GrowthResult {
                growth_rate: f64::NAN,
                time_to_max_growth: f64::NAN,
                od_at_max_growth: f64::NAN,
                max_od: f64::NAN,
            };
        }

        let (first, last) = between_times_indices(&times, self.start_time * 60000.0, self.end_time * 60000.0);

        if first.is_none() || last.is_none() {
            warn!(
                "No data points found between start_time={} and end_time={}. This may be due to negative OD values being removed.",
                self.start_time, self.end_time
            );
            return GrowthResult {
                growth_rate: f64::NAN,
                time_to_max_growth: f64::NAN,
                od_at_max_growth: f64::NAN,
                max_od: f64::NAN,
            };
        }

        let first = first.unwrap();
        let last = last.unwrap();

        let first_val = data[0];
        let t_slice: Vec<f64> = times[first..=last].iter().map(|&t| t / 60000.0).collect();
        let log_od: Vec<f64> = data[first..=last].iter().map(|&v| (v / first_val).ln()).collect();

        // Linear regression
        let n_pts = t_slice.len() as f64;
        let sx: f64 = t_slice.iter().sum();
        let sy: f64 = log_od.iter().sum();
        let sxx: f64 = t_slice.iter().map(|&t| t * t).sum();
        let sxy: f64 = t_slice.iter().zip(log_od.iter()).map(|(&t, &y)| t * y).sum();

        let slope = (n_pts * sxy - sx * sy) / (n_pts * sxx - sx * sx);

        // Geometric mean of OD in window
        let window_data: Vec<f64> = data[first..=last].to_vec();
        let geomean = (window_data.iter().map(|v| v.ln()).sum::<f64>() / window_data.len() as f64).exp();

        GrowthResult {
            growth_rate: slope,
            time_to_max_growth: (self.start_time + self.end_time) / 2.0,
            od_at_max_growth: geomean,
            max_od: data.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

/// Finite difference method.
pub struct FiniteDiff {
    pub diff_type: FiniteDiffType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FiniteDiffType {
    Central,
    OneSided,
}

impl Default for FiniteDiff {
    fn default() -> Self {
        FiniteDiff {
            diff_type: FiniteDiffType::Central,
        }
    }
}

impl GrowthRateMethod for FiniteDiff {
    fn compute(&self, df: &DataFrame, time_col: &DataFrame) -> GrowthResult {
        let col_name = df.names()[0];
        let data = df.column_f64(col_name).unwrap();
        let times_raw = time_col.column_f64(time_col.names()[0]).unwrap();

        // Filter positive values
        let mut t: Vec<f64> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        for (&ti, &yi) in times_raw.iter().zip(data.iter()) {
            if yi > 0.0 {
                t.push(ti / 60000.0);
                y.push(yi);
            }
        }

        let n = t.len();
        if n < 2 {
            warn!("Not enough data points ({}) after log scaling and removing <= 0 values.", n);
            return GrowthResult {
                growth_rate: f64::NAN,
                time_to_max_growth: f64::NAN,
                od_at_max_growth: f64::NAN,
                max_od: f64::NAN,
            };
        }

        let first_y = y[0];
        let ly: Vec<f64> = y.iter().map(|&v| (v / first_y).ln()).collect();

        let mut max_gr = 0.0_f64;
        let mut best_time = f64::NAN;
        let mut best_od = f64::NAN;

        match self.diff_type {
            FiniteDiffType::Central => {
                for k in 1..(n - 1) {
                    let deriv = (ly[k + 1] - ly[k - 1]) / (t[k + 1] - t[k - 1]);
                    if deriv > max_gr {
                        max_gr = deriv;
                        best_time = t[k];
                        best_od = y[k];
                    }
                }
            }
            FiniteDiffType::OneSided => {
                for k in 0..(n - 1) {
                    let deriv = (ly[k + 1] - ly[k]) / (t[k + 1] - t[k]);
                    if deriv > max_gr {
                        max_gr = deriv;
                        best_time = (t[k] + t[k + 1]) / 2.0;
                        best_od = ((ly[k] + ly[k + 1]) / 2.0).exp() * first_y;
                    }
                }
            }
        }

        GrowthResult {
            growth_rate: max_gr,
            time_to_max_growth: best_time,
            od_at_max_growth: best_od,
            max_od: data.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

/// Parametric growth rate fitting.
pub struct ParametricGrowthRate {
    pub func: Box<dyn Fn(f64, &[f64]) -> f64>,
    pub initial_params: Vec<f64>,
}

// Parametric model constructors
pub fn logistic() -> ParametricGrowthRate {
    ParametricGrowthRate {
        func: Box::new(|t, p| p[1] / (1.0 + (4.0 * p[0] / p[1] * (p[2] - t) + 2.0).exp())),
        initial_params: vec![1.0, 4.0, 5.0],
    }
}

pub fn gompertz() -> ParametricGrowthRate {
    ParametricGrowthRate {
        func: Box::new(|t, p| {
            p[1] * (-(p[0] * std::f64::consts::E / p[1] * (p[2] - t) + 1.0).exp()).exp()
        }),
        initial_params: vec![1.0, 4.0, 5.0],
    }
}

pub fn modified_gompertz() -> ParametricGrowthRate {
    ParametricGrowthRate {
        func: Box::new(|t, p| {
            p[1] * (-(p[0] * std::f64::consts::E / p[1] * (p[2] - t) + 1.0).exp()).exp()
                + p[1] * (p[3] * (t - p[4])).exp()
        }),
        initial_params: vec![1.0, 4.0, 5.0, 1.0, 4.0],
    }
}

pub fn richards() -> ParametricGrowthRate {
    ParametricGrowthRate {
        func: Box::new(|t, p| {
            let nu = p[3].exp();
            let denom = 1.0
                + nu * (1.0 + nu).exp()
                    * (p[0] / p[1] * (1.0 + nu).powf(1.0 + 1.0 / nu) * (p[2] - t)).exp();
            p[1] / denom.powf(1.0 / nu)
        }),
        initial_params: vec![1.0, 4.0, 5.0, 0.0],
    }
}

impl GrowthRateMethod for ParametricGrowthRate {
    fn compute(&self, df: &DataFrame, time_col: &DataFrame) -> GrowthResult {
        let col_name = df.names()[0];
        let data = df.column_f64(col_name).unwrap();
        let times_raw = time_col.column_f64(time_col.names()[0]).unwrap();

        // Convert to minutes and filter positive values
        let mut t: Vec<f64> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        for (&ti, &yi) in times_raw.iter().zip(data.iter()) {
            if yi > 0.0 {
                t.push(ti / 60000.0);
                y.push(yi);
            }
        }

        let n = t.len();
        if n < 2 {
            warn!("Not enough data points ({}) after log scaling and removing <= 0 values.", n);
            return GrowthResult {
                growth_rate: f64::NAN,
                time_to_max_growth: f64::NAN,
                od_at_max_growth: f64::NAN,
                max_od: f64::NAN,
            };
        }

        let first_y = y[0];
        let ly: Vec<f64> = y.iter().map(|&v| (v / first_y).ln()).collect();

        // Levenberg-Marquardt style optimization
        let mut params = self.initial_params.clone();
        let mut lambda = 0.001;
        let max_iter = 500;
        let np = params.len();

        let compute_ssr = |p: &[f64]| -> f64 {
            t.iter()
                .zip(ly.iter())
                .map(|(&ti, &lyi)| {
                    let r = (self.func)(ti, p) - lyi;
                    r * r
                })
                .sum()
        };

        let mut current_ssr = compute_ssr(&params);

        for _ in 0..max_iter {
            let residuals: Vec<f64> = t
                .iter()
                .zip(ly.iter())
                .map(|(&ti, &lyi)| (self.func)(ti, &params) - lyi)
                .collect();

            // Compute Jacobian numerically
            let eps = 1e-8;
            let mut jac = vec![vec![0.0; np]; n];
            for j in 0..np {
                let mut params_plus = params.clone();
                params_plus[j] += eps;
                for i in 0..n {
                    let f_plus = (self.func)(t[i], &params_plus);
                    let f_base = (self.func)(t[i], &params);
                    jac[i][j] = (f_plus - f_base) / eps;
                }
            }

            // J^T * J + lambda * diag(J^T*J)
            let mut jtj = vec![vec![0.0; np]; np];
            let mut jtr = vec![0.0; np];
            for i in 0..n {
                for j in 0..np {
                    jtr[j] -= jac[i][j] * residuals[i];
                    for k in 0..np {
                        jtj[j][k] += jac[i][j] * jac[i][k];
                    }
                }
            }
            for j in 0..np {
                jtj[j][j] *= 1.0 + lambda;
                if jtj[j][j] == 0.0 {
                    jtj[j][j] = lambda;
                }
            }

            if let Some(delta) = solve_linear(&jtj, &jtr) {
                let mut new_params = params.clone();
                for j in 0..np {
                    new_params[j] += delta[j];
                }
                let new_ssr = compute_ssr(&new_params);
                if new_ssr < current_ssr {
                    params = new_params;
                    current_ssr = new_ssr;
                    lambda *= 0.5;
                } else {
                    lambda *= 2.0;
                }
            } else {
                lambda *= 10.0;
            }

            if current_ssr < 1e-12 {
                break;
            }
        }

        let gr = params[0];

        // Find time to max growth using refined time grid
        let t_min = *t.first().unwrap();
        let t_max = *t.last().unwrap();
        let n_refined = 100 * n;
        let dt = (t_max - t_min) / (n_refined as f64 - 1.0);

        let mut best_time = t_min;
        let mut min_diff = f64::INFINITY;

        for i in 0..n_refined {
            let ti = t_min + dt * i as f64;
            // Numerical derivative
            let f_plus = (self.func)(ti + eps_deriv(), &params);
            let f_minus = (self.func)(ti - eps_deriv(), &params);
            let deriv = (f_plus - f_minus) / (2.0 * eps_deriv());
            let diff = (deriv - gr).abs();
            if diff < min_diff {
                min_diff = diff;
                best_time = ti;
            }
        }

        let od_at_max = ((self.func)(best_time, &params)).exp() * first_y;
        let max_od_val = params[1].exp() * first_y;

        GrowthResult {
            growth_rate: gr,
            time_to_max_growth: best_time,
            od_at_max_growth: od_at_max,
            max_od: max_od_val,
        }
    }
}

fn eps_deriv() -> f64 {
    1e-6
}

/// Regularization smoothing method.
pub struct Regularization {
    pub order: usize,
}

impl Default for Regularization {
    fn default() -> Self {
        Regularization { order: 4 }
    }
}

impl GrowthRateMethod for Regularization {
    fn compute(&self, df: &DataFrame, time_col: &DataFrame) -> GrowthResult {
        let col_name = df.names()[0];
        let data = df.column_f64(col_name).unwrap();
        let times_raw = time_col.column_f64(time_col.names()[0]).unwrap();

        let mut t: Vec<f64> = Vec::new();
        let mut y: Vec<f64> = Vec::new();
        for (&ti, &yi) in times_raw.iter().zip(data.iter()) {
            if yi > 0.0 {
                t.push(ti / 60000.0);
                y.push(yi);
            }
        }

        let n = t.len();
        if n < 2 {
            warn!("Not enough data points ({}) after log scaling and removing <= 0 values.", n);
            return GrowthResult {
                growth_rate: f64::NAN,
                time_to_max_growth: f64::NAN,
                od_at_max_growth: f64::NAN,
                max_od: f64::NAN,
            };
        }

        let first_y = y[0];
        let ly: Vec<f64> = y.iter().map(|&v| (v / first_y).ln()).collect();

        // Simple regularization: fit a smoothing polynomial using Tikhonov regularization
        // Use a high-degree polynomial with regularization
        let degree = self.order.min(n - 1);

        // Build Vandermonde matrix
        let t_min = t[0];
        let t_max = t[n - 1];
        let t_norm: Vec<f64> = t.iter().map(|&ti| (ti - t_min) / (t_max - t_min)).collect();

        let mut vandermonde = vec![vec![0.0; degree + 1]; n];
        for i in 0..n {
            for j in 0..=degree {
                vandermonde[i][j] = t_norm[i].powi(j as i32);
            }
        }

        // Regularized least squares: (V^T V + lambda * I) * c = V^T * y
        let lambda = 0.001; // regularization parameter
        let np = degree + 1;
        let mut vtv = vec![vec![0.0; np]; np];
        let mut vty = vec![0.0; np];

        for i in 0..n {
            for j in 0..np {
                vty[j] += vandermonde[i][j] * ly[i];
                for k in 0..np {
                    vtv[j][k] += vandermonde[i][j] * vandermonde[i][k];
                }
            }
        }
        for j in 0..np {
            vtv[j][j] += lambda;
        }

        let coeffs = solve_linear(&vtv, &vty).unwrap_or_else(|| vec![0.0; np]);

        // Evaluate derivative on refined grid
        let n_refined = 100 * n;
        let mut max_deriv = f64::NEG_INFINITY;
        let mut best_time = t[0];
        let mut best_val = 0.0;

        for i in 0..n_refined {
            let tn = i as f64 / (n_refined as f64 - 1.0);
            let ti = t_min + tn * (t_max - t_min);

            // Evaluate derivative of polynomial
            let mut deriv = 0.0;
            for j in 1..=degree {
                deriv += coeffs[j] * j as f64 * tn.powi((j - 1) as i32) / (t_max - t_min);
            }

            if deriv > max_deriv {
                max_deriv = deriv;
                best_time = ti;
                // Evaluate polynomial value
                let mut val = 0.0;
                for j in 0..=degree {
                    val += coeffs[j] * tn.powi(j as i32);
                }
                best_val = val;
            }
        }

        // Find max OD from the smoothed curve
        let mut max_val = f64::NEG_INFINITY;
        for i in 0..n_refined {
            let tn = i as f64 / (n_refined as f64 - 1.0);
            let mut val = 0.0;
            for j in 0..=degree {
                val += coeffs[j] * tn.powi(j as i32);
            }
            if val > max_val {
                max_val = val;
            }
        }

        GrowthResult {
            growth_rate: max_deriv,
            time_to_max_growth: best_time,
            od_at_max_growth: best_val.exp() * first_y,
            max_od: max_val.exp() * first_y,
        }
    }
}

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return None;
        }

        aug.swap(col, max_row);

        // Eliminate below
        for row in (col + 1)..n {
            let factor = aug[row][col] / aug[col][col];
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exponential_data() -> (DataFrame, DataFrame) {
        // Perfect exponential: doubling every minute
        // OD = 0.05 * 2^t where t is in minutes
        let od_vals: Vec<f64> = (0..11).map(|i| 0.05 * 2.0_f64.powi(i)).collect();
        let time_vals: Vec<f64> = (0..11).map(|i| (i as f64) * 60000.0).collect();

        let mut od = DataFrame::new();
        od.insert_f64_column("A", od_vals);
        let mut time = DataFrame::new();
        time.insert_f64_column("Time", time_vals);
        (od, time)
    }

    #[test]
    fn test_growth_rate_moving_window() {
        let (od, time) = make_exponential_data();
        let method = MovingWindow {
            window_size: 3,
            method: MovingWindowMethod::Endpoints,
        };
        let result = growth_rate(&od, &time, &method, Recalibrate::Negative, 0.001);
        let gr = result.column_f64("A").unwrap()[0];
        assert!(
            (gr - 2.0_f64.ln()).abs() < 1e-6,
            "growth rate {} != {}",
            gr,
            2.0_f64.ln()
        );
    }

    #[test]
    fn test_growth_rate_endpoints() {
        let (od, time) = make_exponential_data();
        let method = Endpoints {
            start_time: 1.0,
            end_time: 5.0,
        };
        let result = growth_rate(&od, &time, &method, Recalibrate::Negative, 0.001);
        let gr = result.column_f64("A").unwrap()[0];
        assert!(
            (gr - 2.0_f64.ln()).abs() < 1e-6,
            "growth rate {} != {}",
            gr,
            2.0_f64.ln()
        );
    }

    #[test]
    fn test_growth_rate_linear_on_log() {
        let (od, time) = make_exponential_data();
        let method = LinearOnLog {
            start_time: 1.0,
            end_time: 5.0,
        };
        let result = growth_rate(&od, &time, &method, Recalibrate::Negative, 0.001);
        let gr = result.column_f64("A").unwrap()[0];
        assert!(
            (gr - 2.0_f64.ln()).abs() < 1e-6,
            "growth rate {} != {}",
            gr,
            2.0_f64.ln()
        );
    }

    #[test]
    fn test_growth_rate_finite_diff_central() {
        let (od, time) = make_exponential_data();
        let method = FiniteDiff::default();
        let result = growth_rate(&od, &time, &method, Recalibrate::Negative, 0.001);
        let gr = result.column_f64("A").unwrap()[0];
        assert!(
            (gr - 2.0_f64.ln()).abs() < 1e-6,
            "growth rate {} != {}",
            gr,
            2.0_f64.ln()
        );
    }

    #[test]
    fn test_growth_rate_finite_diff_onesided() {
        let (od, time) = make_exponential_data();
        let method = FiniteDiff {
            diff_type: FiniteDiffType::OneSided,
        };
        let result = growth_rate(&od, &time, &method, Recalibrate::Negative, 0.001);
        let gr = result.column_f64("A").unwrap()[0];
        assert!(
            (gr - 2.0_f64.ln()).abs() < 1e-6,
            "growth rate {} != {}",
            gr,
            2.0_f64.ln()
        );
    }

    #[test]
    fn test_growth_rate_regularization() {
        let (od, time) = make_exponential_data();
        let method = Regularization { order: 4 };
        let result = growth_rate(&od, &time, &method, Recalibrate::Negative, 0.001);
        let gr = result.column_f64("A").unwrap()[0];
        assert!(
            (gr - 2.0_f64.ln()).abs() < 1e-2,
            "growth rate {} != {} (tol 0.01)",
            gr,
            2.0_f64.ln()
        );
    }

    #[test]
    fn test_doubling_time() {
        let (od, time) = make_exponential_data();
        let method = MovingWindow {
            window_size: 3,
            method: MovingWindowMethod::Endpoints,
        };
        let result = doubling_time(&od, &time, &method, Recalibrate::Negative, 0.001);
        let dt = result.column_f64("A").unwrap()[0];
        assert!((dt - 1.0).abs() < 1e-6, "doubling time {} != 1.0", dt);
    }

    #[test]
    fn test_parametric_logistic() {
        // Use a sigmoidal curve
        let n = 61;
        let time_vals: Vec<f64> = (0..n).map(|i| (i as f64) * 10000.0).collect();
        let od_vals: Vec<f64> = time_vals
            .iter()
            .map(|&t| {
                let t_min = t / 60000.0;
                (0.7 / (1.0 + (-2.0 * (t_min - 5.0)).exp())).exp()
            })
            .collect();

        let mut od = DataFrame::new();
        od.insert_f64_column("A", od_vals);
        let mut time = DataFrame::new();
        time.insert_f64_column("Time", time_vals);

        let method = logistic();
        let result = growth_rate(&od, &time, &method, Recalibrate::Negative, 0.001);
        let gr = result.column_f64("A").unwrap()[0];
        assert!(
            (gr - 0.35).abs() < 0.05,
            "logistic growth rate {} not near 0.35",
            gr
        );
    }
}
