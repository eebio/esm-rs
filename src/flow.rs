use crate::dataframe::DataFrame;

// ---- Gate Types ----

/// Trait for all gating methods.
pub trait GatingMethod {
    fn apply(&self, data: &DataFrame) -> DataFrame;
}

/// High/Low gate on a single channel.
#[derive(Debug, Clone)]
pub struct HighLowGate {
    pub channel: String,
    pub min: f64,
    pub max: f64,
}

impl HighLowGate {
    pub fn new(channel: &str) -> Self {
        HighLowGate {
            channel: channel.to_string(),
            min: f64::NEG_INFINITY,
            max: f64::INFINITY,
        }
    }

    pub fn with_min(mut self, min: f64) -> Self {
        self.min = min;
        self
    }

    pub fn with_max(mut self, max: f64) -> Self {
        self.max = max;
        self
    }

    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.min = min;
        self.max = max;
        self
    }
}

impl GatingMethod for HighLowGate {
    fn apply(&self, data: &DataFrame) -> DataFrame {
        let col = data.column_f64(&self.channel).unwrap();
        let mask: Vec<bool> = col.iter().map(|&v| v >= self.min && v < self.max).collect();
        data.filter(&mask)
    }
}

/// Rectangle gate on two channels.
#[derive(Debug, Clone)]
pub struct RectangleGate {
    pub channel_x: String,
    pub channel_y: String,
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

impl GatingMethod for RectangleGate {
    fn apply(&self, data: &DataFrame) -> DataFrame {
        let x = data.column_f64(&self.channel_x).unwrap();
        let y = data.column_f64(&self.channel_y).unwrap();
        let mask: Vec<bool> = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| {
                xi >= self.x_min && xi < self.x_max && yi >= self.y_min && yi < self.y_max
            })
            .collect();
        data.filter(&mask)
    }
}

/// Quadrant gate on two channels.
#[derive(Debug, Clone)]
pub struct QuadrantGate {
    pub channel_x: String,
    pub channel_y: String,
    pub x_cutoff: f64,
    pub y_cutoff: f64,
    pub quadrant: i32,
}

impl GatingMethod for QuadrantGate {
    fn apply(&self, data: &DataFrame) -> DataFrame {
        let x = data.column_f64(&self.channel_x).unwrap();
        let y = data.column_f64(&self.channel_y).unwrap();
        let mask: Vec<bool> = match self.quadrant {
            1 => x
                .iter()
                .zip(y.iter())
                .map(|(&xi, &yi)| xi >= self.x_cutoff && yi >= self.y_cutoff)
                .collect(),
            2 => x
                .iter()
                .zip(y.iter())
                .map(|(&xi, &yi)| xi >= self.x_cutoff && yi < self.y_cutoff)
                .collect(),
            3 => x
                .iter()
                .zip(y.iter())
                .map(|(&xi, &yi)| xi < self.x_cutoff && yi < self.y_cutoff)
                .collect(),
            4 => x
                .iter()
                .zip(y.iter())
                .map(|(&xi, &yi)| xi < self.x_cutoff && yi >= self.y_cutoff)
                .collect(),
            _ => panic!("Quadrant must be between 1 and 4."),
        };
        data.filter(&mask)
    }
}

/// Polygon gate on two channels.
#[derive(Debug, Clone)]
pub struct PolygonGate {
    pub channel_x: String,
    pub channel_y: String,
    pub points: Vec<(f64, f64)>,
}

impl GatingMethod for PolygonGate {
    fn apply(&self, data: &DataFrame) -> DataFrame {
        let x = data.column_f64(&self.channel_x).unwrap();
        let y = data.column_f64(&self.channel_y).unwrap();
        let mask: Vec<bool> = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| point_in_polygon(xi, yi, &self.points))
            .collect();
        data.filter(&mask)
    }
}

/// Test if a point is inside or on the boundary of a polygon.
///
/// Matches Julia's `Meshes.PolyArea` membership semantics, which treats
/// points lying exactly on an edge as inside. Edge points are detected
/// first; otherwise standard ray-casting is used.
fn point_in_polygon(px: f64, py: f64, polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    for i in 0..n {
        let (x1, y1) = polygon[i];
        let (x2, y2) = polygon[(i + 1) % n];
        if point_on_segment(px, py, x1, y1, x2, y2) {
            return true;
        }
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn point_on_segment(px: f64, py: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> bool {
    let cross = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1);
    if cross.abs() > 1e-9 {
        return false;
    }
    let dot = (px - x1) * (x2 - x1) + (py - y1) * (y2 - y1);
    let len_sq = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
    dot >= 0.0 && dot <= len_sq
}

/// Ellipse gate on two channels.
#[derive(Debug, Clone)]
pub struct EllipseGate {
    pub channel_x: String,
    pub channel_y: String,
    pub center: (f64, f64),
    pub a: f64,
    pub b: f64,
    pub angle: f64, // in degrees
}

impl EllipseGate {
    /// Create an EllipseGate from points (and optional center).
    /// If center is provided and fewer than 5 points given, reflects points through center.
    pub fn from_points(
        channel_x: &str,
        channel_y: &str,
        center: Option<(f64, f64)>,
        points: &[(f64, f64)],
    ) -> Self {
        let mut xs: Vec<f64> = points.iter().map(|p| p.0).collect();
        let mut ys: Vec<f64> = points.iter().map(|p| p.1).collect();

        if xs.len() < 5 {
            if let Some((cx, cy)) = center {
                let orig_len = xs.len();
                for i in 0..(5 - orig_len) {
                    let idx = i % orig_len;
                    xs.push(points[idx].0 + 2.0 * (cx - points[idx].0));
                    ys.push(points[idx].1 + 2.0 * (cy - points[idx].1));
                }
            } else {
                panic!(
                    "At least 3 points and a center or 5 points without a center are required."
                );
            }
        }

        let (a, b, theta, cx, cy, _) = crate::fit_ellipse::fit_ellipse(&xs, &ys);
        let angle = theta.to_degrees();

        EllipseGate {
            channel_x: channel_x.to_string(),
            channel_y: channel_y.to_string(),
            center: (cx, cy),
            a,
            b,
            angle,
        }
    }
}

impl GatingMethod for EllipseGate {
    fn apply(&self, data: &DataFrame) -> DataFrame {
        let x = data.column_f64(&self.channel_x).unwrap();
        let y = data.column_f64(&self.channel_y).unwrap();
        let cos_a = self.angle.to_radians().cos();
        let sin_a = self.angle.to_radians().sin();
        let (cx, cy) = self.center;
        let (a, b) = (self.a, self.b);

        let mask: Vec<bool> = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| {
                let x_rot = cos_a * (xi - cx) + sin_a * (yi - cy);
                let y_rot = -sin_a * (xi - cx) + cos_a * (yi - cy);
                let val = (x_rot * x_rot) / (a * a) + (y_rot * y_rot) / (b * b);
                val <= 1.0
            })
            .collect();
        data.filter(&mask)
    }
}

/// KDE-based automatic gating on two channels.
#[derive(Debug, Clone)]
pub struct KdeGate {
    pub channels: Vec<String>,
    pub gate_frac: f64,
    pub nbins: usize,
}

impl KdeGate {
    pub fn new(channels: Vec<String>) -> Self {
        KdeGate {
            channels,
            gate_frac: 0.65,
            nbins: 1024,
        }
    }

    pub fn with_gate_frac(mut self, frac: f64) -> Self {
        self.gate_frac = frac;
        self
    }
}

impl GatingMethod for KdeGate {
    fn apply(&self, data: &DataFrame) -> DataFrame {
        assert_eq!(
            self.channels.len(),
            2,
            "2 channels must be specified for density gating."
        );

        let x = data.column_f64(&self.channels[0]).unwrap();
        let y = data.column_f64(&self.channels[1]).unwrap();
        let n = x.len();

        // Simple 2D KDE using Gaussian kernel
        let density_values = kde_2d(&x, &y);

        // Sort by density, keep top fraction
        let mut indexed: Vec<(usize, f64)> =
            density_values.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Match Julia's KernelDensity-based gate (src/flow.jl): the threshold
        // is taken from `sorted_indices[ceil(gate_frac * N)]` in 1-based
        // indexing, then events with `density > threshold` are kept.
        let k_one_based = (self.gate_frac * n as f64).ceil() as usize;
        let top_idx = k_one_based.saturating_sub(1).min(n - 1);
        let threshold = indexed[top_idx].1;

        let mask: Vec<bool> = density_values.iter().map(|&d| d > threshold).collect();
        data.filter(&mask)
    }
}

/// Simple 2D Gaussian KDE.
fn kde_2d(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len() as f64;

    // Silverman's rule of thumb for bandwidth
    let std_x = std_dev(x);
    let std_y = std_dev(y);
    let bw_x = 1.06 * std_x * n.powf(-0.2);
    let bw_y = 1.06 * std_y * n.powf(-0.2);

    if bw_x == 0.0 || bw_y == 0.0 {
        return vec![1.0; x.len()];
    }

    // For each point, compute KDE value
    let mut density = vec![0.0; x.len()];
    for i in 0..x.len() {
        for j in 0..x.len() {
            let dx = (x[i] - x[j]) / bw_x;
            let dy = (y[i] - y[j]) / bw_y;
            density[i] += (-0.5 * (dx * dx + dy * dy)).exp();
        }
        density[i] /= n * 2.0 * std::f64::consts::PI * bw_x * bw_y;
    }
    density
}

fn std_dev(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

// ---- Logical Gates ----

/// AND gate: intersection of two gates.
pub struct AndGate<A: GatingMethod, B: GatingMethod> {
    pub gate1: A,
    pub gate2: B,
}

impl<A: GatingMethod, B: GatingMethod> GatingMethod for AndGate<A, B> {
    fn apply(&self, data: &DataFrame) -> DataFrame {
        let data = self.gate1.apply(data);
        self.gate2.apply(&data)
    }
}

/// OR gate: union of two gates.
pub struct OrGate<A: GatingMethod, B: GatingMethod> {
    pub gate1: A,
    pub gate2: B,
}

impl<A: GatingMethod, B: GatingMethod> GatingMethod for OrGate<A, B> {
    fn apply(&self, data: &DataFrame) -> DataFrame {
        let data1 = self.gate1.apply(data);
        let data2 = self.gate2.apply(data);

        let ids1: Vec<i64> = data1
            .column("id")
            .unwrap()
            .iter()
            .filter_map(|v| v.as_i64())
            .collect();
        let ids2: Vec<i64> = data2
            .column("id")
            .unwrap()
            .iter()
            .filter_map(|v| v.as_i64())
            .collect();

        let all_ids = data.column("id").unwrap();
        let mask: Vec<bool> = all_ids
            .iter()
            .map(|v| {
                let id = v.as_i64().unwrap();
                ids1.contains(&id) || ids2.contains(&id)
            })
            .collect();
        data.filter(&mask)
    }
}

/// NOT gate: complement of a gate.
pub struct NotGate<A: GatingMethod> {
    pub gate1: A,
}

impl<A: GatingMethod> GatingMethod for NotGate<A> {
    fn apply(&self, data: &DataFrame) -> DataFrame {
        let gated = self.gate1.apply(data);
        let gated_ids: Vec<i64> = gated
            .column("id")
            .unwrap()
            .iter()
            .filter_map(|v| v.as_i64())
            .collect();

        let all_ids = data.column("id").unwrap();
        let mask: Vec<bool> = all_ids
            .iter()
            .map(|v| {
                let id = v.as_i64().unwrap();
                !gated_ids.contains(&id)
            })
            .collect();
        data.filter(&mask)
    }
}

// ---- Convenience Functions ----

/// Apply a gate to data.
pub fn gate(data: &DataFrame, method: &dyn GatingMethod) -> DataFrame {
    method.apply(data)
}

/// Count events in flow data.
pub fn event_count(data: &DataFrame) -> usize {
    data.nrow()
}

/// Calculate proportion of events after gating.
pub fn gated_proportion_from_gate(data: &DataFrame, method: &dyn GatingMethod) -> f64 {
    let total = event_count(data);
    let gated = event_count(&gate(data, method));
    gated as f64 / total as f64
}

/// Calculate proportion from before/after data.
pub fn gated_proportion(before: &DataFrame, after: &DataFrame) -> f64 {
    event_count(after) as f64 / event_count(before) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataframe::Value;

    fn mock_flow_data() -> DataFrame {
        let mut df = DataFrame::new();
        df.insert_f64_column("FSC_A", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        df.insert_f64_column("FSC_A.min", vec![0.0; 5]);
        df.insert_f64_column("FSC_A.max", vec![1e5; 5]);
        df.insert_f64_column("SSC_A", vec![100.0, 200.0, 300.0, 400.0, 500.0]);
        df.insert_f64_column("SSC_A.min", vec![0.0; 5]);
        df.insert_f64_column("SSC_A.max", vec![1e5; 5]);
        df.insert_f64_column("FL1_A", vec![510.0, 520.0, 530.0, 540.0, 550.0]);
        df.insert_f64_column("FL1_A.min", vec![0.0; 5]);
        df.insert_f64_column("FL1_A.max", vec![1e5; 5]);
        df.insert_f64_column("FL2_A", vec![51.0, 52.0, 53.0, 54.0, 55.0]);
        df.insert_f64_column("FL2_A.min", vec![0.0; 5]);
        df.insert_f64_column("FL2_A.max", vec![1e5; 5]);
        df.insert_column(
            "id",
            vec![
                Value::Int(1),
                Value::Int(2),
                Value::Int(3),
                Value::Int(4),
                Value::Int(5),
            ],
        );
        df
    }

    #[test]
    fn test_highlow_gate() {
        let data = mock_flow_data();
        let g = HighLowGate::new("FL1_A").with_min(515.0);
        let result = gate(&data, &g);
        assert_eq!(result.column_f64("FL1_A").unwrap(), vec![520.0, 530.0, 540.0, 550.0]);

        let g = HighLowGate::new("SSC_A").with_max(301.0);
        let result = gate(&data, &g);
        assert_eq!(result.column_f64("SSC_A").unwrap(), vec![100.0, 200.0, 300.0]);

        // Upper bounds exclusive, lower bounds inclusive
        let g = HighLowGate::new("FSC_A").with_range(2.0, 4.0);
        let result = gate(&data, &g);
        assert_eq!(result.column_f64("FL1_A").unwrap(), vec![520.0, 530.0]);

        // No limits = all data
        let g = HighLowGate::new("FSC_A");
        let result = gate(&data, &g);
        assert_eq!(result.column_f64("FL2_A").unwrap(), vec![51.0, 52.0, 53.0, 54.0, 55.0]);
    }

    #[test]
    fn test_rectangle_gate() {
        let data = mock_flow_data();
        let g = RectangleGate {
            channel_x: "FSC_A".to_string(),
            channel_y: "SSC_A".to_string(),
            x_min: 2.0,
            x_max: 4.5,
            y_min: 50.0,
            y_max: 301.0,
        };
        let result = gate(&data, &g);
        assert_eq!(result.column_f64("FSC_A").unwrap(), vec![2.0, 3.0]);

        let g = RectangleGate {
            channel_x: "FL1_A".to_string(),
            channel_y: "FL2_A".to_string(),
            x_min: 0.0,
            x_max: 545.0,
            y_min: 52.0,
            y_max: 54.0,
        };
        let result = gate(&data, &g);
        assert_eq!(result.column_f64("SSC_A").unwrap(), vec![200.0, 300.0]);
    }

    #[test]
    fn test_quadrant_gate() {
        let data = mock_flow_data();

        let g = QuadrantGate {
            channel_x: "FL1_A".to_string(),
            channel_y: "FL2_A".to_string(),
            x_cutoff: 535.0,
            y_cutoff: 54.5,
            quadrant: 1,
        };
        assert_eq!(gate(&data, &g).column_f64("FL1_A").unwrap(), vec![550.0]);

        let g = QuadrantGate { quadrant: 2, ..g.clone() };
        assert_eq!(gate(&data, &g).column_f64("FL1_A").unwrap(), vec![540.0]);

        let g = QuadrantGate { quadrant: 3, channel_x: "FL1_A".to_string(), channel_y: "FL2_A".to_string(), x_cutoff: 535.0, y_cutoff: 54.5 };
        assert_eq!(
            gate(&data, &g).column_f64("SSC_A").unwrap(),
            vec![100.0, 200.0, 300.0]
        );

        let g = QuadrantGate { quadrant: 4, channel_x: "FL1_A".to_string(), channel_y: "FL2_A".to_string(), x_cutoff: 535.0, y_cutoff: 54.5 };
        assert!(gate(&data, &g).column_f64("FL1_A").unwrap().is_empty());
    }

    #[test]
    #[should_panic(expected = "Quadrant must be between 1 and 4")]
    fn test_quadrant_gate_invalid() {
        let data = mock_flow_data();
        let g = QuadrantGate {
            channel_x: "FL1_A".to_string(),
            channel_y: "FL2_A".to_string(),
            x_cutoff: 535.0,
            y_cutoff: 54.5,
            quadrant: 5,
        };
        gate(&data, &g);
    }

    #[test]
    fn test_polygon_gate() {
        let data = mock_flow_data();
        let polygon = vec![
            (1.5, 150.0),
            (3.0, 100.0),
            (4.5, 150.0),
            (4.0, 450.0),
            (2.0, 400.0),
        ];
        let g = PolygonGate {
            channel_x: "FSC_A".to_string(),
            channel_y: "SSC_A".to_string(),
            points: polygon,
        };
        assert_eq!(
            gate(&data, &g).column_f64("FSC_A").unwrap(),
            vec![2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_event_counting() {
        let data = mock_flow_data();
        assert_eq!(event_count(&data), 5);

        let g = HighLowGate::new("FL1_A").with_min(530.0);
        assert_eq!(event_count(&gate(&data, &g)), 3);

        let g = RectangleGate {
            channel_x: "FSC_A".to_string(),
            channel_y: "SSC_A".to_string(),
            x_min: 2.0,
            x_max: 4.5,
            y_min: 50.0,
            y_max: 301.0,
        };
        assert_eq!(event_count(&gate(&data, &g)), 2);
    }

    #[test]
    fn test_gated_proportions() {
        let data = mock_flow_data();
        let g = HighLowGate::new("FL1_A").with_min(530.0);
        let gated = gate(&data, &g);
        assert_eq!(gated_proportion(&data, &gated), 3.0 / 5.0);
        assert_eq!(gated_proportion_from_gate(&data, &g), 3.0 / 5.0);
    }

    #[test]
    fn test_logical_and_gate() {
        let data = mock_flow_data();
        let g = AndGate {
            gate1: HighLowGate::new("FL1_A").with_min(525.0),
            gate2: HighLowGate::new("SSC_A").with_max(450.0),
        };
        let result = gate(&data, &g);
        let ids: Vec<i64> = result
            .column("id")
            .unwrap()
            .iter()
            .filter_map(|v| v.as_i64())
            .collect();
        assert_eq!(ids, vec![3, 4]);
    }

    #[test]
    fn test_logical_or_gate() {
        let data = mock_flow_data();
        let g = OrGate {
            gate1: HighLowGate::new("FL1_A").with_range(525.0, 545.0),
            gate2: HighLowGate::new("SSC_A").with_max(150.0),
        };
        let result = gate(&data, &g);
        let ids: Vec<i64> = result
            .column("id")
            .unwrap()
            .iter()
            .filter_map(|v| v.as_i64())
            .collect();
        assert_eq!(ids, vec![1, 3, 4]);
    }

    #[test]
    fn test_logical_not_gate() {
        let data = mock_flow_data();
        let g = NotGate {
            gate1: HighLowGate::new("FL1_A").with_min(525.0),
        };
        let result = gate(&data, &g);
        let ids: Vec<i64> = result
            .column("id")
            .unwrap()
            .iter()
            .filter_map(|v| v.as_i64())
            .collect();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn test_kde_gate() {
        let data = mock_flow_data();
        let g = KdeGate::new(vec!["FSC_A".to_string(), "SSC_A".to_string()]);
        let result = gate(&data, &g);
        // With 5 data points and 0.65 fraction, should keep ~3 densest points
        let n = event_count(&result);
        assert!(n >= 2 && n <= 4, "KDE gate should keep 2-4 of 5 points, got {}", n);
        // The middle points should be densest - check that extremes are removed
        let fsc = result.column_f64("FSC_A").unwrap();
        assert!(!fsc.contains(&1.0) || !fsc.contains(&5.0),
            "KDE should remove at least one extreme point");
    }

    #[test]
    fn test_data_not_modified() {
        let data = mock_flow_data();
        let original = data.clone();
        let g = HighLowGate::new("FL1_A").with_min(515.0);
        let _result = gate(&data, &g);
        assert_eq!(data, original);
    }
}
