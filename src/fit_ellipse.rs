/// Fit an ellipse to 2D data using ordinary least squares.
///
/// Returns (a, b, theta, x0, y0, p) where:
/// - a, b: semi-major and semi-minor axis lengths
/// - theta: rotation angle in radians
/// - x0, y0: center coordinates
/// - p: parameter vector for quadratic form
///
/// The parametric form is:
/// x(t) = cos(θ)*a*cos(t) - sin(θ)*b*sin(t) + x₀
/// y(t) = sin(θ)*a*cos(t) + cos(θ)*b*sin(t) + y₀
pub fn fit_ellipse(x: &[f64], y: &[f64]) -> (f64, f64, f64, f64, f64, Vec<f64>) {
    assert_eq!(x.len(), y.len());
    let n = x.len();

    // Design matrix: [x^2, x*y, y^2, x, y]
    // Solve M*p = ones(n) using least squares (QR decomposition)
    let m = nalgebra::DMatrix::from_fn(n, 5, |i, j| match j {
        0 => x[i] * x[i],
        1 => x[i] * y[i],
        2 => y[i] * y[i],
        3 => x[i],
        4 => y[i],
        _ => unreachable!(),
    });

    let ones = nalgebra::DVector::from_element(n, 1.0);

    // Solve using SVD (least squares for overdetermined system)
    let svd = m.svd(true, true);
    let p = svd.solve(&ones, 1e-12).expect("SVD solve failed");

    let a_coef = p[0];
    let b_coef = p[1];
    let c_coef = p[2];
    let d_coef = p[3];
    let e_coef = p[4];
    let f_coef = -1.0;

    // Calculate parametric form parameters
    let delta = b_coef * b_coef - 4.0 * a_coef * c_coef;
    let lambda = (a_coef - c_coef) * (a_coef - c_coef) + b_coef * b_coef;

    let inner_val = 2.0
        * (a_coef * e_coef * e_coef + c_coef * d_coef * d_coef
            - b_coef * d_coef * e_coef
            + delta * f_coef);

    let a_axis = (-(inner_val * ((a_coef + c_coef) + lambda.sqrt())).max(0.0).sqrt())
        / (b_coef * b_coef - 4.0 * a_coef * c_coef);
    let b_axis = (-(inner_val * ((a_coef + c_coef) - lambda.sqrt())).max(0.0).sqrt())
        / (b_coef * b_coef - 4.0 * a_coef * c_coef);

    let theta = ((c_coef - a_coef - lambda.sqrt()) / b_coef).atan();

    let x0 = (2.0 * c_coef * d_coef - b_coef * e_coef) / delta;
    let y0 = (2.0 * a_coef * e_coef - b_coef * d_coef) / delta;

    let params = vec![a_coef, b_coef, c_coef, d_coef, e_coef];

    (a_axis, b_axis, theta, x0, y0, params)
}

/// Generate points along an ellipse from parametric form.
pub fn ellipse_from_parametric(
    a: f64,
    b: f64,
    theta: f64,
    x0: f64,
    y0: f64,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let cos_t = theta.cos();
    let sin_t = theta.sin();

    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);

    for i in 0..n {
        let t = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
        // Use prevfloat(2pi) equivalent - just don't quite reach 2pi
        let t = if i == n - 1 {
            2.0 * std::f64::consts::PI * (1.0 - f64::EPSILON)
        } else {
            t
        };
        xs.push(cos_t * a * t.cos() - sin_t * b * t.sin() + x0);
        ys.push(sin_t * a * t.cos() + cos_t * b * t.sin() + y0);
    }

    (xs, ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_fit(theta: f64, a: f64, b: f64, x0: f64, y0: f64, n: usize, noise: f64, tol: f64) {
        assert!(a > b);
        let (x, y) = ellipse_from_parametric(a, b, theta, x0, y0, n);

        // Add noise (deterministic pseudo-random for reproducibility)
        let xn: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| v + noise * pseudo_randn(i as u64))
            .collect();
        let yn: Vec<f64> = y
            .iter()
            .enumerate()
            .map(|(i, &v)| v + noise * pseudo_randn(i as u64 + n as u64))
            .collect();

        let (af, bf, _tf, x0f, y0f, _) = fit_ellipse(&xn, &yn);

        // The fit may swap a and b
        let (af, bf) = if bf > af { (bf, af) } else { (af, bf) };

        assert!(
            (af - a).abs() < tol * a.max(1.0),
            "a: {} vs {} (tol {})",
            af,
            a,
            tol
        );
        assert!(
            (bf - b).abs() < tol * b.max(1.0),
            "b: {} vs {} (tol {})",
            bf,
            b,
            tol
        );
        assert!(
            (x0f - x0).abs() < tol * x0.abs().max(1.0),
            "x0: {} vs {}",
            x0f,
            x0
        );
        assert!(
            (y0f - y0).abs() < tol * y0.abs().max(1.0),
            "y0: {} vs {}",
            y0f,
            y0
        );
    }

    /// Simple deterministic pseudo-random normal-ish values.
    fn pseudo_randn(seed: u64) -> f64 {
        // Simple hash-based approach
        let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (x as f64) / (u64::MAX as f64);
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (x as f64) / (u64::MAX as f64);
        // Box-Muller transform
        let u1 = u1.max(1e-10);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    #[test]
    fn test_fit_ellipse_default() {
        test_fit(
            std::f64::consts::FRAC_PI_3,
            3.0,
            1.5,
            3.0,
            -1.0,
            10000,
            0.001,
            0.1,
        );
    }

    #[test]
    fn test_fit_ellipse_a2() {
        test_fit(
            std::f64::consts::FRAC_PI_3,
            2.0,
            1.5,
            3.0,
            -1.0,
            10000,
            0.001,
            0.1,
        );
    }

    #[test]
    fn test_fit_ellipse_theta0() {
        test_fit(0.0, 3.0, 1.5, 3.0, -1.0, 10000, 0.001, 0.1);
    }

    #[test]
    fn test_fit_ellipse_noisy() {
        test_fit(
            std::f64::consts::FRAC_PI_2,
            3.0,
            1.5,
            3.0,
            -1.0,
            10000,
            0.1,
            0.1,
        );
    }
}
