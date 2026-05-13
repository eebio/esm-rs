//! Port of `test/FitEllipse_tests.jl`.
//!
//! For each rotation/axis/centre combination, sample N points on the ellipse,
//! add small Gaussian noise, fit, and check the recovered parameters against
//! tolerance ε. The RNG is seeded so failures are reproducible.

use esm::fit_ellipse::{ellipse_from_parametric, fit_ellipse};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

const SEED: u64 = 0xE5C0_FEEE_5C0F_FEEE;

fn run(theta: f64, a: f64, b: f64, x0: f64, y0: f64, n: usize, eps: f64, xi: f64) {
    assert!(a > b, "test requires a > b");
    let (xs, ys) = ellipse_from_parametric(a, b, theta, x0, y0, n);
    let mut rng = StdRng::seed_from_u64(SEED);
    let normal = Normal::new(0.0, xi).unwrap();
    let xs_noisy: Vec<f64> = xs.iter().map(|x| x + normal.sample(&mut rng)).collect();
    let ys_noisy: Vec<f64> = ys.iter().map(|y| y + normal.sample(&mut rng)).collect();

    let (mut af, mut bf, _, x0f, y0f, _) = fit_ellipse(&xs_noisy, &ys_noisy);
    if bf > af {
        std::mem::swap(&mut af, &mut bf);
    }

    approx_eq("a", af, a, eps);
    approx_eq("b", bf, b, eps);
    approx_eq("x0", x0f, x0, eps);
    approx_eq("y0", y0f, y0, eps);
}

fn approx_eq(name: &str, got: f64, expected: f64, eps: f64) {
    let abs = (got - expected).abs();
    let rel = abs / expected.abs().max(f64::EPSILON);
    assert!(
        abs <= eps || rel <= eps,
        "{} = {} not within ε={} of {} (abs={}, rel={})",
        name,
        got,
        eps,
        expected,
        abs,
        rel
    );
}

#[test]
fn fit_default() {
    let pi = std::f64::consts::PI;
    run(pi / 3.0, 3.0, 1.5, 3.0, -1.0, 10_000, 0.1, 0.001);
}

#[test]
fn fit_a_two() {
    let pi = std::f64::consts::PI;
    run(pi / 3.0, 2.0, 1.5, 3.0, -1.0, 10_000, 0.1, 0.001);
}

#[test]
fn fit_theta_zero() {
    run(0.0, 3.0, 1.5, 3.0, -1.0, 10_000, 0.1, 0.001);
}

#[test]
fn fit_theta_half_pi_noisy() {
    let pi = std::f64::consts::PI;
    run(pi / 2.0, 3.0, 1.5, 3.0, -1.0, 10_000, 0.1, 0.1);
}

#[test]
fn fit_theta_five_sixths_pi() {
    let pi = std::f64::consts::PI;
    run(5.0 * pi / 6.0, 3.0, 1.5, 3.0, -1.0, 10_000, 0.1, 0.001);
}
