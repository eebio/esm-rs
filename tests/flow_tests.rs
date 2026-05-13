//! Port of `test/flow_tests.jl`.

use esm::dataframe::{DataFrame, Value};
use esm::flow::{
    AndGate, EllipseGate, GatingMethod, HighLowGate, KdeGate, NotGate, OrGate, PolygonGate,
    QuadrantGate, RectangleGate, event_count, gate, gated_proportion,
    gated_proportion_from_gate,
};

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

fn col(df: &DataFrame, name: &str) -> Vec<f64> {
    df.column_f64(name).unwrap()
}

fn ids(df: &DataFrame) -> Vec<i64> {
    df.column("id")
        .unwrap()
        .iter()
        .filter_map(|v| v.as_i64())
        .collect()
}

#[test]
fn manual_gating() {
    let data = mock_flow_data();
    let original = data.clone();

    assert_eq!(
        col(&gate(&data, &HighLowGate::new("FL1_A").with_min(515.0)), "FL1_A"),
        vec![520.0, 530.0, 540.0, 550.0]
    );
    assert_eq!(
        col(&gate(&data, &HighLowGate::new("SSC_A").with_max(301.0)), "SSC_A"),
        vec![100.0, 200.0, 300.0]
    );
    // Upper bounds exclusive, lower bounds inclusive
    assert_eq!(
        col(
            &gate(&data, &HighLowGate::new("FSC_A").with_range(2.0, 4.0)),
            "FL1_A"
        ),
        vec![520.0, 530.0]
    );
    // No limits => unchanged
    assert_eq!(
        col(&gate(&data, &HighLowGate::new("FSC_A")), "FL2_A"),
        vec![51.0, 52.0, 53.0, 54.0, 55.0]
    );

    assert_eq!(
        col(
            &gate(
                &data,
                &RectangleGate {
                    channel_x: "FSC_A".into(),
                    channel_y: "SSC_A".into(),
                    x_min: 2.0,
                    x_max: 4.5,
                    y_min: 50.0,
                    y_max: 301.0,
                }
            ),
            "FSC_A"
        ),
        vec![2.0, 3.0]
    );
    assert_eq!(
        col(
            &gate(
                &data,
                &RectangleGate {
                    channel_x: "FL1_A".into(),
                    channel_y: "FL2_A".into(),
                    x_min: 0.0,
                    x_max: 545.0,
                    y_min: 52.0,
                    y_max: 54.0,
                }
            ),
            "SSC_A"
        ),
        vec![200.0, 300.0]
    );

    let q = |quadrant| QuadrantGate {
        channel_x: "FL1_A".into(),
        channel_y: "FL2_A".into(),
        x_cutoff: 535.0,
        y_cutoff: 54.5,
        quadrant,
    };
    assert_eq!(col(&gate(&data, &q(1)), "FL1_A"), vec![550.0]);
    assert_eq!(col(&gate(&data, &q(2)), "FL1_A"), vec![540.0]);
    assert_eq!(col(&gate(&data, &q(3)), "SSC_A"), vec![100.0, 200.0, 300.0]);
    assert!(col(&gate(&data, &q(4)), "FL1_A").is_empty());

    let polygon = vec![
        (1.5, 150.0),
        (3.0, 100.0),
        (4.5, 150.0),
        (4.0, 450.0),
        (2.0, 400.0),
    ];
    assert_eq!(
        col(
            &gate(
                &data,
                &PolygonGate {
                    channel_x: "FSC_A".into(),
                    channel_y: "SSC_A".into(),
                    points: polygon,
                }
            ),
            "FSC_A"
        ),
        vec![2.0, 3.0, 4.0]
    );

    let edge_poly = vec![
        (2.0, 200.0),
        (5.0, 200.0),
        (5.0, 600.0),
        (3.5, 500.0),
        (3.2, 250.0),
    ];
    assert_eq!(
        col(
            &gate(
                &data,
                &PolygonGate {
                    channel_x: "FSC_A".into(),
                    channel_y: "SSC_A".into(),
                    points: edge_poly,
                }
            ),
            "FSC_A"
        ),
        vec![2.0, 4.0, 5.0]
    );

    let eg = EllipseGate::from_points(
        "FSC_A",
        "SSC_A",
        Some((3.0, 300.0)),
        &[(1.9, 200.0), (4.0, 450.0), (3.0, 500.0)],
    );
    assert_eq!(col(&gate(&data, &eg), "FSC_A"), vec![2.0, 3.0, 4.0]);

    let eg = EllipseGate::from_points(
        "FSC_A",
        "SSC_A",
        None,
        &[
            (1.9, 200.0),
            (4.0, 450.0),
            (3.0, 500.0),
            (2.5, 400.0),
            (2.5, 100.0),
        ],
    );
    assert_eq!(col(&gate(&data, &eg), "SSC_A"), vec![200.0, 300.0]);

    // Original data is untouched.
    assert_eq!(data, original);
}

#[test]
#[should_panic(expected = "Quadrant must be between")]
fn manual_gating_invalid_quadrant() {
    let data = mock_flow_data();
    let q = QuadrantGate {
        channel_x: "FL1_A".into(),
        channel_y: "FL2_A".into(),
        x_cutoff: 535.0,
        y_cutoff: 54.5,
        quadrant: 5,
    };
    gate(&data, &q);
}

#[test]
fn event_counting() {
    let data = mock_flow_data();
    assert_eq!(event_count(&data), 5);

    let gated = gate(&data, &HighLowGate::new("FL1_A").with_min(530.0));
    assert_eq!(event_count(&gated), 3);

    let gated2 = gate(
        &data,
        &RectangleGate {
            channel_x: "FSC_A".into(),
            channel_y: "SSC_A".into(),
            x_min: 2.0,
            x_max: 4.5,
            y_min: 50.0,
            y_max: 301.0,
        },
    );
    assert_eq!(event_count(&gated2), 2);
}

#[test]
fn gated_proportions() {
    let data = mock_flow_data();
    let original = data.clone();

    let gated = gate(&data, &HighLowGate::new("FL1_A").with_min(530.0));
    assert_eq!(gated_proportion(&data, &gated), 3.0 / 5.0);
    assert_eq!(
        gated_proportion_from_gate(&data, &HighLowGate::new("FL1_A").with_min(530.0)),
        3.0 / 5.0
    );

    let gated2 = gate(
        &data,
        &RectangleGate {
            channel_x: "FSC_A".into(),
            channel_y: "SSC_A".into(),
            x_min: 2.0,
            x_max: 4.5,
            y_min: 50.0,
            y_max: 301.0,
        },
    );
    assert_eq!(gated_proportion(&data, &gated2), 2.0 / 5.0);
    assert_eq!(
        gated_proportion_from_gate(
            &data,
            &RectangleGate {
                channel_x: "FSC_A".into(),
                channel_y: "SSC_A".into(),
                x_min: 2.0,
                x_max: 4.5,
                y_min: 50.0,
                y_max: 301.0,
            }
        ),
        2.0 / 5.0
    );

    assert_eq!(data, original);
}

#[test]
fn autogating_matches_high_low() {
    let data = mock_flow_data();
    let original = data.clone();

    let kde = KdeGate::new(vec!["FSC_A".into(), "SSC_A".into()]);
    let hl = HighLowGate::new("FSC_A").with_range(1.5, 4.5);
    assert_eq!(gate(&data, &kde), gate(&data, &hl));

    assert_eq!(data, original);
}

#[test]
fn logical_gates() {
    let data = mock_flow_data();
    let original = data.clone();

    let and = AndGate {
        gate1: HighLowGate::new("FL1_A").with_min(525.0),
        gate2: HighLowGate::new("SSC_A").with_max(450.0),
    };
    assert_eq!(ids(&gate(&data, &and)), vec![3, 4]);

    let or = OrGate {
        gate1: HighLowGate::new("FL1_A").with_range(525.0, 545.0),
        gate2: HighLowGate::new("SSC_A").with_max(150.0),
    };
    assert_eq!(ids(&gate(&data, &or)), vec![1, 3, 4]);

    let not = NotGate {
        gate1: HighLowGate::new("FL1_A").with_min(525.0),
    };
    assert_eq!(ids(&gate(&data, &not)), vec![1, 2]);

    // and/or/not have the same effect via the same constructors.
    let and2 = AndGate {
        gate1: HighLowGate::new("FL1_A").with_min(525.0),
        gate2: HighLowGate::new("SSC_A").with_max(450.0),
    };
    assert_eq!(
        col(&gate(&data, &and2), "FL1_A"),
        vec![530.0, 540.0]
    );

    // KDE combined with a HighLowGate.
    let combined = AndGate {
        gate1: KdeGate::new(vec!["FSC_A".into(), "SSC_A".into()]),
        gate2: HighLowGate::new("FSC_A").with_max(3.5),
    };
    assert_eq!(col(&gate(&data, &combined), "FL1_A"), vec![520.0, 530.0]);

    assert_eq!(data, original);
}
