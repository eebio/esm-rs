use regex::Regex;

/// Format a channel name: replace non-alphanumeric chars with '_', strip leading/trailing '_',
/// collapse consecutive '_'.
pub fn format_channel(channel: &str) -> String {
    let re_non_alnum = Regex::new(r"[^a-zA-Z0-9]").unwrap();
    let re_multi_underscore = Regex::new(r"_+").unwrap();
    let replaced = re_non_alnum.replace_all(channel, "_");
    let collapsed = re_multi_underscore.replace_all(&replaced, "_");
    collapsed.trim_matches('_').to_string()
}

/// Expand a condensed group name into individual sample IDs.
///
/// e.g. "plate_01_a[1:3]" -> ["plate_01_a1", "plate_01_a2", "plate_01_a3"]
/// e.g. "plate_01_[a:c]1" -> ["plate_01_a1", "plate_01_b1", "plate_01_c1"]
/// e.g. "plate_01_a[1:3:10]" -> ["plate_01_a1", "plate_01_a4", "plate_01_a7", "plate_01_a10"]
/// Walk a Julia-style `start:step:stop` range, including negative steps, and
/// push each value (formatted by `fmt`) into `out`.
fn push_range<F: Fn(i64) -> String>(
    out: &mut Vec<String>,
    start: i64,
    step: i64,
    stop: i64,
    fmt: F,
) {
    if step == 0 {
        return;
    }
    let mut i = start;
    if step > 0 {
        while i <= stop {
            out.push(fmt(i));
            i += step;
        }
    } else {
        while i >= stop {
            out.push(fmt(i));
            i += step;
        }
    }
}

pub fn expand_group(group: &str) -> Vec<String> {
    let bracket_re = Regex::new(r"\[([^\]]+)\]").unwrap();

    // Parse each bracketed section
    let mut parts: Vec<Vec<String>> = Vec::new();
    for cap in bracket_re.captures_iter(group) {
        let s = &cap[1];
        let mut expanded: Vec<String> = Vec::new();
        for item in s.split(',') {
            let item = item.trim();
            let colon_count = item.chars().filter(|&c| c == ':').count();
            if colon_count > 0 {
                let segments: Vec<&str> = item.splitn(3, ':').collect();
                let (start_str, step_str, stop_str) = if colon_count == 2 {
                    (segments[0], segments[1], segments[2])
                } else {
                    (segments[0], "1", segments[1])
                };

                // Check if numeric or character range
                if start_str.chars().next().unwrap().is_ascii_digit()
                    || (start_str.len() > 1 && start_str.starts_with('-'))
                {
                    let start: i64 = start_str.parse().unwrap();
                    let step: i64 = step_str.parse().unwrap();
                    let stop: i64 = stop_str.parse().unwrap();
                    push_range(&mut expanded, start, step, stop, |n| n.to_string());
                } else {
                    let start = start_str.chars().next().unwrap() as i64;
                    let step: i64 = step_str.parse().unwrap();
                    let stop = stop_str.chars().next().unwrap() as i64;
                    push_range(&mut expanded, start, step, stop, |n| {
                        String::from(n as u8 as char)
                    });
                }
            } else {
                expanded.push(item.to_string());
            }
        }
        parts.push(expanded);
    }

    if parts.is_empty() {
        return vec![group.to_string()];
    }

    // Replace bracketed sections with "{}" for formatting
    let fmt = bracket_re.replace_all(group, "{}").to_string();

    // Generate cartesian product (leftmost dimension varies fastest, matching Julia's Iterators.product)
    let mut ids = Vec::new();
    let mut indices = vec![0usize; parts.len()];
    loop {
        let mut id = fmt.clone();
        for (i, idx) in indices.iter().enumerate() {
            id = id.replacen("{}", &parts[i][*idx], 1);
        }
        ids.push(id);

        // Increment indices (leftmost first to match Julia behavior)
        let mut carry = true;
        for i in 0..indices.len() {
            if carry {
                indices[i] += 1;
                if indices[i] >= parts[i].len() {
                    indices[i] = 0;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            break;
        }
    }

    ids
}

/// Expand a comma-separated list of group specs, where commas inside brackets are ignored.
pub fn expand_groups(groups: &str) -> Vec<String> {
    let bracket_re = Regex::new(r"[\[\]]").unwrap();
    let mut expanded = Vec::new();

    // Split on commas not inside brackets manually
    let wells = split_outside_brackets(groups);
    for well in wells {
        let well = well.trim();
        if bracket_re.is_match(well) {
            expanded.extend(expand_group(well));
        } else {
            expanded.push(well.to_string());
        }
    }
    expanded
}

/// Split a string on commas that are not inside square brackets.
fn split_outside_brackets(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut depth = 0;
    for ch in s.chars() {
        match ch {
            '[' => {
                depth += 1;
                current.push(ch);
            }
            ']' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                parts.push(current.clone());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// Compute index ranges where values in a slice fall between min and max (inclusive).
/// Returns (first_index, last_index) as 0-based options.
pub fn index_between_vals(data: &[f64], minv: f64, maxv: f64) -> (Option<usize>, Option<usize>) {
    let first = data.iter().position(|&x| x >= minv && x <= maxv);
    let last = data.iter().rposition(|&x| x >= minv && x <= maxv);
    (first, last)
}

/// DataFrame variant of [`index_between_vals`], mirroring Julia's
/// `index_between_vals(df; minv, maxv)`. The returned indices are 0-based.
pub fn index_between_vals_df(
    df: &crate::dataframe::DataFrame,
    minv: f64,
    maxv: f64,
) -> indexmap::IndexMap<String, (Option<usize>, Option<usize>)> {
    let mut out = indexmap::IndexMap::new();
    for name in df.names() {
        let col = df.column_f64(name).unwrap_or_default();
        out.insert(name.to_string(), index_between_vals(&col, minv, maxv));
    }
    out
}

/// Return the slice of `df` corresponding to the rows whose entry in the
/// first column of `time_col` lies between `mint` and `maxt` (in minutes).
/// Returns an empty DataFrame if no rows match.
pub fn between_times(
    df: &crate::dataframe::DataFrame,
    time_col: &crate::dataframe::DataFrame,
    mint: f64,
    maxt: f64,
) -> crate::dataframe::DataFrame {
    let first_col = time_col
        .names()
        .first()
        .map(|s| s.to_string())
        .expect("time_col has at least one column");
    let times = time_col.column_f64(&first_col).unwrap_or_default();
    let min_ms = mint * 60000.0;
    let max_ms = maxt * 60000.0;
    let (first, last) = index_between_vals(&times, min_ms, max_ms);
    match (first, last) {
        (Some(a), Some(b)) => df.slice(a, b + 1),
        _ => df.slice(0, 0),
    }
}

/// Return the row of `df` corresponding to the last sample at or before
/// `time_point` minutes, mirroring Julia's `at_time(df, time_col, t)`.
/// Returns an empty DataFrame if no sample is at or before that time.
pub fn at_time(
    df: &crate::dataframe::DataFrame,
    time_col: &crate::dataframe::DataFrame,
    time_point: f64,
) -> crate::dataframe::DataFrame {
    let first_col = time_col
        .names()
        .first()
        .map(|s| s.to_string())
        .expect("time_col has at least one column");
    let times = time_col.column_f64(&first_col).unwrap_or_default();
    let max_ms = time_point * 60000.0;
    let (_, last) = index_between_vals(&times, 0.0, max_ms);
    match last {
        Some(idx) => df.slice(idx, idx + 1),
        None => df.slice(0, 0),
    }
}

/// For each column of `od_df`, find the first row whose value equals
/// `target_od` and return the corresponding cell of `target_df`. Mirrors
/// Julia's `at_od(od_df, target_df, target_od)`. Columns where no row
/// matches receive `Value::Missing`. Columns where the target exceeds the
/// maximum recorded OD are skipped entirely.
pub fn at_od(
    od_df: &crate::dataframe::DataFrame,
    target_df: &crate::dataframe::DataFrame,
    target_od: f64,
) -> crate::dataframe::DataFrame {
    let mut result = crate::dataframe::DataFrame::new();
    for name in od_df.names() {
        let od_col = od_df.column_f64(name).unwrap_or_default();
        if od_col.is_empty() {
            continue;
        }
        let max_od = od_col
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |a, b| if b.is_nan() { a } else { a.max(b) });
        if target_od > max_od {
            continue;
        }
        let (first, _) = index_between_vals(&od_col, target_od, target_od);
        let cell = match first {
            Some(idx) => target_df
                .column(name)
                .and_then(|c| c.get(idx))
                .cloned()
                .unwrap_or(crate::dataframe::Value::Missing),
            None => crate::dataframe::Value::Missing,
        };
        result.insert_column(name.to_string(), vec![cell]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_channel() {
        assert_eq!(format_channel("FL1-A"), "FL1_A");
        assert_eq!(format_channel("SSC-H"), "SSC_H");
        assert_eq!(format_channel("__test__"), "test");
        assert_eq!(format_channel("a--b"), "a_b");
        assert_eq!(format_channel("hello world"), "hello_world");
    }

    #[test]
    fn test_expand_group() {
        assert_eq!(
            expand_group("plate_01_a[1:3]"),
            vec!["plate_01_a1", "plate_01_a2", "plate_01_a3"]
        );
        assert_eq!(
            expand_group("plate_01_a[2:4]"),
            vec!["plate_01_a2", "plate_01_a3", "plate_01_a4"]
        );
        assert_eq!(
            expand_group("plate_01_[a:d]1"),
            vec!["plate_01_a1", "plate_01_b1", "plate_01_c1", "plate_01_d1"]
        );
        assert_eq!(
            expand_group("plate_01_[a:c]1[2:3]"),
            vec![
                "plate_01_a12",
                "plate_01_b12",
                "plate_01_c12",
                "plate_01_a13",
                "plate_01_b13",
                "plate_01_c13"
            ]
        );
        assert_eq!(
            expand_group("plate_01_[a:2:e]1"),
            vec!["plate_01_a1", "plate_01_c1", "plate_01_e1"]
        );
        assert_eq!(
            expand_group("plate_01_a[1:3:10]"),
            vec!["plate_01_a1", "plate_01_a4", "plate_01_a7", "plate_01_a10"]
        );
    }

    #[test]
    fn test_expand_groups() {
        assert_eq!(
            expand_groups("plate_01_a[1:3],plate_01_b[2:4], plate_02_c5,plate_01_d[1,2]"),
            vec![
                "plate_01_a1",
                "plate_01_a2",
                "plate_01_a3",
                "plate_01_b2",
                "plate_01_b3",
                "plate_01_b4",
                "plate_02_c5",
                "plate_01_d1",
                "plate_01_d2"
            ]
        );
        assert_eq!(
            expand_groups("plate_01_a[1:3]"),
            vec!["plate_01_a1", "plate_01_a2", "plate_01_a3"]
        );
    }

    #[test]
    fn test_index_between_vals() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(index_between_vals(&data, 3.0, 8.0), (Some(2), Some(7)));
        assert_eq!(index_between_vals(&data, 5.0, 10.0), (Some(4), Some(9)));
        assert_eq!(
            index_between_vals(&data, f64::NEG_INFINITY, f64::INFINITY),
            (Some(0), Some(9))
        );
        assert_eq!(index_between_vals(&data, 11.0, 20.0), (None, None));
    }
}
