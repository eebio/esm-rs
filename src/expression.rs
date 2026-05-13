//! Expression evaluation engine using rhai.
//!
//! This module provides a simplified expression language for ESM transformations.
//! Julia-like expressions are translated to rhai-compatible syntax and evaluated
//! with access to ESM data (groups, samples, transformations).

use indexmap::IndexMap;
use rhai::{Dynamic, Engine};

use crate::dataframe::DataFrame;
use crate::esm_files;
use crate::esm_types::EsmZones;

/// The expression evaluation context.
pub struct ExpressionEngine {
    _engine: Engine,
}

/// Result of evaluating an expression.
#[derive(Debug, Clone)]
pub enum ExprResult {
    /// A DataFrame result.
    Frame(DataFrame),
    /// A scalar numeric result.
    Number(f64),
    /// A matrix (Vec of Vec).
    Matrix(Vec<Vec<f64>>),
    /// A symbol that couldn't be resolved.
    Symbol(String),
    /// An integer result.
    Int(i64),
}

impl ExprResult {
    pub fn as_number(&self) -> Option<f64> {
        match self {
            ExprResult::Number(n) => Some(*n),
            ExprResult::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_frame(&self) -> Option<&DataFrame> {
        match self {
            ExprResult::Frame(df) => Some(df),
            _ => None,
        }
    }
}

impl ExpressionEngine {
    pub fn new() -> Self {
        let engine = Engine::new();
        ExpressionEngine { _engine: engine }
    }
}

impl Default for ExpressionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluate an ESM transformation expression.
///
/// This is the Rust equivalent of Julia's `sexp_to_nested_list` + `eval`.
/// It handles:
/// - Numeric literals
/// - String literals
/// - Symbol resolution (transformations, groups, samples)
/// - Channel access via dot notation (e.g., `plate_01.FL1_A`)
/// - Function calls (sum, hcat, colmean, etc.)
/// - Array/matrix literals
pub fn evaluate_expression(
    equation: &str,
    es: &EsmZones,
    trans_map: &IndexMap<String, String>,
) -> Result<ExprResult, String> {
    let trimmed = equation.trim();

    // Try as a pure number first
    if let Ok(n) = trimmed.parse::<i64>() {
        return Ok(ExprResult::Int(n));
    }
    if let Ok(n) = trimmed.parse::<f64>() {
        return Ok(ExprResult::Number(n));
    }

    // Check if it's a string literal
    if trimmed.starts_with('"') && trimmed.ends_with('"') {
        return Ok(ExprResult::Symbol(trimmed[1..trimmed.len() - 1].to_string()));
    }

    // Check if it's a simple symbol (transformation, group, or sample)
    if is_simple_identifier(trimmed) {
        return resolve_symbol(trimmed, es, trans_map);
    }

    // Check for dot notation: symbol.channel
    if let Some((base, channel)) = trimmed.split_once('.') {
        if is_simple_identifier(base) && is_simple_identifier(channel) {
            let base_result = resolve_symbol(base, es, trans_map)?;
            if let ExprResult::Frame(df) = &base_result {
                // Check if channel exists
                let has_channel = df.names().iter().any(|n| {
                    n.split('.').last().map_or(false, |c| c == channel)
                        || n.split('.').any(|part| part == channel)
                });
                if has_channel {
                    let filtered = esm_files::filter_channel(df, channel);
                    return Ok(ExprResult::Frame(filtered));
                }
            }
            // If it wasn't a channel access, try the full name as a sample
            return resolve_symbol(trimmed, es, trans_map);
        }
    }

    // Try to evaluate as a rhai expression
    // First, pre-process Julia-like syntax to rhai
    let rhai_expr = julia_to_rhai(trimmed);

    // For simple function calls like sum([1,2,3,4])
    if rhai_expr.starts_with("sum(") {
        if let Some(inner) = extract_array_arg(&rhai_expr, "sum") {
            let sum: f64 = inner.iter().sum();
            return Ok(ExprResult::Number(sum));
        }
    }

    // Matrix literals like [1 2 3; 4 5 6]
    if trimmed.starts_with('[') && trimmed.ends_with(']') && trimmed.contains(';') {
        if let Some(matrix) = parse_matrix_literal(trimmed) {
            return Ok(ExprResult::Matrix(matrix));
        }
    }

    // Complex expressions with function calls on data
    evaluate_complex_expression(trimmed, es, trans_map)
}

/// Check if a string is a simple identifier (letters, digits, underscores).
fn is_simple_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_alphanumeric() || c == '_')
        && !s.chars().next().unwrap().is_ascii_digit()
}

/// Resolve a symbol to ESM data.
fn resolve_symbol(
    name: &str,
    es: &EsmZones,
    trans_map: &IndexMap<String, String>,
) -> Result<ExprResult, String> {
    // Is it a transformation?
    if let Some(equation) = trans_map.get(name) {
        return evaluate_expression(equation, es, trans_map);
    }

    // Is it a group?
    let group_names = esm_files::group_names(es);
    if group_names.contains(&name.to_string()) {
        let df = esm_files::get_group(es, name);
        return Ok(ExprResult::Frame(df));
    }

    // Is it a sample?
    let sample_names = esm_files::sample_names(es);
    let sample_ids: Vec<String> = sample_names
        .iter()
        .map(|n| esm_files::get_sample_id(n).to_string())
        .collect();
    if sample_ids.contains(&name.to_string()) {
        let df = esm_files::get_sample(es, name);
        return Ok(ExprResult::Frame(df));
    }

    // Just return it as a symbol
    Ok(ExprResult::Symbol(name.to_string()))
}

/// Translate Julia-like syntax to rhai.
fn julia_to_rhai(expr: &str) -> String {
    let mut result = expr.to_string();
    // Replace .+ .* .- ./ with regular operators (element-wise in Julia)
    result = result.replace(".-", "-");
    result = result.replace(".+", "+");
    result = result.replace(".*", "*");
    result = result.replace("./", "/");
    result
}

/// Extract array argument from a function call like sum([1,2,3]).
fn extract_array_arg(expr: &str, func_name: &str) -> Option<Vec<f64>> {
    let prefix = format!("{}([", func_name);
    let suffix = "])";
    if expr.starts_with(&prefix) && expr.ends_with(suffix) {
        let inner = &expr[prefix.len()..expr.len() - suffix.len()];
        let values: Result<Vec<f64>, _> = inner.split(',').map(|s| s.trim().parse()).collect();
        return values.ok();
    }
    None
}

/// Parse a Julia matrix literal like [1 2 3; 4 5 6].
fn parse_matrix_literal(expr: &str) -> Option<Vec<Vec<f64>>> {
    let inner = expr.trim_start_matches('[').trim_end_matches(']');
    let rows: Vec<&str> = inner.split(';').collect();
    let mut matrix = Vec::new();
    for row in rows {
        let vals: Result<Vec<f64>, _> = row.split_whitespace().map(|s| s.trim().parse()).collect();
        matrix.push(vals.ok()?);
    }
    Some(matrix)
}

/// Evaluate complex expressions involving data operations.
fn evaluate_complex_expression(
    expr: &str,
    es: &EsmZones,
    trans_map: &IndexMap<String, String>,
) -> Result<ExprResult, String> {
    // 1. Top-level element-wise binary operator `.-` (lowest precedence we care about).
    //    Check this before function-call patterns so that `hcat(...).-colmean(...)`
    //    is split correctly rather than swallowed by the `hcat(` prefix.
    if let Some(pos) = find_top_level_op(expr, ".-") {
        let (left, right) = expr.split_at(pos);
        let right = &right[".-".len()..];
        let left_val = evaluate_expression(left.trim(), es, trans_map)?;
        let right_val = evaluate_expression(right.trim(), es, trans_map)?;
        if let (ExprResult::Frame(left_df), ExprResult::Frame(right_df)) =
            (left_val, right_val)
        {
            // If the right-hand side is a single column (e.g. the result of
            // `colmean(...)`), Julia broadcasts it against every column of the
            // left-hand side.
            if right_df.ncol() == 1 {
                let first = right_df.names().first().map(|s| s.to_string()).unwrap_or_default();
                let v = right_df.column_f64(&first).unwrap_or_default();
                return Ok(ExprResult::Frame(left_df.sub_vec(&v)));
            }
            return Ok(ExprResult::Frame(left_df.sub_df(&right_df)));
        }
        return Err("'.-' operands did not resolve to DataFrames".to_string());
    }

    // 2. hcat(a, b, ...) - horizontal concatenation. Require matched parens
    //    so combined forms like `hcat(...).-colmean(...)` are not misparsed.
    if let Some(inner) = matched_call_body("hcat", expr) {
        let args = split_args(inner);
        let mut result: Option<DataFrame> = None;
        for arg in &args {
            let val = evaluate_expression(arg.trim(), es, trans_map)?;
            match val {
                ExprResult::Frame(df) => {
                    result = Some(match result {
                        Some(existing) => existing.hcat(&df),
                        None => df,
                    });
                }
                _ => return Err(format!("hcat argument '{}' did not resolve to a DataFrame", arg)),
            }
        }
        return Ok(ExprResult::Frame(result.unwrap_or_default()));
    }

    // 3. colmean(expr) - column mean.
    if let Some(inner) = matched_call_body("colmean", expr) {
        let val = evaluate_expression(inner.trim(), es, trans_map)?;
        if let ExprResult::Frame(df) = val {
            let mean = df.colmean();
            let mut result = DataFrame::new();
            let first_name = df.names().first().map(|s| s.to_string()).unwrap_or_default();
            result.insert_f64_column(first_name, mean);
            return Ok(ExprResult::Frame(result));
        }
    }

    // 4. Fallback: try rhai for simple arithmetic.
    let engine = Engine::new();
    match engine.eval::<Dynamic>(expr) {
        Ok(val) => {
            if let Some(n) = val.as_float().ok() {
                Ok(ExprResult::Number(n))
            } else if let Some(n) = val.as_int().ok() {
                Ok(ExprResult::Int(n))
            } else {
                Ok(ExprResult::Symbol(format!("{:?}", val)))
            }
        }
        Err(_) => Ok(ExprResult::Symbol(expr.to_string())),
    }
}

/// Locate a depth-0 occurrence of `op` in `expr`, ignoring matches inside
/// parentheses or square brackets.
fn find_top_level_op(expr: &str, op: &str) -> Option<usize> {
    let bytes = expr.as_bytes();
    let op_bytes = op.as_bytes();
    let mut depth: i32 = 0;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'(' || c == b'[' {
            depth += 1;
        } else if c == b')' || c == b']' {
            depth -= 1;
        } else if depth == 0
            && i + op_bytes.len() <= bytes.len()
            && bytes[i..i + op_bytes.len()] == *op_bytes
        {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// If `expr` is exactly `name(...)` with the trailing `)` matching the
/// opening `(` after `name`, return the inner body. Otherwise None.
fn matched_call_body<'a>(name: &str, expr: &'a str) -> Option<&'a str> {
    let prefix_len = name.len() + 1; // include `(`
    if !expr.starts_with(name) || !expr.ends_with(')') || expr.len() <= prefix_len {
        return None;
    }
    if expr.as_bytes()[name.len()] != b'(' {
        return None;
    }
    let body = &expr[prefix_len..expr.len() - 1];
    let mut depth: i32 = 0;
    for c in body.chars() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    if depth == 0 { Some(body) } else { None }
}

/// Split function arguments, respecting nested parentheses and brackets.
fn split_args(s: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in s.chars() {
        match ch {
            '(' | '[' => {
                depth += 1;
                current.push(ch);
            }
            ')' | ']' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                args.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    if !current.trim().is_empty() {
        args.push(current.trim().to_string());
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_number_literal() {
        let es = empty_esm();
        let tm = IndexMap::new();
        let result = evaluate_expression("5", &es, &tm).unwrap();
        assert_eq!(result.as_number().unwrap(), 5.0);
    }

    #[test]
    fn test_float_literal() {
        let es = empty_esm();
        let tm = IndexMap::new();
        let result = evaluate_expression("3.14", &es, &tm).unwrap();
        assert!((result.as_number().unwrap() - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_sum_array() {
        let es = empty_esm();
        let tm = IndexMap::new();
        let result = evaluate_expression("sum([1,2,3,4])", &es, &tm).unwrap();
        assert_eq!(result.as_number().unwrap(), 10.0);
    }

    #[test]
    fn test_matrix_literal() {
        let es = empty_esm();
        let tm = IndexMap::new();
        let result = evaluate_expression("[1 2 3; 4 5 6]", &es, &tm).unwrap();
        if let ExprResult::Matrix(m) = result {
            assert_eq!(m, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        } else {
            panic!("Expected matrix result");
        }
    }

    #[test]
    fn test_transformation_resolution() {
        let es = empty_esm();
        let mut tm = IndexMap::new();
        tm.insert("extra_transform".to_string(), "sum([1,2,3,4])".to_string());
        let result = evaluate_expression("extra_transform", &es, &tm).unwrap();
        assert_eq!(result.as_number().unwrap(), 10.0);
    }

    #[test]
    fn test_simple_identifier() {
        assert!(is_simple_identifier("plate_01"));
        assert!(is_simple_identifier("FL1_A"));
        assert!(!is_simple_identifier("1abc"));
        assert!(!is_simple_identifier(""));
        assert!(!is_simple_identifier("a.b"));
    }

    fn empty_esm() -> EsmZones {
        EsmZones {
            samples: DataFrame::new(),
            groups: DataFrame::new(),
            transformations: IndexMap::new(),
            views: IndexMap::new(),
            metadata: IndexMap::new(),
        }
    }
}
