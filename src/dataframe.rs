use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

/// A single value in a DataFrame cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Float(f64),
    Int(i64),
    Str(String),
    Bool(bool),
    Missing,
}

impl Value {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(v) => Some(*v),
            Value::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(v) => Some(*v),
            Value::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn is_missing(&self) -> bool {
        matches!(self, Value::Missing)
    }

    pub fn is_nan(&self) -> bool {
        matches!(self, Value::Float(v) if v.is_nan())
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Float(v) => write!(f, "{v}"),
            Value::Int(v) => write!(f, "{v}"),
            Value::Str(s) => write!(f, "{s}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Missing => write!(f, "missing"),
        }
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Float(v)
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Int(v)
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Str(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Str(v.to_string())
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<Option<f64>> for Value {
    fn from(v: Option<f64>) -> Self {
        match v {
            Some(f) => Value::Float(f),
            None => Value::Missing,
        }
    }
}

/// Column-oriented DataFrame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFrame {
    pub columns: IndexMap<String, Vec<Value>>,
}

impl DataFrame {
    /// Create an empty DataFrame.
    pub fn new() -> Self {
        DataFrame {
            columns: IndexMap::new(),
        }
    }

    /// Create a DataFrame from an IndexMap of columns.
    pub fn from_columns(columns: IndexMap<String, Vec<Value>>) -> Self {
        DataFrame { columns }
    }

    /// Number of rows.
    pub fn nrow(&self) -> usize {
        self.columns.values().next().map_or(0, |c| c.len())
    }

    /// Number of columns.
    pub fn ncol(&self) -> usize {
        self.columns.len()
    }

    /// Get column names.
    pub fn names(&self) -> Vec<&str> {
        self.columns.keys().map(|s| s.as_str()).collect()
    }

    /// Check if column exists.
    pub fn has_column(&self, name: &str) -> bool {
        self.columns.contains_key(name)
    }

    /// Get a column by name.
    pub fn column(&self, name: &str) -> Option<&Vec<Value>> {
        self.columns.get(name)
    }

    /// Get a column as f64 values (NaN for non-float values).
    pub fn column_f64(&self, name: &str) -> Option<Vec<f64>> {
        self.columns.get(name).map(|col| {
            col.iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect()
        })
    }

    /// Add a column.
    pub fn insert_column(&mut self, name: impl Into<String>, data: Vec<Value>) {
        self.columns.insert(name.into(), data);
    }

    /// Add a column of f64 values.
    pub fn insert_f64_column(&mut self, name: impl Into<String>, data: Vec<f64>) {
        self.columns
            .insert(name.into(), data.into_iter().map(Value::Float).collect());
    }

    /// Add a column of i64 values.
    pub fn insert_i64_column(&mut self, name: impl Into<String>, data: Vec<i64>) {
        self.columns
            .insert(name.into(), data.into_iter().map(Value::Int).collect());
    }

    /// Select specific columns by name, returning a new DataFrame.
    pub fn select(&self, names: &[&str]) -> DataFrame {
        let mut cols = IndexMap::new();
        for &name in names {
            if let Some(col) = self.columns.get(name) {
                cols.insert(name.to_string(), col.clone());
            }
        }
        DataFrame { columns: cols }
    }

    /// Select columns matching a predicate on their names.
    pub fn select_by<F>(&self, predicate: F) -> DataFrame
    where
        F: Fn(&str) -> bool,
    {
        let mut cols = IndexMap::new();
        for (name, col) in &self.columns {
            if predicate(name) {
                cols.insert(name.clone(), col.clone());
            }
        }
        DataFrame { columns: cols }
    }

    /// Filter rows by a boolean mask.
    pub fn filter(&self, mask: &[bool]) -> DataFrame {
        let mut cols = IndexMap::new();
        for (name, col) in &self.columns {
            let filtered: Vec<Value> = col
                .iter()
                .zip(mask.iter())
                .filter(|(_, m)| **m)
                .map(|(v, _)| v.clone())
                .collect();
            cols.insert(name.clone(), filtered);
        }
        DataFrame { columns: cols }
    }

    /// Get a single row as a Vec of Values.
    pub fn row(&self, idx: usize) -> Vec<(&str, &Value)> {
        self.columns
            .iter()
            .map(|(name, col)| (name.as_str(), &col[idx]))
            .collect()
    }

    /// Get rows by index range.
    pub fn slice(&self, start: usize, end: usize) -> DataFrame {
        let mut cols = IndexMap::new();
        for (name, col) in &self.columns {
            cols.insert(name.clone(), col[start..end].to_vec());
        }
        DataFrame { columns: cols }
    }

    /// Horizontal concatenation (add columns from another DataFrame).
    pub fn hcat(&self, other: &DataFrame) -> DataFrame {
        let mut cols = self.columns.clone();
        for (name, col) in &other.columns {
            cols.insert(name.clone(), col.clone());
        }
        DataFrame { columns: cols }
    }

    /// Horizontal concatenation with unique column names.
    pub fn hcat_unique(&self, other: &DataFrame) -> DataFrame {
        let mut cols = self.columns.clone();
        for (name, col) in &other.columns {
            let mut unique_name = name.clone();
            let mut counter = 1;
            while cols.contains_key(&unique_name) {
                unique_name = format!("{name}_{counter}");
                counter += 1;
            }
            cols.insert(unique_name, col.clone());
        }
        DataFrame { columns: cols }
    }

    /// Vertical concatenation (append rows from another DataFrame).
    pub fn vcat(&self, other: &DataFrame) -> DataFrame {
        let mut cols = IndexMap::new();
        for (name, col) in &self.columns {
            let mut new_col = col.clone();
            if let Some(other_col) = other.columns.get(name) {
                new_col.extend(other_col.iter().cloned());
            } else {
                new_col.extend(std::iter::repeat_n(Value::Missing, other.nrow()));
            }
            cols.insert(name.clone(), new_col);
        }
        DataFrame { columns: cols }
    }

    /// Sort columns by name alphabetically.
    pub fn sort_columns(&self) -> DataFrame {
        let mut names: Vec<&String> = self.columns.keys().collect();
        names.sort();
        let mut cols = IndexMap::new();
        for name in names {
            cols.insert(name.clone(), self.columns[name].clone());
        }
        DataFrame { columns: cols }
    }

    /// Rename a column.
    pub fn rename(&mut self, old_name: &str, new_name: &str) {
        if let Some(idx) = self.columns.get_index_of(old_name) {
            let (_, col) = self.columns.swap_remove_index(idx).unwrap();
            self.columns.insert(new_name.to_string(), col);
            // Move the new column to the original position
            let new_idx = self.columns.get_index_of(new_name).unwrap();
            self.columns.move_index(new_idx, idx);
        }
    }

    /// Remove columns where all values are Missing.
    pub fn remove_all_missing_columns(&self) -> DataFrame {
        let mut cols = IndexMap::new();
        for (name, col) in &self.columns {
            if !col.iter().all(|v| v.is_missing()) {
                cols.insert(name.clone(), col.clone());
            }
        }
        DataFrame { columns: cols }
    }

    /// Element-wise subtraction: self - scalar.
    pub fn sub_scalar(&self, scalar: f64) -> DataFrame {
        self.apply_scalar(scalar, |a, b| a - b)
    }

    /// Element-wise addition: self + scalar.
    pub fn add_scalar(&self, scalar: f64) -> DataFrame {
        self.apply_scalar(scalar, |a, b| a + b)
    }

    /// Element-wise division by scalar.
    pub fn div_scalar(&self, scalar: f64) -> DataFrame {
        self.apply_scalar(scalar, |a, b| a / b)
    }

    /// Element-wise multiplication by scalar.
    pub fn mul_scalar(&self, scalar: f64) -> DataFrame {
        self.apply_scalar(scalar, |a, b| a * b)
    }

    fn apply_scalar<F>(&self, scalar: f64, op: F) -> DataFrame
    where
        F: Fn(f64, f64) -> f64,
    {
        let mut cols = IndexMap::new();
        for (name, col) in &self.columns {
            let new_col: Vec<Value> = col
                .iter()
                .map(|v| match v.as_f64() {
                    Some(f) => Value::Float(op(f, scalar)),
                    None => v.clone(),
                })
                .collect();
            cols.insert(name.clone(), new_col);
        }
        DataFrame { columns: cols }
    }

    /// Element-wise subtraction: self - other (column by column, matching names).
    pub fn sub_df(&self, other: &DataFrame) -> DataFrame {
        self.apply_df(other, |a, b| a - b)
    }

    /// Element-wise addition: self + other.
    pub fn add_df(&self, other: &DataFrame) -> DataFrame {
        self.apply_df(other, |a, b| a + b)
    }

    fn apply_df<F>(&self, other: &DataFrame, op: F) -> DataFrame
    where
        F: Fn(f64, f64) -> f64,
    {
        let mut cols = IndexMap::new();
        for (name, col) in &self.columns {
            if let Some(other_col) = other.columns.get(name) {
                let new_col: Vec<Value> = col
                    .iter()
                    .zip(other_col.iter())
                    .map(|(a, b)| match (a.as_f64(), b.as_f64()) {
                        (Some(fa), Some(fb)) => Value::Float(op(fa, fb)),
                        _ => Value::Missing,
                    })
                    .collect();
                cols.insert(name.clone(), new_col);
            } else {
                cols.insert(name.clone(), col.clone());
            }
        }
        DataFrame { columns: cols }
    }

    /// Subtract a vector element-wise from all columns.
    pub fn sub_vec(&self, vec: &[f64]) -> DataFrame {
        let mut cols = IndexMap::new();
        for (name, col) in &self.columns {
            let new_col: Vec<Value> = col
                .iter()
                .zip(vec.iter())
                .map(|(v, s)| match v.as_f64() {
                    Some(f) => Value::Float(f - s),
                    None => v.clone(),
                })
                .collect();
            cols.insert(name.clone(), new_col);
        }
        DataFrame { columns: cols }
    }

    /// Compute column-wise mean across all columns, returning a Vec<f64>.
    pub fn colmean(&self) -> Vec<f64> {
        let nrow = self.nrow();
        if nrow == 0 {
            return vec![];
        }
        let mut result = vec![0.0; nrow];
        let mut count = 0;
        for col in self.columns.values() {
            let mut all_numeric = true;
            for (i, v) in col.iter().enumerate() {
                if let Some(f) = v.as_f64() {
                    result[i] += f;
                } else {
                    all_numeric = false;
                    break;
                }
            }
            if all_numeric {
                count += 1;
            } else {
                // Undo the partial additions
                for (i, v) in col.iter().enumerate() {
                    if let Some(f) = v.as_f64() {
                        result[i] -= f;
                    }
                }
            }
        }
        if count > 0 {
            for v in &mut result {
                *v /= count as f64;
            }
        }
        result
    }
}

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for DataFrame {
    fn eq(&self, other: &Self) -> bool {
        if self.columns.len() != other.columns.len() {
            return false;
        }
        for (name, col) in &self.columns {
            match other.columns.get(name) {
                Some(other_col) => {
                    if col.len() != other_col.len() {
                        return false;
                    }
                    for (a, b) in col.iter().zip(other_col.iter()) {
                        match (a, b) {
                            (Value::Float(fa), Value::Float(fb)) => {
                                if (fa - fb).abs() > 1e-10 && !(fa.is_nan() && fb.is_nan()) {
                                    return false;
                                }
                            }
                            _ => {
                                if a != b {
                                    return false;
                                }
                            }
                        }
                    }
                }
                None => return false,
            }
        }
        true
    }
}

/// Helper to check approximate equality of DataFrames.
pub fn df_approx_eq(a: &DataFrame, b: &DataFrame, atol: f64) -> bool {
    if a.columns.len() != b.columns.len() {
        return false;
    }
    for (name, col) in &a.columns {
        match b.columns.get(name) {
            Some(other_col) => {
                if col.len() != other_col.len() {
                    return false;
                }
                for (va, vb) in col.iter().zip(other_col.iter()) {
                    match (va.as_f64(), vb.as_f64()) {
                        (Some(fa), Some(fb)) => {
                            if (fa - fb).abs() > atol && !(fa.is_nan() && fb.is_nan()) {
                                return false;
                            }
                        }
                        _ => {
                            if va != vb {
                                return false;
                            }
                        }
                    }
                }
            }
            None => return false,
        }
    }
    true
}

/// Convenience macro for creating DataFrames.
#[macro_export]
macro_rules! dataframe {
    ($($name:expr => $values:expr),* $(,)?) => {{
        let mut df = $crate::dataframe::DataFrame::new();
        $(
            df.insert_column($name, $values);
        )*
        df
    }};
}
