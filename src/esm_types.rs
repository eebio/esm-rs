use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::dataframe::DataFrame;

/// The main ESM data container, analogous to Julia's esm_zones.
#[derive(Debug, Clone)]
pub struct EsmZones {
    /// All sample data with columns: name, channel, type, values, metadata, + group flags
    pub samples: DataFrame,
    /// Group definitions with columns: group, sample_IDs, metadata
    pub groups: DataFrame,
    /// Transformation equations: name -> {"equation": "..."}
    pub transformations: IndexMap<String, IndexMap<String, String>>,
    /// View definitions: name -> {"data": [...]}
    pub views: IndexMap<String, IndexMap<String, Vec<String>>>,
    /// File metadata
    pub metadata: IndexMap<String, JsonValue>,
}

/// Raw ESM file format (JSON structure).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsmFile {
    #[serde(default)]
    pub samples: IndexMap<String, EsmSample>,
    #[serde(default)]
    pub groups: IndexMap<String, EsmGroup>,
    #[serde(default)]
    pub transformations: IndexMap<String, IndexMap<String, String>>,
    #[serde(default)]
    pub views: IndexMap<String, IndexMap<String, JsonValue>>,
    #[serde(default)]
    pub metadata: IndexMap<String, JsonValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsmSample {
    #[serde(rename = "type")]
    pub sample_type: String,
    pub values: IndexMap<String, Vec<JsonValue>>,
    pub metadata: IndexMap<String, JsonValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EsmGroup {
    #[serde(rename = "type")]
    pub group_type: String,
    #[serde(rename = "sample_IDs")]
    pub sample_ids: Vec<String>,
    pub metadata: IndexMap<String, JsonValue>,
}
