//! Capture-station solver facade.
//!
//! The implementation is split by responsibility so each production module
//! stays below 1,000 lines while the parent `panorama_stitching` module keeps
//! the historical helper API through this facade.

use super::*;

#[path = "panorama_station_grouping.rs"]
mod grouping;
pub(crate) use grouping::*;

#[path = "panorama_station_relation.rs"]
mod relation;
pub(crate) use relation::*;

#[path = "panorama_station_closure.rs"]
mod closure;
pub(crate) use closure::*;
