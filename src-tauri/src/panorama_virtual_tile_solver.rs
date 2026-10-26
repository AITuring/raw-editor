//! Virtual-tile matching, relation fitting, and station-solver support.
//!
//! This facade keeps the existing panorama_stitching API stable while the
//! implementation is organized by solver responsibility.

use super::*;

#[path = "panorama_virtual_tile_types.rs"]
mod types;
pub(super) use types::*;

#[path = "panorama_virtual_tile_feature_geometry.rs"]
mod feature_geometry;
pub(super) use feature_geometry::*;

#[path = "panorama_virtual_tile_guided_match.rs"]
mod guided_match;
pub(super) use guided_match::*;

#[path = "panorama_virtual_tile_direct.rs"]
mod direct;
pub(super) use direct::*;

#[path = "panorama_virtual_tile_residual.rs"]
mod residual;
pub(super) use residual::*;

#[path = "panorama_virtual_tile_direct_station.rs"]
mod direct_station;
pub(super) use direct_station::*;

#[path = "panorama_virtual_tile_pyramid.rs"]
mod pyramid;
pub(super) use pyramid::*;

#[path = "panorama_virtual_tile_polish.rs"]
mod polish;
pub(super) use polish::*;

#[path = "panorama_virtual_tile_prior.rs"]
mod prior;
pub(super) use prior::*;

#[path = "panorama_virtual_tile_repairs.rs"]
mod repairs;
pub(super) use repairs::*;
