//! `Virtual_Tile_Store`: the versioned, lossless on-disk cache of one focus
//! fused capture station (需求 4.3–4.5, 4.9–4.11).
//!
//! Layout, rooted at `<app_cache_dir>/stack-virtual-tiles/v1/`:
//!
//! ```text
//! <cache_key>/                  # directory name = cache_key in lower hex
//!   meta.json                   # the three cache key ingredients + payload digests
//!   pixels.f32.zst              # f32 little endian, 3 values per pixel
//!   ownership.u16.zst           # u16 little endian, 1 value per pixel
//!   coverage.bits.zst           # 1 bit per pixel, MSB first inside a byte
//!   confidence.f16.zst          # f16 little endian, 1 value per pixel
//! .tmp-<uuid>/                  # an entry being written; renamed into place
//! index.json                    # cache_key -> { last_access_epoch_ms, bytes }
//! ```
//!
//! Three properties of this module carry the requirements:
//!
//! * **Lossless (需求 4.3).**  Pixels, ownership and coverage are written as
//!   raw little endian element planes and only then compressed with zstd, which
//!   is a lossless general purpose compressor.  No image encoder is involved,
//!   so no predictor, chroma subsampling or colour transform can perturb the
//!   round trip: [`VirtualTileStore::load`] returns the very same `f32`
//!   channel values, `u16` owner identifiers and coverage flags that
//!   [`VirtualTileStore::store`] was handed.  `Sharpness_Confidence` is the one
//!   plane the design stores at `f16` width; it is element-for-element exact
//!   for the value actually written, but a wider input is rounded once on the
//!   way in.
//! * **Atomic (需求 4.9).**  Every payload plus `meta.json` is written into
//!   `.tmp-<uuid>/`, `fsync`ed, and only then `rename`d onto `<cache_key>/`.  A
//!   reader therefore never observes a partially written entry: the entry
//!   directory either does not exist or is complete.
//! * **Never fatal (需求 4.10 / 4.11).**  A corrupt entry is deleted and
//!   reported as a miss; an unwritable directory or a failed write leaves the
//!   in-memory tile untouched and is reported as
//!   [`degradation::CACHE_WRITE_UNAVAILABLE`].  No path in this module returns
//!   an error that could abort a run.
//!
//! The cache key is also the run's RANSAC seed source, see
//! [`super::determinism::derive_run_seed_from_cache_key`].

// The cache is declared complete up front so its unit tests can pin the layout,
// the key derivation and every verification branch, while the fusion call site
// that reads and writes entries arrives with the Tile_Compositor stage. Only
// `source_file_sha256` and `CacheKeyInputs` have a production caller today (the
// run seed in `panorama_stitching`). Drop this allow once the compositor leases
// tiles through the store.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use half::f16;
use image::Rgb32FImage;
use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::degradation;
use super::report::VIRTUAL_TILE_CACHE_LIMIT_BYTES;
use crate::panorama_utils::stitching::{
    ColorEncoding, ConfidenceMap, CoverageMask, OwnershipMap, SourceProvenance, VirtualTile,
};

// Cache constants, key derivation, metadata schema, and index helpers.
include!("virtual_tile/key_meta.inc.rs");
// Public lookup/store outcomes, resident leases, and eviction limits.
include!("virtual_tile/store_api.inc.rs");
// Atomic VirtualTileStore load/store implementation and validation.
include!("virtual_tile/store.inc.rs");
// Test-only accessors for cache corruption and payload properties.
include!("virtual_tile/test_access.inc.rs");
// Lossless f32/u16/f16/bit-plane codec and filesystem helpers.
include!("virtual_tile/codec.inc.rs");

// Store regression and corruption tests.
include!("virtual_tile/tests.inc.rs");
