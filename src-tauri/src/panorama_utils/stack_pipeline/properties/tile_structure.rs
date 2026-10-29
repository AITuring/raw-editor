//! Virtual_Tile structure properties (13, 18, 23).
//!
//! Coverage and ownership agree in both directions, provenance sums to the owned
//! pixels, and the masks are isomorphic to the pixels; illegal variants are
//! derived from the drawn legal tiles.

use super::super::test_support::arb_coverage_shape;
use super::super::test_support::arb_virtual_tile;
use super::super::test_support::virtual_tile_with_shape;
use super::super::virtual_tile;
use crate::panorama_utils::stitching;
use proptest::prelude::*;
use sha2::Digest;
use sha2::Sha256;

// ---------------------------------------------------------------------------
// Stage 2 harness: Virtual_Tile structure and the Virtual_Tile disk cache
// ---------------------------------------------------------------------------
//
// `virtual_tile.rs` already carries 18 unit tests, and they pin *one* fixture
// each: a 4x3 tile with a ragged interior, one hand picked missing field, one
// flipped payload byte, one eviction with a budget of a single entry.  The
// properties below are the generalisation of exactly those assertions to the
// input space the fixtures cannot reach — every coverage family, both
// Sharpness_Confidence representations, both colour encodings, 1x1 to 48x48
// tiles including sides that are not multiples of 8, 1..4 Source_RAWs, every
// required `meta.json` key rather than `color_encoding` alone, every payload
// rather than `ownership.u16.zst` alone, and arbitrary entry-size / access-order
// sequences rather than one fixed pair.
//
// Where a property has a direction the generator cannot exercise — "an illegal
// tile is rejected", "a corrupt entry is a miss" — the test derives the illegal
// variant from the drawn legal one.  That is what keeps these tests able to
// fail: asserting the invariants of a value the constructor just accepted is by
// itself a tautology.

/// Tile side range of the cache properties.
///
/// Large enough that the payload planes are real multi-kilobyte buffers and
/// that `48 % 8 == 0` as well as odd sides occur, small enough that 100 cases
/// of disk round trips stay inside the suite's time budget.  The pixel scale of
/// a real Virtual_Tile is irrelevant to every assertion here: the cache is
/// element-wise, so a 48x48 tile exercises the same code paths as a 9000x6000
/// one.
pub(super) const CACHE_PROPERTY_MIN_SIDE: u32 = 16;

pub(super) const CACHE_PROPERTY_MAX_SIDE: u32 = 48;

/// Rebuild a tile with its masks replaced, returning the constructor's verdict.
///
/// Used by the rejection halves of Properties 13, 18 and 23: the drawn tile is
/// legal, so the only way to test the invariant checks is to hand them a
/// deliberately broken variant.
pub(super) fn rebuild_tile(
    tile: &stitching::VirtualTile,
    pixels: image::Rgb32FImage,
    ownership: stitching::OwnershipMap,
    confidence: stitching::ConfidenceMap,
    coverage: stitching::CoverageMask,
    provenance: Vec<stitching::SourceProvenance>,
) -> Result<stitching::VirtualTile, String> {
    stitching::VirtualTile::new(
        tile.station_index,
        tile.tile_to_world,
        pixels,
        ownership,
        confidence,
        coverage,
        tile.color_encoding,
        provenance,
    )
}

/// Provenance records that agree with `ownership`, so a rejection can only come
/// from the invariant under test rather than from the pixel-count invariant.
pub(super) fn provenance_for(
    ownership: &stitching::OwnershipMap,
) -> Vec<stitching::SourceProvenance> {
    ownership
        .legend()
        .iter()
        .zip(ownership.owned_pixel_counts())
        .map(|(path, owned_pixels)| stitching::SourceProvenance {
            absolute_path: path.clone(),
            sha256: Some([0u8; 32]),
            owned_pixels,
        })
        .collect()
}

/// Union bounds of the non-zero pixels as `(x, y, w, h)`, computed from the
/// pixel buffer alone so it is an independent answer to compare
/// `CoverageMask::covered_bounds` against (需求 4.7).
pub(super) fn nonzero_pixel_bounds(pixels: &image::Rgb32FImage) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut any = false;
    for (x, y, pixel) in pixels.enumerate_pixels() {
        if pixel.0 == [0.0, 0.0, 0.0] {
            continue;
        }
        any = true;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    any.then(|| (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 13: 对于任意
    // Virtual_Tile，Coverage_Mask 中被标记为已覆盖的像素在 Ownership_Map 中都有唯一一个
    // Source_RAW 标识；Coverage_Mask 中未覆盖的像素在 Ownership_Map 中都是无 owner 保留
    // 标识且 alpha 为 0。
    //
    // 既有单元测试只固定了一个 4x3 的锯齿掩膜；这里推广到空覆盖、全覆盖、单像素、矩形、
    // 双斑块与逐像素随机六个族，以及 1x1..17x17 的非方形尺寸。测试的另一半是拒绝方向：
    // 任意单个像素上的双向不一致都必须被 `VirtualTile::new` 拒绝，否则这条属性只是在
    // 复述生成器。
    //
    // **Validates: Requirements 3.7, 3.11**
    #[test]
    fn coverage_and_ownership_agree_in_both_directions(
        shape in arb_coverage_shape(),
        legend_len in 1usize..=4,
        probe in any::<usize>(),
    ) {
        let tile = virtual_tile_with_shape(&shape, legend_len);
        let legend_len = tile.ownership.legend().len();

        // 1. 已覆盖 ⇒ 恰好一个合法 Source_RAW 标识。`owner_at` returns a single
        //    `u16`, so "at most one" is structural; what has to hold is that it
        //    is neither the reserved identifier nor outside the legend.
        // 2. 未覆盖 ⇒ 保留标识且 alpha 为 0。The Coverage_Mask *is* the tile's
        //    alpha channel (`pixels` is `Rgb32FImage`, it has none of its own),
        //    so "alpha 为 0" is the coverage byte reading 0, and the pixel
        //    itself must be untouched.
        for y in 0..shape.height {
            for x in 0..shape.width {
                let owner = tile.ownership.owner_at(x, y);
                let covered = tile.coverage.is_covered(x, y);
                prop_assert_eq!(
                    covered,
                    shape.is_covered(x, y),
                    "coverage at ({}, {}) does not follow the drawn shape",
                    x,
                    y
                );
                if covered {
                    prop_assert_ne!(
                        owner,
                        stitching::NO_OWNER,
                        "covered pixel ({}, {}) has no owner",
                        x,
                        y
                    );
                    prop_assert!(
                        (owner as usize) <= legend_len,
                        "covered pixel ({x}, {y}) points at owner {owner} outside a legend of \
                         {legend_len} source(s)"
                    );
                } else {
                    prop_assert_eq!(
                        owner,
                        stitching::NO_OWNER,
                        "uncovered pixel ({}, {}) carries owner {}",
                        x,
                        y,
                        owner
                    );
                    prop_assert_eq!(
                        tile.coverage.covered()[shape.index(x, y)],
                        0u8,
                        "uncovered pixel ({}, {}) has a non-zero alpha",
                        x,
                        y
                    );
                    prop_assert_eq!(
                        tile.pixels.get_pixel(x, y).0,
                        [0.0f32, 0.0, 0.0],
                        "uncovered pixel ({}, {}) carries colour",
                        x,
                        y
                    );
                }
            }
        }
        prop_assert_eq!(
            tile.ownership.assigned_pixels(),
            shape.covered_count(),
            "the owned pixel count must equal the covered pixel count"
        );

        // 3. 拒绝方向: a single disagreeing pixel, in either direction, must
        //    fail construction.  The provenance is rebuilt from the mutated
        //    ownership every time, so the pixel-count invariant cannot be the
        //    one doing the rejecting.
        let covered_indices = (0..shape.covered.len())
            .filter(|&index| shape.covered[index])
            .collect::<Vec<_>>();
        let uncovered_indices = (0..shape.covered.len())
            .filter(|&index| !shape.covered[index])
            .collect::<Vec<_>>();

        if !uncovered_indices.is_empty() {
            // (a) an uncovered pixel handed an owner.
            let index = uncovered_indices[probe % uncovered_indices.len()];
            let mut owners = tile.ownership.owners().to_vec();
            owners[index] = 1;
            let ownership = stitching::OwnershipMap::new(
                shape.width,
                shape.height,
                owners,
                tile.ownership.legend().to_vec(),
            ).expect("the owner identifier stays inside the legend");
            let provenance = provenance_for(&ownership);
            let verdict = rebuild_tile(
                &tile,
                tile.pixels.clone(),
                ownership,
                tile.sharpness_confidence.clone(),
                tile.coverage.clone(),
                provenance,
            );
            let error = verdict.err();
            prop_assert!(
                error.as_deref().is_some_and(|message| message.contains("covered=")),
                "owning an uncovered pixel must be rejected by the coverage/ownership \
                 invariant, got {error:?}"
            );
        }

        if !covered_indices.is_empty() {
            // (b) a covered pixel with the reserved identifier.
            let index = covered_indices[probe % covered_indices.len()];
            let mut owners = tile.ownership.owners().to_vec();
            owners[index] = stitching::NO_OWNER;
            let ownership = stitching::OwnershipMap::new(
                shape.width,
                shape.height,
                owners,
                tile.ownership.legend().to_vec(),
            ).expect("clearing an owner keeps the map legal on its own");
            let provenance = provenance_for(&ownership);
            let verdict = rebuild_tile(
                &tile,
                tile.pixels.clone(),
                ownership,
                tile.sharpness_confidence.clone(),
                tile.coverage.clone(),
                provenance,
            );
            let error = verdict.err();
            prop_assert!(
                error.as_deref().is_some_and(|message| message.contains("covered=")),
                "a covered pixel without an owner must be rejected, got {error:?}"
            );

            // (c) the same disagreement reached from the mask side: the pixel
            //     keeps its owner but loses its coverage.
            let mut covered = tile.coverage.covered().to_vec();
            covered[index] = 0;
            let coverage = stitching::CoverageMask::from_bytes(shape.width, shape.height, covered)
                .expect("the coverage plane keeps one byte per pixel");
            let verdict = rebuild_tile(
                &tile,
                tile.pixels.clone(),
                tile.ownership.clone(),
                tile.sharpness_confidence.clone(),
                coverage,
                tile.provenance.clone(),
            );
            prop_assert!(
                verdict.is_err(),
                "un-covering an owned pixel must be rejected"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 18: 对于任意
    // Capture_Station 合成结果，溯源记录条目数等于参与合成的 Source_RAW 数量，每条记录都
    // 携带绝对路径、对该文件全部字节计算的 SHA-256 与 ownership 像素数，且各条目 ownership
    // 像素数之和等于 Ownership_Map 中已赋 owner 的像素数。
    //
    // 求和恒等在任意覆盖形状、任意 1..4 源图数与任意 owner 分布上成立；"对该文件全部字节
    // 计算" 这一条用一个随机长度的临时文件对 `source_file_sha256` 复核（既有单元测试只用
    // 一个固定长度）。拒绝方向覆盖条目数不符、像素数被扰动 ±1 与路径与 legend 错位三种。
    //
    // **Validates: Requirements 4.1**
    #[test]
    fn virtual_tile_provenance_sums_to_the_owned_pixels(
        tile in arb_virtual_tile(),
        payload in prop::collection::vec(any::<u8>(), 0..4096),
        probe in any::<usize>(),
    ) {
        let legend = tile.ownership.legend();

        // 1. 条目数等于参与合成的 Source_RAW 数量 — the Ownership_Map legend is
        //    that set, and the constructor holds the two to the same length.
        prop_assert_eq!(
            tile.provenance.len(),
            legend.len(),
            "one provenance record per participating Source_RAW"
        );

        // 2. 每条记录携带绝对路径、SHA-256 与 ownership 像素数.
        let mut expected_counts = vec![0u64; legend.len()];
        for &owner in tile.ownership.owners() {
            if owner != stitching::NO_OWNER {
                expected_counts[owner as usize - 1] += 1;
            }
        }
        for (index, record) in tile.provenance.iter().enumerate() {
            prop_assert!(
                record.absolute_path.is_absolute(),
                "provenance {} carries the relative path {}",
                index,
                record.absolute_path.display()
            );
            prop_assert_eq!(
                &record.absolute_path,
                &legend[index],
                "provenance {} is not its ownership legend entry",
                index
            );
            prop_assert!(
                record.sha256.is_some(),
                "provenance {} carries no SHA-256",
                index
            );
            prop_assert_eq!(
                record.owned_pixels,
                expected_counts[index],
                "provenance {} claims {} owned pixel(s)",
                index,
                record.owned_pixels
            );
        }

        // 3. 求和恒等.
        let sum = tile
            .provenance
            .iter()
            .fold(0u64, |total, record| total + record.owned_pixels);
        prop_assert_eq!(
            sum,
            tile.ownership.assigned_pixels(),
            "the provenance pixel counts must sum to the owned pixels"
        );
        prop_assert_eq!(
            sum,
            tile.coverage.covered_pixels(),
            "every owned pixel is a covered pixel, so the sum is the coverage too"
        );

        // 4. "对该文件全部字节计算的 SHA-256": the digest helper the fusion call
        //    site uses must agree with a digest of the whole file, at any
        //    length, including the empty file and lengths that are not a
        //    multiple of the read chunk.
        let directory = tempfile::TempDir::new().expect("temp dir");
        let source = directory.path().join("DSC_0001.NEF");
        std::fs::write(&source, &payload).expect("writing the source fixture");
        let expected: [u8; 32] = Sha256::digest(&payload).into();
        prop_assert_eq!(
            virtual_tile::source_file_sha256(&source).expect("the digest succeeds"),
            expected,
            "the provenance digest must cover every byte of the file"
        );

        // 5. 拒绝方向: a perturbed count, a dropped record and a legend the
        //    provenance no longer matches must all fail construction.
        let mut perturbed = tile.provenance.clone();
        let slot = probe % perturbed.len();
        perturbed[slot].owned_pixels += 1;
        prop_assert!(
            rebuild_tile(
                &tile,
                tile.pixels.clone(),
                tile.ownership.clone(),
                tile.sharpness_confidence.clone(),
                tile.coverage.clone(),
                perturbed,
            ).is_err(),
            "an owned pixel count that is one too high must be rejected"
        );

        let mut dropped = tile.provenance.clone();
        dropped.remove(slot);
        prop_assert!(
            rebuild_tile(
                &tile,
                tile.pixels.clone(),
                tile.ownership.clone(),
                tile.sharpness_confidence.clone(),
                tile.coverage.clone(),
                dropped,
            ).is_err(),
            "a missing provenance record must be rejected"
        );

        if tile.provenance.len() >= 2 {
            let mut misaligned = tile.provenance.clone();
            misaligned.swap(0, 1);
            prop_assert!(
                rebuild_tile(
                    &tile,
                    tile.pixels.clone(),
                    tile.ownership.clone(),
                    tile.sharpness_confidence.clone(),
                    tile.coverage.clone(),
                    misaligned,
                ).is_err(),
                "provenance that no longer follows the legend order must be rejected"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 23: 对于任意
    // Virtual_Tile，Ownership_Map、Sharpness_Confidence、Coverage_Mask 与合成像素的宽高
    // 完全相同，每个像素的 Sharpness_Confidence 等于其所属 ownership 单元在 `[0, 1]` 内的
    // 值，且合成像素的有效范围等于 Coverage_Mask 已覆盖像素的联合轴对齐外接边界。
    //
    // 生成器给出非方形尺寸，所以四个平面中任何一个把行列索引算反都会在这里变红；
    // 有效范围一条与 `covered_bounds()` 的独立实现对照。拒绝方向覆盖三个平面各自的尺寸
    // 不符、置信度越界与"像素越出覆盖边界"。
    //
    // **Validates: Requirements 4.6, 4.7**
    #[test]
    fn virtual_tile_masks_are_isomorphic_to_the_pixels(tile in arb_virtual_tile()) {
        let (width, height) = tile.pixels.dimensions();

        // 1. 宽高完全相同 — including the tile's own record of them.
        prop_assert_eq!((tile.width, tile.height), (width, height));
        prop_assert_eq!(tile.ownership.dimensions(), (width, height));
        prop_assert_eq!(tile.sharpness_confidence.dimensions(), (width, height));
        prop_assert_eq!(tile.coverage.dimensions(), (width, height));

        // 2. 每个像素的 Sharpness_Confidence 在 [0, 1] 内, and the two accessors
        //    of every plane agree on the row major index.  A swapped row/column
        //    is only observable on a non-square tile, which the generator draws.
        let row_major = tile.sharpness_confidence.to_row_major();
        prop_assert_eq!(row_major.len(), width as usize * height as usize);
        for y in 0..height {
            for x in 0..width {
                let index = y as usize * width as usize + x as usize;
                let confidence = tile.sharpness_confidence.value_at(x, y);
                prop_assert!(
                    confidence.is_finite() && (0.0..=1.0).contains(&confidence),
                    "confidence {confidence} at ({x}, {y}) is outside [0, 1]"
                );
                prop_assert_eq!(
                    confidence,
                    row_major[index],
                    "confidence at ({}, {}) disagrees with the row major plane",
                    x,
                    y
                );
                prop_assert_eq!(
                    tile.ownership.owner_at(x, y),
                    tile.ownership.owners()[index],
                    "owner at ({}, {}) disagrees with the row major plane",
                    x,
                    y
                );
                prop_assert_eq!(
                    tile.coverage.is_covered(x, y),
                    tile.coverage.covered()[index] > 0,
                    "coverage at ({}, {}) disagrees with the row major plane",
                    x,
                    y
                );
                // The pixel carries colour exactly where a source owns it.
                prop_assert_eq!(
                    tile.pixels.get_pixel(x, y).0 != [0.0, 0.0, 0.0],
                    tile.coverage.is_covered(x, y),
                    "pixel ({}, {}) and its coverage flag disagree",
                    x,
                    y
                );
            }
        }

        // 3. 有效范围等于覆盖像素的联合外接边界.
        prop_assert_eq!(
            tile.coverage.covered_bounds(),
            nonzero_pixel_bounds(&tile.pixels),
            "the valid pixel extent must be the coverage union bounds"
        );

        // 4. 拒绝方向: each plane, one pixel too tall.
        let taller_ownership = stitching::OwnershipMap::new(
            width,
            height + 1,
            vec![stitching::NO_OWNER; width as usize * (height as usize + 1)],
            tile.ownership.legend().to_vec(),
        ).expect("an all-unowned map is legal on its own");
        prop_assert!(
            rebuild_tile(
                &tile,
                tile.pixels.clone(),
                taller_ownership,
                tile.sharpness_confidence.clone(),
                tile.coverage.clone(),
                tile.provenance.clone(),
            ).is_err(),
            "an Ownership_Map of another size must be rejected"
        );
        let taller_coverage = stitching::CoverageMask::from_bytes(
            width,
            height + 1,
            vec![0u8; width as usize * (height as usize + 1)],
        ).expect("an empty mask is legal on its own");
        prop_assert!(
            rebuild_tile(
                &tile,
                tile.pixels.clone(),
                tile.ownership.clone(),
                tile.sharpness_confidence.clone(),
                taller_coverage,
                tile.provenance.clone(),
            ).is_err(),
            "a Coverage_Mask of another size must be rejected"
        );
        let taller_confidence = stitching::ConfidenceMap::zero(width, height + 1);
        prop_assert!(
            rebuild_tile(
                &tile,
                tile.pixels.clone(),
                tile.ownership.clone(),
                taller_confidence,
                tile.coverage.clone(),
                tile.provenance.clone(),
            ).is_err(),
            "a Sharpness_Confidence map of another size must be rejected"
        );

        // 5. 拒绝方向: a confidence value just outside the unit range.
        let mut out_of_range = row_major.clone();
        out_of_range[0] = 1.0 + f32::EPSILON;
        let out_of_range = stitching::ConfidenceMap::per_pixel(width, height, out_of_range)
            .expect("the plane still has one value per pixel");
        prop_assert!(
            rebuild_tile(
                &tile,
                tile.pixels.clone(),
                tile.ownership.clone(),
                out_of_range,
                tile.coverage.clone(),
                tile.provenance.clone(),
            ).is_err(),
            "a Sharpness_Confidence above 1 must be rejected"
        );

        // 6. 拒绝方向: a pixel outside the coverage union bounds.  Only
        //    reachable when the bounds are not already the whole tile, which is
        //    why the coverage families include empty, single pixel and
        //    rectangular masks.
        if let Some((left, top, bounds_width, bounds_height)) = tile.coverage.covered_bounds() {
            let outside = (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .find(|&(x, y)| {
                    x < left || y < top || x >= left + bounds_width || y >= top + bounds_height
                });
            if let Some((x, y)) = outside {
                let mut pixels = tile.pixels.clone();
                pixels.put_pixel(x, y, image::Rgb([1.0, 1.0, 1.0]));
                prop_assert!(
                    rebuild_tile(
                        &tile,
                        pixels,
                        tile.ownership.clone(),
                        tile.sharpness_confidence.clone(),
                        tile.coverage.clone(),
                        tile.provenance.clone(),
                    ).is_err(),
                    "a pixel at ({x}, {y}), outside the coverage bounds \
                     ({left}, {top}, {bounds_width}, {bounds_height}), must be rejected"
                );
            }
        } else {
            let mut pixels = tile.pixels.clone();
            pixels.put_pixel(0, 0, image::Rgb([1.0, 1.0, 1.0]));
            prop_assert!(
                rebuild_tile(
                    &tile,
                    pixels,
                    tile.ownership.clone(),
                    tile.sharpness_confidence.clone(),
                    tile.coverage.clone(),
                    tile.provenance.clone(),
                ).is_err(),
                "a tile with pixels but no coverage must be rejected"
            );
        }
    }
}
