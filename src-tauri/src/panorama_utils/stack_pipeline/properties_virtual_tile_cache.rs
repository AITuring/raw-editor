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
const CACHE_PROPERTY_MIN_SIDE: u32 = 16;
const CACHE_PROPERTY_MAX_SIDE: u32 = 48;

/// Rebuild a tile with its masks replaced, returning the constructor's verdict.
///
/// Used by the rejection halves of Properties 13, 18 and 23: the drawn tile is
/// legal, so the only way to test the invariant checks is to hand them a
/// deliberately broken variant.
fn rebuild_tile(
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
fn provenance_for(ownership: &stitching::OwnershipMap) -> Vec<stitching::SourceProvenance> {
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
fn nonzero_pixel_bounds(pixels: &image::Rgb32FImage) -> Option<(u32, u32, u32, u32)> {
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
// ---------------------------------------------------------------------------
// Virtual_Tile_Store properties (需求 4.3–4.5, 4.9–4.11, 14.4)
// ---------------------------------------------------------------------------

/// Cache key ingredients for a tile whose sources are not on disk.
///
/// Properties 19 and 21 are about the payload and the index, not about the
/// digest of a real file; Property 20 builds its ingredients from real files
/// instead.
fn cache_inputs_for(tile: &stitching::VirtualTile, version: &str) -> CacheKeyInputs {
    let sources = tile
        .provenance
        .iter()
        .enumerate()
        .map(|(index, record)| {
            let mut digest = record.sha256.unwrap_or([0u8; 32]);
            digest[0] = digest[0].wrapping_add(index as u8);
            (record.absolute_path.to_string_lossy().to_string(), digest)
        })
        .collect::<Vec<_>>();
    CacheKeyInputs::new(version, &sources)
}

/// Element-for-element comparison of a stored and a loaded tile (需求 4.3).
fn assert_cache_round_trip(
    stored: &stitching::VirtualTile,
    loaded: &stitching::VirtualTile,
) -> Result<(), TestCaseError> {
    prop_assert_eq!(loaded.station_index, stored.station_index);
    prop_assert_eq!((loaded.width, loaded.height), (stored.width, stored.height));
    prop_assert_eq!(loaded.pixels.dimensions(), stored.pixels.dimensions());
    prop_assert_eq!(loaded.color_encoding, stored.color_encoding);
    prop_assert_eq!(&loaded.provenance, &stored.provenance);
    prop_assert_eq!(&loaded.ownership, &stored.ownership);
    prop_assert_eq!(
        loaded.coverage.covered_bounds(),
        stored.coverage.covered_bounds()
    );
    for (index, (left, right)) in loaded
        .tile_to_world
        .iter()
        .zip(stored.tile_to_world.iter())
        .enumerate()
    {
        prop_assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "tile_to_world coefficient {} changed ({} vs {})",
            index,
            left,
            right
        );
    }
    // Pixels: raw bits, so a single rounded mantissa or a lost sign fails.
    for (index, (left, right)) in loaded
        .pixels
        .as_raw()
        .iter()
        .zip(stored.pixels.as_raw().iter())
        .enumerate()
    {
        prop_assert_eq!(
            left.to_bits(),
            right.to_bits(),
            "pixel channel {} changed ({} vs {})",
            index,
            left,
            right
        );
    }
    for y in 0..stored.height {
        for x in 0..stored.width {
            prop_assert_eq!(
                loaded.coverage.is_covered(x, y),
                stored.coverage.is_covered(x, y),
                "coverage at ({}, {})",
                x,
                y
            );
            prop_assert_eq!(
                loaded.ownership.owner_at(x, y),
                stored.ownership.owner_at(x, y),
                "owner at ({}, {})",
                x,
                y
            );
            // Sharpness_Confidence is the one plane the design stores at f16
            // width, so the contract is "exactly the value that was written",
            // which for an f16-representable input is bit equality.
            prop_assert_eq!(
                loaded.sharpness_confidence.value_at(x, y),
                f16::from_f32(stored.sharpness_confidence.value_at(x, y)).to_f32(),
                "confidence at ({}, {})",
                x,
                y
            );
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 19: 对于任意
    // Virtual_Tile，写入缓存后读回的合成像素每通道值、Ownership_Map 每个 owner 标识与
    // Coverage_Mask 每个覆盖标记都与写入时逐元素完全相同。
    //
    // 既有单元测试只往返了一个 4x3 的固定瓦片；这里推广到 16x16..48x48（含非方形与
    // 不是 8 的倍数的宽度，即一位一像素编码的尾字节）、六个覆盖族、两种色彩编码、
    // 两种 Sharpness_Confidence 表示与有无溯源摘要。
    //
    // **Validates: Requirements 4.3**
    #[test]
    fn virtual_tile_cache_round_trip_is_lossless(
        tile in arb_virtual_tile_in(CACHE_PROPERTY_MIN_SIDE, CACHE_PROPERTY_MAX_SIDE),
        drop_digests in any::<bool>(),
    ) {
        let mut tile = tile;
        if drop_digests {
            // The fusion call site does not always have a digest yet, so the
            // absent case has to round trip as "absent" rather than as zeroes.
            for record in &mut tile.provenance {
                record.sha256 = None;
            }
        }
        let directory = TempDir::new().expect("temp dir");
        let store = VirtualTileStore::new(directory.path());
        let inputs = cache_inputs_for(&tile, "stack-prop.19");

        prop_assert!(
            matches!(store.load(&inputs), CacheLookup::Miss),
            "an empty cache must miss"
        );
        let bytes = match store.store(&inputs, &tile) {
            StoreOutcome::Written { bytes, evicted, total_bytes } => {
                prop_assert_eq!(evicted, 0, "a single small entry cannot exceed 64 GiB");
                prop_assert_eq!(total_bytes, bytes);
                bytes
            }
            other => {
                prop_assert!(false, "a temp dir must be writable: {other:?}");
                unreachable!()
            }
        };
        prop_assert!(bytes > 0, "the entry must have a size");

        let loaded = match store.load(&inputs) {
            CacheLookup::Hit(loaded) => *loaded,
            other => {
                prop_assert!(false, "the entry just written must verify: {other:?}");
                unreachable!()
            }
        };
        assert_cache_round_trip(&tile, &loaded)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 20: 对于任意
    // Source_RAW 路径集合、SHA-256 集合与流程版本标识三元组，用同一三元组读取此前以该三元组
    // 写入的缓存条目必然命中且解码的 Source_RAW 数量为 0；三要素中任一项被扰动则必然未命中。
    //
    // "解码的 Source_RAW 数量为 0" 在这里是可证的而非声明的：命中之前源文件已被删除，
    // 任何试图读取源文件的代码路径都会失败。既有单元测试用的是三个手写的扰动，这里扰动的
    // 路径、摘要与文件内容都是随机的，并额外断言集合语义（重排三元组仍然命中）。
    //
    // **Validates: Requirements 4.5, 4.9**
    #[test]
    fn a_cache_hit_requires_all_three_key_ingredients(
        tile in arb_virtual_tile_in(CACHE_PROPERTY_MIN_SIDE, 24),
        contents in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..512), 1..=3),
        probe in any::<usize>(),
    ) {
        let sources = TempDir::new().expect("temp dir");
        let mut pairs = Vec::new();
        for (index, content) in contents.iter().enumerate() {
            let path = sources.path().join(format!("DSC_{:04}.NEF", 3680 + index));
            fs::write(&path, content).expect("writing a source fixture");
            let digest = virtual_tile::source_file_sha256(&path)
                .expect("the read only digest succeeds");
            pairs.push((path.to_string_lossy().to_string(), digest));
        }
        let inputs = CacheKeyInputs::new(STACK_PIPELINE_VERSION, &pairs);

        let cache = TempDir::new().expect("temp dir");
        let store = VirtualTileStore::new(cache.path());
        prop_assert!(
            matches!(store.store(&inputs, &tile), StoreOutcome::Written { .. }),
            "a temp dir must be writable"
        );

        // Every Source_RAW disappears before the hit, so a hit that decoded
        // anything could not have succeeded.
        for (path, _) in &pairs {
            fs::remove_file(path).expect("removing a source fixture");
            prop_assert!(!Path::new(path).exists());
        }
        match store.load(&inputs) {
            CacheLookup::Hit(loaded) => {
                prop_assert_eq!((loaded.width, loaded.height), (tile.width, tile.height));
                prop_assert_eq!(&loaded.ownership, &tile.ownership);
            }
            other => {
                prop_assert!(false, "the same triple must hit: {other:?}");
            }
        }

        // Set semantics: the ingredients are sorted, so a permuted triple is
        // the same triple (需求 4.5).
        let mut permuted = pairs.clone();
        permuted.reverse();
        let permuted = CacheKeyInputs::new(STACK_PIPELINE_VERSION, &permuted);
        prop_assert_eq!(permuted.cache_key(), inputs.cache_key());
        prop_assert!(matches!(store.load(&permuted), CacheLookup::Hit(_)));

        // 1. A different pipeline version.
        let other_version = CacheKeyInputs::new(
            &format!("{STACK_PIPELINE_VERSION}.next"),
            &pairs,
        );
        // 2. A different path set.
        let mut renamed = pairs.clone();
        let slot = probe % renamed.len();
        renamed[slot].0.push_str(".bak");
        let other_paths = CacheKeyInputs::new(STACK_PIPELINE_VERSION, &renamed);
        // 3. A different digest set — one flipped bit of one source file.
        let mut edited = pairs.clone();
        edited[slot].1[probe % 32] ^= 0x01;
        let other_digests = CacheKeyInputs::new(STACK_PIPELINE_VERSION, &edited);

        for (what, probe_inputs) in [
            ("the pipeline version", other_version),
            ("the source path set", other_paths),
            ("the source digest set", other_digests),
        ] {
            prop_assert_ne!(
                probe_inputs.cache_key(),
                inputs.cache_key(),
                "perturbing {} must change the cache key",
                what
            );
            prop_assert!(
                matches!(store.load(&probe_inputs), CacheLookup::Miss),
                "perturbing {what} must miss"
            );
        }
        // None of the misses touched the entry that does verify.
        prop_assert!(matches!(store.load(&inputs), CacheLookup::Hit(_)));
    }
}

/// Access times of `index.json` need to be distinguishable, and the index
/// records them in milliseconds, so the accesses of Property 21 are spaced.
const CACHE_ACCESS_SPACING: Duration = Duration::from_millis(2);

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 21: 对于任意缓存条目大小
    // 序列与任意访问序列，淘汰后缓存总占用不超过 64 GiB，且每个被删除条目的最后访问时间都
    // 不晚于所有被保留条目的最后访问时间。
    //
    // 既有单元测试用的是"预算刚好一条"的两条目场景；这里的条目大小序列、访问序列与预算
    // 比例都是随机的。64 GiB 本身不写盘：预算被注入为实测字节数的一个比例，同时断言生产
    // 默认预算确实是 64 GiB。
    //
    // **Validates: Requirements 4.4**
    #[test]
    fn cache_eviction_keeps_the_budget_and_the_access_order(
        sides in prop::collection::vec(4u32..=12, 2..=4),
        access in prop::collection::vec(any::<usize>(), 0..6),
        budget_quarters in 1u64..=4,
    ) {
        let _run_scope = crate::panorama_utils::stack_pipeline::degradation::begin_run_scope();
        let directory = TempDir::new().expect("temp dir");
        prop_assert_eq!(
            VirtualTileStore::new(directory.path()).limit_bytes(),
            VIRTUAL_TILE_CACHE_LIMIT_BYTES,
            "the production budget is 64 GiB"
        );

        let store = VirtualTileStore::new(directory.path());
        let mut entries = Vec::new();
        for (index, &side) in sides.iter().enumerate() {
            let tile = virtual_tile_with_shape(&full_coverage_shape(side, side + 1), 1);
            let inputs = CacheKeyInputs::new(
                STACK_PIPELINE_VERSION,
                &[(format!("/scan/DSC_{index:04}.NEF"), [index as u8; 32])],
            );
            let bytes = match store.store(&inputs, &tile) {
                StoreOutcome::Written { bytes, evicted, .. } => {
                    prop_assert_eq!(evicted, 0, "64 GiB holds every entry of this property");
                    bytes
                }
                other => {
                    prop_assert!(false, "a temp dir must be writable: {other:?}");
                    unreachable!()
                }
            };
            entries.push((inputs, bytes));
            std::thread::sleep(CACHE_ACCESS_SPACING);
        }

        // The access sequence: every hit refreshes `last_access_epoch_ms`.
        for &slot in &access {
            let (inputs, _) = &entries[slot % entries.len()];
            prop_assert!(matches!(store.load(inputs), CacheLookup::Hit(_)));
            std::thread::sleep(CACHE_ACCESS_SPACING);
        }

        let before = store
            .index_snapshot_for_test()
            .into_iter()
            .map(|(key, last_access, _)| (key, last_access))
            .collect::<HashMap<_, _>>();
        prop_assert_eq!(
            before.len(),
            entries.len(),
            "every stored entry is in the index"
        );

        // A budget that forces at least one eviction: the new entry alone
        // already pushes the cache past the size of everything written so far.
        let total_before = entries.iter().map(|(_, bytes)| *bytes).sum::<u64>();
        let limit = (total_before * budget_quarters / 4).max(1);
        let store = VirtualTileStore::new(directory.path()).with_limit_bytes(limit);
        let extra = CacheKeyInputs::new(
            STACK_PIPELINE_VERSION,
            &[("/scan/DSC_9999.NEF".to_string(), [0xAB; 32])],
        );
        let evicted = match store.store(&extra, &virtual_tile_with_shape(&full_coverage_shape(8, 9), 1)) {
            StoreOutcome::Written { evicted, total_bytes, .. } => {
                prop_assert!(
                    total_bytes <= limit,
                    "the cache holds {total_bytes} byte(s) under a {limit} byte budget"
                );
                evicted
            }
            other => {
                prop_assert!(false, "a temp dir must be writable: {other:?}");
                unreachable!()
            }
        };
        prop_assert!(
            evicted >= 1,
            "a budget of {limit} byte(s) over {total_before} byte(s) must evict"
        );
        prop_assert!(store.total_bytes() <= limit, "the budget is a ceiling");

        // Whole entries only, and the index agrees with the disk.
        let after = store
            .index_snapshot_for_test()
            .into_iter()
            .map(|(key, last_access, _)| (key, last_access))
            .collect::<HashMap<_, _>>();
        let mut deleted_access = Vec::new();
        let mut retained_access = Vec::new();
        for (inputs, _) in &entries {
            let key = inputs.cache_key();
            let entry = store.entry_dir_for_test(&key);
            prop_assert_eq!(
                entry.exists(),
                after.contains_key(&key),
                "the index and the disk disagree about {}",
                key
            );
            let last_access = before[&key];
            if entry.exists() {
                retained_access.push(last_access);
                prop_assert!(
                    entry.join(test_access::META_FILE_NAME).exists(),
                    "a retained entry keeps its meta.json"
                );
            } else {
                deleted_access.push(last_access);
            }
        }
        prop_assert!(
            !deleted_access.is_empty(),
            "the eviction pass reported {evicted} deletion(s) but nothing is gone"
        );
        if let (Some(&newest_deleted), Some(&oldest_retained)) = (
            deleted_access.iter().max(),
            retained_access.iter().min(),
        ) {
            prop_assert!(
                newest_deleted <= oldest_retained,
                "a deleted entry was accessed at {newest_deleted}, later than the retained \
                 entry accessed at {oldest_retained}"
            );
        }
    }
}

/// One way a cache entry can be damaged on disk (需求 4.10).
///
/// The design names three conditions — a missing field, recorded dimensions
/// that disagree with the payload, a SHA-256 set that cannot be re-verified —
/// and each of them has more than one on-disk shape.  This enum is here rather
/// than in `test_support` because it mutates the *cache layout*, which only
/// `virtual_tile::test_access` exposes; it is not an input generator.
#[derive(Debug, Clone, Copy)]
enum CacheCorruption {
    /// One required `meta.json` key is gone.
    RemoveMetaKey(usize),
    /// One required key of a nested `meta.json` object is gone: a
    /// `cache_key_inputs` ingredient, a `coverage_bounds` corner, a
    /// `provenance` record field or a `payloads` entry field.
    RemoveNestedMetaKey(usize, usize),
    /// One payload file is gone.
    RemovePayload(usize),
    /// `meta.json` records a taller tile than the payloads hold.
    GrowRecordedHeight,
    /// `meta.json` records a wider tile than the payloads hold.
    GrowRecordedWidth,
    /// One bit of one payload plane is flipped and the plane re-compressed, so
    /// only the recorded digest can catch it.
    FlipPayloadByte(usize, usize),
    /// One payload file is not a valid frame any more.
    TruncatePayload(usize),
    /// The recorded source digest set no longer matches the run.
    RewriteDigestSet,
    /// The recorded source path set no longer matches the run.
    RewritePathSet,
}

fn arb_cache_corruption() -> impl Strategy<Value = CacheCorruption> {
    prop_oneof![
        any::<usize>().prop_map(CacheCorruption::RemoveMetaKey),
        (any::<usize>(), any::<usize>())
            .prop_map(|(section, key)| CacheCorruption::RemoveNestedMetaKey(section, key)),
        any::<usize>().prop_map(CacheCorruption::RemovePayload),
        Just(CacheCorruption::GrowRecordedHeight),
        Just(CacheCorruption::GrowRecordedWidth),
        (any::<usize>(), any::<usize>())
            .prop_map(|(payload, byte)| CacheCorruption::FlipPayloadByte(payload, byte)),
        any::<usize>().prop_map(CacheCorruption::TruncatePayload),
        Just(CacheCorruption::RewriteDigestSet),
        Just(CacheCorruption::RewritePathSet),
    ]
}

fn read_meta(entry: &Path) -> Value {
    serde_json::from_slice(&fs::read(entry.join(test_access::META_FILE_NAME)).expect("meta.json"))
        .expect("meta.json parses")
}

fn write_meta(entry: &Path, meta: &Value) {
    fs::write(
        entry.join(test_access::META_FILE_NAME),
        serde_json::to_vec(meta).expect("meta.json re-serializes"),
    )
    .expect("writing meta.json");
}

impl CacheCorruption {
    /// The reason identifier 需求 4.10 demands for this damage.
    fn expected_reason(self) -> &'static str {
        match self {
            Self::RemoveMetaKey(_)
            | Self::RemoveNestedMetaKey(_, _)
            | Self::RemovePayload(_)
            | Self::TruncatePayload(_) => degradation::CACHE_ENTRY_FIELD_MISSING,
            Self::GrowRecordedHeight | Self::GrowRecordedWidth => {
                degradation::CACHE_ENTRY_DIMENSION_MISMATCH
            }
            Self::FlipPayloadByte(_, _) | Self::RewriteDigestSet | Self::RewritePathSet => {
                degradation::CACHE_ENTRY_SHA_MISMATCH
            }
        }
    }

    fn apply(self, entry: &Path) {
        match self {
            Self::RemoveMetaKey(which) => {
                let key =
                    test_access::META_REQUIRED_KEYS[which % test_access::META_REQUIRED_KEYS.len()];
                let mut meta = read_meta(entry);
                meta.as_object_mut()
                    .expect("meta.json is an object")
                    .remove(key);
                write_meta(entry, &meta);
            }
            Self::RemoveNestedMetaKey(section, which) => {
                const SECTIONS: &[(&str, &[&str])] = &[
                    ("cache_key_inputs", test_access::CACHE_KEY_INPUTS_KEYS),
                    ("coverage_bounds", test_access::COVERAGE_BOUNDS_KEYS),
                    ("provenance", test_access::PROVENANCE_KEYS),
                    ("payloads", test_access::PAYLOAD_KEYS),
                ];
                let (root_key, keys) = SECTIONS[section % SECTIONS.len()];
                let key = keys[which % keys.len()];
                let mut meta = read_meta(entry);
                let target = match root_key {
                    "provenance" => meta[root_key]
                        .as_array_mut()
                        .and_then(|records| records.first_mut()),
                    "payloads" => meta[root_key]
                        .as_object_mut()
                        .and_then(|payloads| payloads.values_mut().next()),
                    _ => Some(&mut meta[root_key]),
                };
                let removed = target
                    .and_then(Value::as_object_mut)
                    .is_some_and(|object| object.remove(key).is_some());
                if !removed {
                    // The section is nullable or empty for this tile — a fully
                    // uncovered tile records `coverage_bounds: null` — so there
                    // is no nested key to delete.  Deleting the section itself
                    // is the same class of damage.
                    meta.as_object_mut()
                        .expect("meta.json is an object")
                        .remove(root_key);
                }
                write_meta(entry, &meta);
            }
            Self::RemovePayload(which) => {
                let name =
                    test_access::REQUIRED_PAYLOADS[which % test_access::REQUIRED_PAYLOADS.len()];
                fs::remove_file(entry.join(name)).expect("removing a payload");
            }
            Self::GrowRecordedHeight => {
                let mut meta = read_meta(entry);
                let height = meta["height"].as_u64().expect("height is a number");
                meta["height"] = serde_json::json!(height + 1);
                write_meta(entry, &meta);
            }
            Self::GrowRecordedWidth => {
                let mut meta = read_meta(entry);
                let width = meta["width"].as_u64().expect("width is a number");
                meta["width"] = serde_json::json!(width + 1);
                write_meta(entry, &meta);
            }
            Self::FlipPayloadByte(which, byte) => {
                let name =
                    test_access::REQUIRED_PAYLOADS[which % test_access::REQUIRED_PAYLOADS.len()];
                let path = entry.join(name);
                let mut plane = test_access::read_payload(&path).expect("the payload decodes");
                let slot = byte % plane.len();
                plane[slot] ^= 0x01;
                fs::write(&path, test_access::compress_payload(&plane))
                    .expect("writing the payload");
            }
            Self::TruncatePayload(which) => {
                let name =
                    test_access::REQUIRED_PAYLOADS[which % test_access::REQUIRED_PAYLOADS.len()];
                fs::write(entry.join(name), b"not a frame").expect("writing the payload");
            }
            Self::RewriteDigestSet => {
                let mut meta = read_meta(entry);
                meta["cache_key_inputs"]["sorted_source_sha256"][0] =
                    serde_json::json!("0".repeat(64));
                write_meta(entry, &meta);
            }
            Self::RewritePathSet => {
                let mut meta = read_meta(entry);
                meta["cache_key_inputs"]["sorted_paths"][0] = serde_json::json!("/moved.NEF");
                write_meta(entry, &meta);
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 22: 对于任意缓存条目与
    // 任意一种损坏（字段缺失、记录尺寸与实际像素尺寸不符、SHA-256 集合无法复核），读取结果
    // 为未命中、该条目被删除、Virtual_Tile 被重新合成，且缓存失效原因标识符被记录。
    //
    // 既有单元测试各自固定了一种损坏与一个受损字段；这里 `meta.json` 的 13 个根级必需键、
    // 嵌套对象（`cache_key_inputs` / `coverage_bounds` / `provenance` / `payloads`）内的
    // 每个键、4 个载荷、载荷内任意一位、以及路径集合与摘要集合都在输入空间内。
    //
    // **Validates: Requirements 4.10**
    #[test]
    fn a_corrupt_cache_entry_is_always_a_miss(
        tile in arb_virtual_tile_in(CACHE_PROPERTY_MIN_SIDE, 24),
        corruption in arb_cache_corruption(),
    ) {
        let directory = TempDir::new().expect("temp dir");
        let store = VirtualTileStore::new(directory.path());
        let inputs = cache_inputs_for(&tile, "stack-prop.22");
        prop_assert!(
            matches!(store.store(&inputs, &tile), StoreOutcome::Written { .. }),
            "a temp dir must be writable"
        );
        let entry = store.entry_dir_for_test(&inputs.cache_key());
        prop_assert!(matches!(store.load(&inputs), CacheLookup::Hit(_)));

        corruption.apply(&entry);

        // 1. 未命中，且失效原因标识符来自冻结标识符集合（需求 12.7）。
        match store.load(&inputs) {
            CacheLookup::Invalid { reason, detail } => {
                prop_assert_eq!(
                    reason,
                    corruption.expected_reason(),
                    "{:?} was reported as {} ({})",
                    corruption,
                    reason,
                    detail
                );
                prop_assert!(
                    frozen_reason_identifiers().contains(reason),
                    "the cache invalidation reason must be a frozen identifier"
                );
                prop_assert!(!detail.is_empty(), "the reason must carry a detail");
            }
            other => {
                prop_assert!(false, "{corruption:?} must invalidate the entry, got {other:?}");
            }
        }

        // 2. 该条目被删除，后续读取是普通未命中（而不是反复失败的坏条目）。
        prop_assert!(!entry.exists(), "a corrupt entry must be deleted whole");
        prop_assert!(matches!(store.load(&inputs), CacheLookup::Miss));

        // 3. Virtual_Tile 被重新合成: the caller re-fuses and stores again, and
        //    the entry that comes back is the original one element for element.
        prop_assert!(
            matches!(store.store(&inputs, &tile), StoreOutcome::Written { .. }),
            "a temp dir must be writable"
        );
        match store.load(&inputs) {
            CacheLookup::Hit(loaded) => assert_cache_round_trip(&tile, &loaded)?,
            other => {
                prop_assert!(false, "the re-fused entry must verify: {other:?}");
            }
        }
    }
}

/// How a run can end (需求 4.8 / 12.8 / 13.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EndingPath {
    /// Fused, cached, read back.
    Success,
    /// The cache could not be written; the run continues on the in-memory tile.
    Degraded,
    /// The selection was rejected before anything was written.
    Rejected,
    /// The run was cancelled while a tile was leased.
    Cancelled,
    /// A diagnostics bundle was written.
    DiagnosticsWrite,
}

fn arb_ending_path() -> impl Strategy<Value = EndingPath> {
    prop_oneof![
        Just(EndingPath::Success),
        Just(EndingPath::Degraded),
        Just(EndingPath::Rejected),
        Just(EndingPath::Cancelled),
        Just(EndingPath::DiagnosticsWrite),
    ]
}

/// `(file name, length, modified)` of every entry of `directory`, sorted.
///
/// Compared before and after a run so a created, deleted or rewritten file in
/// the Source_RAW directory fails the property (需求 4.8).
fn directory_fingerprint(directory: &Path) -> Vec<(String, u64, Option<SystemTime>)> {
    let mut entries = fs::read_dir(directory)
        .expect("the source directory is readable")
        .flatten()
        .map(|entry| {
            let metadata = entry.metadata().expect("metadata");
            (
                entry.file_name().to_string_lossy().to_string(),
                metadata.len(),
                metadata.modified().ok(),
            )
        })
        .collect::<Vec<_>>();
    entries.sort();
    entries
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 24: 对于任意输入集合与
    // 任意结束路径（成功、降级、拒绝、取消、诊断写入），运行结束后每个 Source_RAW 的绝对
    // 路径存在性与 SHA-256 都与运行开始前一致，且 Source_RAW 所在目录中没有被创建、修改或
    // 删除任何文件。
    //
    // 五条结束路径都用真实临时文件驱动真实的缓存写入、租约与报告序列化；目录指纹
    // （文件名、长度、修改时间）在运行前后逐项对照，所以任何往源目录里写东西的代码路径
    // 都会在这里变红。
    //
    // **Validates: Requirements 4.8, 12.8, 13.4**
    #[test]
    fn source_raw_files_survive_every_ending_path(
        tile in arb_virtual_tile_in(CACHE_PROPERTY_MIN_SIDE, 24),
        contents in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..1024), 1..=4),
        ending in arb_ending_path(),
        probe in any::<usize>(),
    ) {
        let sources = TempDir::new().expect("temp dir");
        let mut pairs = Vec::new();
        for (index, content) in contents.iter().enumerate() {
            let path = sources.path().join(format!("DSC_{:04}.NEF", 3680 + index));
            fs::write(&path, content).expect("writing a source fixture");
            pairs.push(path);
        }
        // Before: existence, digest and the directory listing.
        let before_digests = pairs
            .iter()
            .map(|path| {
                (
                    path.clone(),
                    virtual_tile::source_file_sha256(path).expect("the digest succeeds"),
                )
            })
            .collect::<Vec<_>>();
        let before_listing = directory_fingerprint(sources.path());

        let inputs = CacheKeyInputs::new(
            STACK_PIPELINE_VERSION,
            &before_digests
                .iter()
                .map(|(path, digest)| (path.to_string_lossy().to_string(), *digest))
                .collect::<Vec<_>>(),
        );
        let cache = TempDir::new().expect("temp dir");

        // Run the ending path.  Each one has to really happen, or the property
        // would hold for a run that never touched anything.
        match ending {
            EndingPath::Success => {
                let store = VirtualTileStore::new(cache.path());
                prop_assert!(
                    matches!(store.store(&inputs, &tile), StoreOutcome::Written { .. }),
                    "a temp dir must be writable"
                );
                prop_assert!(matches!(store.load(&inputs), CacheLookup::Hit(_)));
            }
            EndingPath::Degraded => {
                // A regular file where the cache root must be a directory.
                fs::write(cache.path().join("stack-virtual-tiles"), b"blocked")
                    .expect("writing the blocker");
                let store = VirtualTileStore::new(cache.path());
                match store.store(&inputs, &tile) {
                    StoreOutcome::Unavailable { reason, .. } => {
                        prop_assert_eq!(reason, degradation::CACHE_WRITE_UNAVAILABLE);
                    }
                    other => {
                        prop_assert!(false, "an unwritable cache must degrade: {other:?}");
                    }
                }
                // 需求 4.11: the run continues on the in-memory tile.
                prop_assert_eq!(
                    tile.coverage.covered_pixels(),
                    tile.ownership.assigned_pixels()
                );
            }
            EndingPath::Rejected => {
                let mut ledger = DegradationLedger::new();
                let (reason, _) = FAILURE_REASONS[probe % FAILURE_REASONS.len()];
                ledger.record(reason, Value::Null);
                prop_assert_eq!(ledger.entries().len(), 1);
            }
            EndingPath::Cancelled => {
                let store = VirtualTileStore::new(cache.path());
                // A tile is resident and the run is abandoned mid-way.
                let lease = store
                    .lease(tile.station_index, || Ok(tile.clone()))
                    .expect("the first lease fits");
                prop_assert_eq!(lease.station_index(), tile.station_index);
                drop(lease);
                prop_assert_eq!(store.active_virtual_tile_leases(), 0);
            }
            EndingPath::DiagnosticsWrite => {
                let report = super::report::StackReport::default();
                let document = serde_json::to_vec_pretty(&report).expect("the report serializes");
                let path = cache.path().join("stack-report.json");
                fs::write(&path, &document).expect("writing the diagnostics bundle");
                prop_assert!(path.exists() && !document.is_empty());
            }
        }

        // After: the same existence, the same digest, the same directory.
        for (path, before) in &before_digests {
            prop_assert!(
                path.exists(),
                "{} disappeared on the {:?} path",
                path.display(),
                ending
            );
            let after = virtual_tile::source_file_sha256(path)
                .expect("the source is still readable");
            prop_assert_eq!(
                &after,
                before,
                "{} changed on the {:?} path",
                path.display(),
                ending
            );
        }
        prop_assert_eq!(
            directory_fingerprint(sources.path()),
            before_listing,
            "the Source_RAW directory changed on the {:?} path",
            ending
        );
    }
}

/// One step of the Virtual_Tile access sequence of Property 88.
#[derive(Debug, Clone, Copy)]
enum LeaseStep {
    /// Lease the tile of station `index % station_count`.
    Acquire(usize),
    /// Drop the `index % held` lease currently held.
    Release(usize),
}

fn arb_lease_step() -> impl Strategy<Value = LeaseStep> {
    prop_oneof![
        3 => any::<usize>().prop_map(LeaseStep::Acquire),
        2 => any::<usize>().prop_map(LeaseStep::Release),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, ..ProptestConfig::default() })]

    // Feature: layered-camera-group-focus-stitching, Property 88: 对于任意
    // Capture_Station 数量与任意 Virtual_Tile 访问序列，任意时刻同时常驻内存的完整尺寸
    // Virtual_Tile 数量不超过 2 个，且报告记录的最大同时常驻数量等于实际观测到的最大值。
    //
    // 既有单元测试走的是几条手写的租约序列；这里的机位数量与取/放序列是随机的，并且在
    // 每一步之后检查上限。像素规模与这条属性无关，所以用最小合法瓦片（需求 14.4 约束的是
    // 常驻数量，不是瓦片内容）。
    //
    // TODO(task 11.x): `resources.max_resident_virtual_tiles` 的报告写入点随
    // Stack_Report 资源段落落地；这里断言的是 store 观测到的峰值，即该字段的数据来源。
    //
    // **Validates: Requirements 14.4**
    #[test]
    fn at_most_two_virtual_tiles_are_resident(
        station_count in 1usize..=6,
        steps in prop::collection::vec(arb_lease_step(), 1..24),
    ) {
        let directory = TempDir::new().expect("temp dir");
        let store = VirtualTileStore::new(directory.path());
        let syntheses = std::cell::Cell::new(0usize);
        let mut held: Vec<virtual_tile::TileLease> = Vec::new();
        let mut observed_peak = 0usize;
        let mut refusals = 0usize;

        for step in &steps {
            match *step {
                LeaseStep::Acquire(slot) => {
                    let station = slot % station_count;
                    let before = syntheses.get();
                    match store.lease(station, || {
                        syntheses.set(syntheses.get() + 1);
                        Ok(minimal_virtual_tile(station))
                    }) {
                        Ok(lease) => {
                            prop_assert_eq!(lease.station_index(), station);
                            prop_assert_eq!(lease.tile().station_index, station);
                            held.push(lease);
                        }
                        Err(LeaseError::ResidencyExhausted { limit, held: live }) => {
                            prop_assert_eq!(limit, MAX_RESIDENT_VIRTUAL_TILES);
                            prop_assert_eq!(
                                live,
                                MAX_RESIDENT_VIRTUAL_TILES,
                                "a lease may only be refused when every slot is held"
                            );
                            prop_assert_eq!(
                                syntheses.get(),
                                before,
                                "a refused lease must not synthesise the tile it cannot hold"
                            );
                            refusals += 1;
                        }
                        Err(other) => {
                            prop_assert!(false, "unexpected lease failure: {other}");
                        }
                    }
                }
                LeaseStep::Release(slot) => {
                    if !held.is_empty() {
                        let index = slot % held.len();
                        held.remove(index);
                    }
                }
            }
            // The ceiling holds after every single step, not just at the end.
            let resident = store.resident_virtual_tiles();
            prop_assert!(
                resident <= MAX_RESIDENT_VIRTUAL_TILES,
                "{resident} resident tile(s) exceed the ceiling of \
                 {MAX_RESIDENT_VIRTUAL_TILES}"
            );
            let live_stations = held
                .iter()
                .map(|lease| lease.station_index())
                .collect::<BTreeSet<_>>();
            prop_assert!(
                live_stations.len() <= MAX_RESIDENT_VIRTUAL_TILES,
                "{} distinct station(s) are leased at once",
                live_stations.len()
            );
            observed_peak = observed_peak.max(resident);
        }

        // 报告记录的最大同时常驻数量等于实际观测到的最大值.
        prop_assert_eq!(
            store.max_resident_virtual_tiles(),
            observed_peak,
            "the recorded peak must be the observed peak"
        );

        // A deterministic probe on top of the random sequence, so every case
        // really reaches the ceiling instead of only the lucky ones.
        drop(held);
        let mut probe = Vec::new();
        for station in 0..MAX_RESIDENT_VIRTUAL_TILES {
            probe.push(
                store
                    .lease(100 + station, || Ok(minimal_virtual_tile(100 + station)))
                    .expect("the first two leases fit"),
            );
        }
        prop_assert_eq!(store.resident_virtual_tiles(), MAX_RESIDENT_VIRTUAL_TILES);
        let before = syntheses.get();
        match store.lease(200, || {
            syntheses.set(syntheses.get() + 1);
            Ok(minimal_virtual_tile(200))
        }) {
            Err(LeaseError::ResidencyExhausted { limit, held: live }) => {
                prop_assert_eq!((limit, live), (MAX_RESIDENT_VIRTUAL_TILES, MAX_RESIDENT_VIRTUAL_TILES));
                prop_assert_eq!(syntheses.get(), before);
                refusals += 1;
            }
            Ok(_) => {
                prop_assert!(false, "a third resident tile must never be admitted");
            }
            Err(other) => {
                prop_assert!(false, "unexpected lease failure: {other}");
            }
        }
        prop_assert!(refusals >= 1, "the ceiling probe must have been refused");
        prop_assert_eq!(
            store.max_resident_virtual_tiles(),
            MAX_RESIDENT_VIRTUAL_TILES
        );
        drop(probe);
    }
}
// ---------------------------------------------------------------------------
