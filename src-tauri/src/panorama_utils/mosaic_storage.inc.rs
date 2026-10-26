struct StreamingMosaicStore {
    temp_dir: tempfile::TempDir,
    width: u32,
    height: u32,
    tile_size: u32,
    tile_columns: u32,
    tile_rows: u32,
}

impl StreamingMosaicStore {
    fn new(width: u32, height: u32) -> Result<Self, String> {
        let temp_dir = tempfile::Builder::new()
            .prefix("raw-editor-focus-stack-")
            .tempdir_in(std::env::temp_dir())
            .map_err(|error| format!("Could not create the tiled focus canvas: {error}"))?;
        Ok(Self {
            temp_dir,
            width,
            height,
            tile_size: STREAMING_TILE_SIZE,
            tile_columns: width.div_ceil(STREAMING_TILE_SIZE),
            tile_rows: height.div_ceil(STREAMING_TILE_SIZE),
        })
    }

    fn tile_extent(&self, column: u32, row: u32) -> (u32, u32, u32, u32) {
        let left = column * self.tile_size;
        let top = row * self.tile_size;
        let width = self.tile_size.min(self.width.saturating_sub(left));
        let height = self.tile_size.min(self.height.saturating_sub(top));
        (left, top, width, height)
    }

    fn tile_path(&self, column: u32, row: u32, suffix: &str) -> PathBuf {
        self.temp_dir
            .path()
            .join(format!("tile-{column}-{row}.{suffix}"))
    }

    fn read_f32_tile(
        &self,
        column: u32,
        row: u32,
        width: u32,
        height: u32,
    ) -> Result<Rgb32FImage, String> {
        let path = self.tile_path(column, row, "rgb");
        if !path.exists() {
            return Ok(Rgb32FImage::new(width, height));
        }
        let bytes = std::fs::read(&path)
            .map_err(|error| format!("Could not read focus tile {}: {error}", path.display()))?;
        let expected = width as usize * height as usize * 3;
        if bytes.len() != expected * std::mem::size_of::<f32>() {
            return Err(format!(
                "Focus tile {} has {} bytes; expected {}",
                path.display(),
                bytes.len(),
                expected * std::mem::size_of::<f32>()
            ));
        }
        let mut values = Vec::with_capacity(expected);
        for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
            values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Rgb32FImage::from_raw(width, height, values)
            .ok_or_else(|| format!("Could not decode focus tile {}", path.display()))
    }

    fn read_mask_tile(&self, column: u32, row: u32, width: u32, height: u32) -> Vec<u8> {
        let path = self.tile_path(column, row, "mask");
        let expected = width as usize * height as usize;
        match std::fs::read(&path) {
            Ok(bytes) if bytes.len() == expected => bytes,
            _ => vec![0; expected],
        }
    }

    fn read_owner_tile(&self, column: u32, row: u32, width: u32, height: u32) -> Vec<u8> {
        let path = self.tile_path(column, row, "owner");
        let expected = width as usize * height as usize;
        match std::fs::read(&path) {
            Ok(bytes) if bytes.len() == expected => bytes,
            _ => vec![0; expected],
        }
    }

    fn load_owner_tile(&self, column: u32, row: u32) -> GrayImage {
        let (_, _, width, height) = self.tile_extent(column, row);
        GrayImage::from_raw(
            width,
            height,
            self.read_owner_tile(column, row, width, height),
        )
        .expect("focus tile owner dimensions")
    }

    fn save_owner_tile(&self, column: u32, row: u32, owner: &GrayImage) -> Result<(), String> {
        let path = self.tile_path(column, row, "owner");
        std::fs::write(&path, owner.as_raw()).map_err(|error| {
            format!(
                "Could not write focus owner tile {}: {error}",
                path.display()
            )
        })
    }

    fn load_tile(&self, column: u32, row: u32) -> Result<(Rgb32FImage, GrayImage), String> {
        let (_, _, width, height) = self.tile_extent(column, row);
        let rgb = self.read_f32_tile(column, row, width, height)?;
        let mask = GrayImage::from_raw(
            width,
            height,
            self.read_mask_tile(column, row, width, height),
        )
        .expect("focus tile mask dimensions");
        Ok((rgb, mask))
    }

    fn save_tile(
        &self,
        column: u32,
        row: u32,
        rgb: &Rgb32FImage,
        mask: &GrayImage,
    ) -> Result<(), String> {
        let rgb_path = self.tile_path(column, row, "rgb");
        let mut rgb_bytes = Vec::with_capacity(rgb.as_raw().len() * std::mem::size_of::<f32>());
        for &value in rgb.as_raw() {
            rgb_bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&rgb_path, rgb_bytes).map_err(|error| {
            format!("Could not write focus tile {}: {error}", rgb_path.display())
        })?;
        let mask_path = self.tile_path(column, row, "mask");
        std::fs::write(&mask_path, mask.as_raw()).map_err(|error| {
            format!(
                "Could not write focus tile {}: {error}",
                mask_path.display()
            )
        })?;
        Ok(())
    }

    fn analysis_region(
        &self,
        left: u32,
        top: u32,
        scale: f64,
        width: u32,
        height: u32,
    ) -> Result<(Rgb32FImage, GrayImage, GrayImage), String> {
        let mut image = Rgb32FImage::new(width, height);
        let mut mask = GrayImage::new(width, height);
        let mut owner = GrayImage::new(width, height);
        if width == 0 || height == 0 || !scale.is_finite() || scale <= 0.0 {
            return Ok((image, mask, owner));
        }
        let right = (left as f64 + width.saturating_sub(1) as f64 * scale)
            .floor()
            .clamp(0.0, self.width.saturating_sub(1) as f64) as u32;
        let bottom = (top as f64 + height.saturating_sub(1) as f64 * scale)
            .floor()
            .clamp(0.0, self.height.saturating_sub(1) as f64) as u32;
        for tile_row in top / self.tile_size..=bottom / self.tile_size {
            for tile_column in left / self.tile_size..=right / self.tile_size {
                let (tile_left, tile_top, tile_width, tile_height) =
                    self.tile_extent(tile_column, tile_row);
                let (tile, tile_mask) = self.load_tile(tile_column, tile_row)?;
                let tile_owner = self.load_owner_tile(tile_column, tile_row);
                let x_start = (((tile_left as f64 - left as f64) / scale).ceil() as i64)
                    .clamp(0, width as i64) as u32;
                let x_end = ((((tile_left + tile_width) as f64 - left as f64) / scale).ceil()
                    as i64)
                    .clamp(0, width as i64) as u32;
                let y_start = (((tile_top as f64 - top as f64) / scale).ceil() as i64)
                    .clamp(0, height as i64) as u32;
                let y_end = ((((tile_top + tile_height) as f64 - top as f64) / scale).ceil() as i64)
                    .clamp(0, height as i64) as u32;
                for y in y_start..y_end {
                    let global_y = (top as f64 + y as f64 * scale).floor() as u32;
                    if global_y < tile_top || global_y >= tile_top + tile_height {
                        continue;
                    }
                    for x in x_start..x_end {
                        let global_x = (left as f64 + x as f64 * scale).floor() as u32;
                        if global_x < tile_left || global_x >= tile_left + tile_width {
                            continue;
                        }
                        let tile_x = global_x - tile_left;
                        let tile_y = global_y - tile_top;
                        if tile_mask.get_pixel(tile_x, tile_y)[0] == 0 {
                            continue;
                        }
                        image.put_pixel(x, y, *tile.get_pixel(tile_x, tile_y));
                        mask.put_pixel(x, y, Luma([255]));
                        owner.put_pixel(x, y, *tile_owner.get_pixel(tile_x, tile_y));
                    }
                }
            }
        }
        Ok((image, mask, owner))
    }

    fn covered_bounds(&self) -> Result<Option<(u32, u32, u32, u32)>, String> {
        let mut bounds: Option<(u32, u32, u32, u32)> = None;
        for row in 0..self.tile_rows {
            for column in 0..self.tile_columns {
                let (left, top, width, height) = self.tile_extent(column, row);
                let mask = self.read_mask_tile(column, row, width, height);
                for y in 0..height {
                    let row_start = y as usize * width as usize;
                    for x in 0..width {
                        if mask[row_start + x as usize] == 0 {
                            continue;
                        }
                        let gx = left + x;
                        let gy = top + y;
                        bounds = Some(match bounds {
                            Some((min_x, min_y, max_x, max_y)) => {
                                (min_x.min(gx), min_y.min(gy), max_x.max(gx), max_y.max(gy))
                            }
                            None => (gx, gy, gx, gy),
                        });
                    }
                }
            }
        }
        Ok(bounds.map(|(min_x, min_y, max_x, max_y)| {
            (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
        }))
    }

    fn materialize(&self, crop: (u32, u32, u32, u32)) -> Result<Rgb32FImage, String> {
        let (crop_left, crop_top, crop_width, crop_height) = crop;
        let mut output = Rgb32FImage::new(crop_width, crop_height);
        for tile_row in 0..self.tile_rows {
            for tile_column in 0..self.tile_columns {
                let (tile_left, tile_top, tile_width, tile_height) =
                    self.tile_extent(tile_column, tile_row);
                let x_start = crop_left.max(tile_left);
                let y_start = crop_top.max(tile_top);
                let x_end = (crop_left + crop_width).min(tile_left + tile_width);
                let y_end = (crop_top + crop_height).min(tile_top + tile_height);
                if x_start >= x_end || y_start >= y_end {
                    continue;
                }
                let (tile, _) = self.load_tile(tile_column, tile_row)?;
                for y in y_start..y_end {
                    let source_y = (y - tile_top) as usize;
                    let destination_y = (y - crop_top) as usize;
                    let source_start =
                        ((x_start - tile_left) as usize * 3) + source_y * tile_width as usize * 3;
                    let source_end = source_start + (x_end - x_start) as usize * 3;
                    let destination_start = (x_start - crop_left) as usize * 3
                        + destination_y * crop_width as usize * 3;
                    let destination_end = destination_start + (x_end - x_start) as usize * 3;
                    output.as_mut()[destination_start..destination_end]
                        .copy_from_slice(&tile.as_raw()[source_start..source_end]);
                }
            }
        }
        Ok(output)
    }
}

