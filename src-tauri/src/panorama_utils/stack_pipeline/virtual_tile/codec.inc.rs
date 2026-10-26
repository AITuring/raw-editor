// ---------------------------------------------------------------------------
// Payload encoding
// ---------------------------------------------------------------------------

fn encode_f32_plane(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn decode_f32_plane(bytes: &[u8], elements: u64) -> Option<Vec<f32>> {
    let elements = usize::try_from(elements).ok()?;
    if bytes.len() != elements * 4 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|chunk| {
                f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            })
            .collect(),
    )
}

fn encode_u16_plane(values: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_u16_plane(bytes: &[u8], elements: u64) -> Option<Vec<u16>> {
    let elements = usize::try_from(elements).ok()?;
    if bytes.len() != elements * 2 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect(),
    )
}

/// `f16` little endian, one value per pixel.  This is the one plane the design
/// stores narrower than it is held in memory; the stored value round trips
/// exactly, and confidence feeds no threshold that `f16` spacing can move.
fn encode_f16_plane(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&f16::from_f32(*value).to_bits().to_le_bytes());
    }
    bytes
}

fn decode_f16_plane(bytes: &[u8], elements: u64) -> Option<Vec<f32>> {
    let elements = usize::try_from(elements).ok()?;
    if bytes.len() != elements * 2 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
    )
}

/// Coverage as one bit per pixel, most significant bit first inside a byte, so
/// the plane is independent of the host byte order.
fn encode_coverage_bits(covered: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0u8; covered.len().div_ceil(8)];
    for (index, &value) in covered.iter().enumerate() {
        if value > 0 {
            bytes[index / 8] |= 0x80 >> (index % 8);
        }
    }
    bytes
}

fn decode_coverage_bits(bytes: &[u8], elements: u64) -> Option<Vec<u8>> {
    let elements = usize::try_from(elements).ok()?;
    if bytes.len() != elements.div_ceil(8) {
        return None;
    }
    let mut covered = vec![0u8; elements];
    for (index, slot) in covered.iter_mut().enumerate() {
        if bytes[index / 8] & (0x80 >> (index % 8)) != 0 {
            *slot = u8::MAX;
        }
    }
    Some(covered)
}

// ---------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------

/// Write `plane` zstd compressed and `fsync` it.  Returns the file size.
fn write_payload(path: &Path, plane: &[u8]) -> Result<u64, String> {
    let compressed = zstd::bulk::compress(plane, ZSTD_LEVEL)
        .map_err(|error| format!("Failed to compress {}: {error}", path.display()))?;
    write_file_sync(path, &compressed)
}

/// Read a zstd payload back to its raw plane.
fn read_payload(path: &Path) -> Result<Vec<u8>, String> {
    let compressed =
        fs::read(path).map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    // `zstd` records the decompressed size in the frame header, so no capacity
    // has to be guessed; a truncated frame fails here rather than silently
    // returning a short plane.
    zstd::stream::decode_all(compressed.as_slice())
        .map_err(|error| format!("Failed to decompress {}: {error}", path.display()))
}

fn write_file_sync(path: &Path, bytes: &[u8]) -> Result<u64, String> {
    let mut file = File::create(path)
        .map_err(|error| format!("Failed to create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("Failed to write {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("Failed to flush {}: {error}", path.display()))?;
    Ok(bytes.len() as u64)
}

/// `fsync` a directory so a rename that follows is durable.  Best effort:
/// Windows cannot open a directory as a file, and a missing barrier costs
/// durability across a power loss, not correctness within a run.
fn sync_directory(path: &Path) {
    #[cfg(unix)]
    if let Ok(handle) = File::open(path) {
        let _ = handle.sync_all();
    }
    #[cfg(not(unix))]
    let _ = path;
}

/// Total file size below `path`, used when the index lost track of an entry.
fn directory_bytes(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .filter_map(|entry| entry.metadata().ok())
        .filter(|metadata| metadata.is_file())
        .fold(0u64, |total, metadata| total.saturating_add(metadata.len()))
}
