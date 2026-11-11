//! Private, run-local lossless backing, using anonymous temporary files.
//! Sizes and shapes come from live trusted metadata, never from file contents.
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};

pub(crate) struct BackingFile {
    file: File,
    pub(crate) bytes: usize,
    digest: [u8; 32],
}

struct HashWriter<'a> {
    file: &'a mut std::fs::File,
    hash: Sha256,
    bytes: usize,
    limit: usize,
}
impl Write for HashWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.len() > self.limit.saturating_sub(self.bytes) {
            return Err(io::Error::other("backing write exceeds reserved budget"));
        }
        let n = self.file.write(buf)?;
        self.hash.update(&buf[..n]);
        self.bytes += n;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}
struct HashReader {
    file: std::fs::File,
    hash: Sha256,
    bytes: usize,
}
impl Read for HashReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.file.read(buf)?;
        self.hash.update(&buf[..n]);
        self.bytes += n;
        Ok(n)
    }
}
impl BackingFile {
    pub(crate) fn write(
        bytes: usize,
        save: impl FnOnce(&mut dyn Write) -> io::Result<()>,
    ) -> io::Result<Self> {
        let mut file = tempfile::tempfile()?;
        let mut writer = HashWriter {
            file: &mut file,
            hash: Sha256::new(),
            bytes: 0,
            limit: bytes,
        };
        save(&mut writer)?;
        writer.flush()?;
        if writer.bytes != bytes {
            return Err(io::Error::other("backing write length mismatch"));
        }
        let digest = writer.hash.finalize().into();
        Ok(Self {
            file,
            bytes,
            digest,
        })
    }
    // Cache access is serialized; cloned handles may share the seek cursor.
    pub(crate) fn read<T>(
        &self,
        load: impl FnOnce(&mut dyn Read) -> io::Result<T>,
    ) -> io::Result<T> {
        let mut file = self.file.try_clone()?;
        if file.metadata()?.len() != self.bytes as u64 {
            return Err(io::Error::other("backing length mismatch"));
        }
        file.seek(SeekFrom::Start(0))?;
        let mut reader = HashReader {
            file,
            hash: Sha256::new(),
            bytes: 0,
        };
        let result = load(&mut reader)?;
        let digest: [u8; 32] = reader.hash.finalize().into();
        if reader.bytes != self.bytes || digest != self.digest {
            return Err(io::Error::other("backing checksum mismatch"));
        }
        Ok(result)
    }
    #[cfg(test)]
    pub(crate) fn corrupt(&self) {
        let mut file = self.file.try_clone().unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&[0xAF]).unwrap();
    }
}
pub(crate) fn write_plane<T: bytemuck::Pod>(
    writer: &mut dyn Write,
    values: &[T],
) -> io::Result<()> {
    writer.write_all(bytemuck::cast_slice(values))
}
pub(crate) fn read_plane<T: bytemuck::Pod + bytemuck::Zeroable>(
    reader: &mut dyn Read,
    len: usize,
) -> io::Result<Vec<T>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(io::Error::other)?;
    values.resize(len, T::zeroed());
    reader.read_exact(bytemuck::cast_slice_mut(&mut values))?;
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bits_corruption_bounds_and_cleanup() {
        let values = [0.0f32, -0.0, f32::from_bits(0x7fc01234), f32::INFINITY];
        let file = BackingFile::write(16, |w| write_plane(w, &values)).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(file.file.metadata().unwrap().nlink(), 0);
        }
        let decoded: Vec<f32> = file.read(|r| read_plane(r, 4)).unwrap();
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&values),
            bytemuck::cast_slice::<_, u8>(&decoded)
        );
        file.corrupt();
        assert!(file.read::<Vec<f32>>(|r| read_plane(r, 4)).is_err());
        drop(file);
        assert!(
            BackingFile::write(4, |_| Err(io::Error::other("injected write failure"))).is_err()
        );
        assert!(BackingFile::write(15, |w| write_plane(w, &values)).is_err());
        assert!(BackingFile::write(17, |w| write_plane(w, &values)).is_err());
    }
}
