//! Lossless plane backing with the complete sampling/owner metadata retained.
//! Deliberately no serialization of source sampling or geometry: their original
//! immutable snapshots are cloned and charged to the cache's RAM budget.
use super::{ConfidenceMap, CoverageMask, FocusStackTileMasks, FocusStackTileRender, OwnershipMap};
use crate::panorama_utils::stack_pipeline::cache_backing::{BackingFile, read_plane, write_plane};
use image::Rgb32FImage;
use std::io;

pub(crate) struct FocusStackBacking {
    template: FocusStackTileRender,
    width: u32,
    height: u32,
    pub(crate) file: BackingFile,
}
impl FocusStackBacking {
    pub(crate) fn disk_bytes(render: &FocusStackTileRender) -> usize {
        let mut bytes = render.image.as_raw().len() * 4;
        if let Some(m) = &render.masks {
            bytes += m.coverage.covered.len() + m.ownership.owners.len() * 2 + m.textured.len();
            if let ConfidenceMap::PerPixel { values, .. } = &m.confidence {
                bytes += values.len() * 4;
            }
        }
        bytes
    }
    pub(crate) fn write(render: &FocusStackTileRender) -> io::Result<Self> {
        let file = BackingFile::write(Self::disk_bytes(render), |w| {
            write_plane(w, render.image.as_raw())?;
            if let Some(m) = &render.masks {
                write_plane(w, &m.coverage.covered)?;
                write_plane(w, &m.ownership.owners)?;
                if let ConfidenceMap::PerPixel { values, .. } = &m.confidence {
                    write_plane(w, values)?;
                }
                write_plane(w, &m.textured)?;
            }
            Ok(())
        })?;
        // Do not clone full masks then clear: that temporarily doubles large planes.
        let masks = render.masks.as_ref().map(|m| FocusStackTileMasks {
            source_refinements: m.source_refinements.clone(),
            coverage: CoverageMask {
                width: m.coverage.width,
                height: m.coverage.height,
                covered: Vec::new(),
            },
            ownership: OwnershipMap {
                width: m.ownership.width,
                height: m.ownership.height,
                owners: Vec::new(),
                legend: m.ownership.legend.clone(),
            },
            confidence: match &m.confidence {
                ConfidenceMap::Uniform { .. } => m.confidence.clone(),
                ConfidenceMap::PerPixel { width, height, .. } => ConfidenceMap::PerPixel {
                    width: *width,
                    height: *height,
                    values: Vec::new(),
                },
            },
            textured: Vec::new(),
            sharpness_evidence: m.sharpness_evidence.clone(),
            native_displacements: m.native_displacements.clone(),
            fusion: m.fusion.clone(),
        });
        Ok(Self {
            template: FocusStackTileRender {
                image: Rgb32FImage::new(0, 0),
                masks,
                sampling_origin: render.sampling_origin,
            },
            width: render.image.width(),
            height: render.image.height(),
            file,
        })
    }
    pub(crate) fn retained_bytes(&self) -> usize {
        self.template.retained_bytes() + 1024
    }
    pub(crate) fn read(&self) -> io::Result<FocusStackTileRender> {
        self.file.read(|r| {
            let mut result = self.template.clone();
            result.image = Rgb32FImage::from_raw(
                self.width,
                self.height,
                read_plane(r, self.width as usize * self.height as usize * 3)?,
            )
            .ok_or_else(|| io::Error::other("invalid backed image dimensions"))?;
            if let Some(m) = &mut result.masks {
                m.coverage.covered =
                    read_plane(r, m.coverage.width as usize * m.coverage.height as usize)?;
                m.ownership.owners =
                    read_plane(r, m.ownership.width as usize * m.ownership.height as usize)?;
                if let ConfidenceMap::PerPixel {
                    width,
                    height,
                    values,
                } = &mut m.confidence
                {
                    *values = read_plane(r, *width as usize * *height as usize)?;
                }
                m.textured = read_plane(r, self.width as usize * self.height as usize)?;
            }
            Ok(result)
        })
    }
}
