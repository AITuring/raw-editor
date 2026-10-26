/// Build the low-frequency field on one sixteenth of the output grid.
pub(crate) const TONE_GRID_DIVISOR: u32 = 16;
/// Gaussian sigma in analysis-grid pixels (requirement 9.2/11.5).
pub(crate) const TONE_LOW_FREQUENCY_SIGMA: f32 = 4.0;
/// Minimum retained pair samples after robust outlier rejection.
pub(crate) const TONE_MIN_SAMPLES: usize = 1024;
/// Affine gain and offset limits shared by the stack Tone_Harmonizer.
pub(crate) const TONE_MIN_GAIN: f32 = 1.0 / 1.25;
pub(crate) const TONE_MAX_GAIN: f32 = 1.25;
pub(crate) const TONE_MAX_ABS_OFFSET: f32 = 0.02;
/// Width of the band on each side of an Owner_Region boundary (需求 9.8).
pub(crate) const TONE_BOUNDARY_BAND_PX: u32 = 16;
/// Boundary windows whose band means are compared separately, so a local
/// tone step is located instead of being averaged over a whole boundary.
pub(crate) const TONE_BOUNDARY_WINDOW_PX: u32 = 256;
/// Normalised luminance range of a usable tone sample on both sides (需求 9.1).
pub(crate) const TONE_MIN_SAMPLE_LUMINANCE: f32 = 0.02;
pub(crate) const TONE_MAX_SAMPLE_LUMINANCE: f32 = 0.98;
/// `ToneTile::validity` of a cell the tile covers entirely: tone evidence.
pub(crate) const TONE_CELL_SAMPLE: u8 = 255;
/// `ToneTile::validity` of a cell the tile covers in part: its covered-only
/// field is corrected with the tile, but it is no tone evidence.
pub(crate) const TONE_CELL_PARTIAL: u8 = 128;
/// Log-gain scatter at which a relation's joint-solve weight halves, and the
/// Huber scale floor of its residual (the values of `photometric.rs`).
const TONE_RELATION_SCATTER_SCALE: f64 = 0.04;
const TONE_RELATION_HUBER_LOG: f64 = 0.025;

// Tone solving is reached from the compositor's free functions, which do not
// carry a mutable Stack_Report reference.  Keep the same run-scoped sink used
// by Intra_Station and Focus_Fuser; report publication takes one snapshot at
// the terminating boundary.
static RUN_RECORDS: Mutex<Vec<ToneTileRecord>> = Mutex::new(Vec::new());
static RUN_REPORT: Mutex<Option<ToneReport>> = Mutex::new(None);
static RUN_TOPOLOGY: Mutex<Option<ToneStationTopology>> = Mutex::new(None);

/// The station context the owner compositor needs for tone evidence: each
/// Virtual_Tile's station index and the run's accepted Station_Relations,
/// both keyed by the tile's `ImageInfo::id`. Only the overlap of an accepted
/// relation yields tone samples (需求 9.1). It is published by the station
/// solve, which runs before the compositor, and like the residual model it is
/// read by the compositor rather than threaded through its signature.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ToneStationTopology {
    /// `(ImageInfo::id, station index)` of every Virtual_Tile.
    pub(crate) stations: Vec<(usize, usize)>,
    /// Accepted Station_Relations as `ImageInfo::id` pairs.
    pub(crate) accepted: Vec<(usize, usize)>,
}

impl ToneStationTopology {
    pub(crate) fn station_of(&self, image_id: usize) -> Option<usize> {
        self.stations
            .iter()
            .find(|(id, _)| *id == image_id)
            .map(|(_, station)| *station)
    }
}

/// Clear the station context at whole-run start. Kept apart from
/// [`reset_run_records`], which a compositor may call at its own start after
/// the station solve published this context.
pub(crate) fn reset_run_topology() {
    *RUN_TOPOLOGY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

pub(crate) fn record_run_topology(topology: ToneStationTopology) {
    *RUN_TOPOLOGY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(topology);
}

pub(crate) fn run_topology_snapshot() -> Option<ToneStationTopology> {
    RUN_TOPOLOGY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

pub(crate) fn reset_run_records() {
    RUN_RECORDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
    *RUN_REPORT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
}

pub(crate) fn record_run_record(record: ToneTileRecord) {
    RUN_RECORDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push(record);
}

pub(crate) fn run_records_snapshot() -> Vec<ToneTileRecord> {
    RUN_RECORDS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

pub(crate) fn record_run_report(report: ToneReport) {
    *RUN_REPORT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(report);
}

pub(crate) fn run_report_snapshot() -> Option<ToneReport> {
    RUN_REPORT
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

/// One source's low-frequency field in output world coordinates.  `low` and
/// `validity` stay at the one-sixteenth analysis resolution; no full-size
/// low-frequency image is allocated by the harmoniser.
#[derive(Debug, Clone)]
pub(crate) struct ToneTile {
    pub(crate) station_index: usize,
    pub(crate) owner_id: u16,
    pub(crate) low: Rgb32FImage,
    pub(crate) validity: GrayImage,
    pub(crate) world_origin: (f64, f64),
    pub(crate) world_size: (f64, f64),
    pub(crate) world_stride: f64,
    /// Unblurred mean of the covered tile pixels in the tile cell under each
    /// analysis cell, and that cell's covered fraction (row-major, the size
    /// of `low`). A pair of tiles is compared through [`PairField`], built
    /// from these over the cells both cover.
    pub(crate) cell_mean: Rgb32FImage,
    pub(crate) cell_coverage: Vec<f32>,
}

/// The low fields of two tiles over their common support only (需求 9.1).
///
/// Both are normalised convolutions of the same doubly covered cells, so they
/// see the same content window and differ by the tiles' tone and not by what
/// either tile covers alone. A tile's own field near its coverage edge
/// averages only its own side, which next to a content gradient separates it
/// from a neighbour whose field is still centred at that position.
pub(crate) struct PairField {
    origin: (f64, f64),
    stride: f64,
    width: usize,
    height: usize,
    first: Vec<[f32; 3]>,
    second: Vec<[f32; 3]>,
    /// Both tiles cover the cell entirely and the common field exists there.
    sample: Vec<bool>,
}

impl PairField {
    pub(crate) fn new(first: &ToneTile, second: &ToneTile) -> Option<Self> {
        let stride = first.world_stride.max(second.world_stride);
        if !stride.is_finite() || stride <= 0.0 {
            return None;
        }
        let origin = (
            first.world_origin.0.max(second.world_origin.0),
            first.world_origin.1.max(second.world_origin.1),
        );
        let end = (
            (first.world_origin.0 + first.world_size.0)
                .min(second.world_origin.0 + second.world_size.0),
            (first.world_origin.1 + first.world_size.1)
                .min(second.world_origin.1 + second.world_size.1),
        );
        if end.0 <= origin.0 || end.1 <= origin.1 {
            return None;
        }
        let width = ((end.0 - origin.0) / stride).ceil() as usize;
        let height = ((end.1 - origin.1) / stride).ceil() as usize;
        let cell = |tile: &ToneTile, x: f64, y: f64| -> Option<([f32; 3], f32)> {
            let cx = ((x - tile.world_origin.0) / tile.world_stride).floor();
            let cy = ((y - tile.world_origin.1) / tile.world_stride).floor();
            if cx < 0.0
                || cy < 0.0
                || cx >= f64::from(tile.cell_mean.width())
                || cy >= f64::from(tile.cell_mean.height())
            {
                return None;
            }
            let (cx, cy) = (cx as u32, cy as u32);
            let coverage = *tile
                .cell_coverage
                .get(cy as usize * tile.cell_mean.width() as usize + cx as usize)?;
            Some((tile.cell_mean.get_pixel(cx, cy).0, coverage))
        };
        let mut weights = vec![[0.0f64; 1]; width * height];
        let mut first_sums = vec![[0.0f64; 3]; width * height];
        let mut second_sums = vec![[0.0f64; 3]; width * height];
        let mut sample = vec![false; width * height];
        for row in 0..height {
            for column in 0..width {
                let x = origin.0 + column as f64 * stride + (stride - 1.0) * 0.5;
                let y = origin.1 + row as f64 * stride + (stride - 1.0) * 0.5;
                let (Some((first_mean, first_coverage)), Some((second_mean, second_coverage))) =
                    (cell(first, x, y), cell(second, x, y))
                else {
                    continue;
                };
                let index = row * width + column;
                let weight = f64::from(first_coverage.min(second_coverage));
                weights[index] = [weight];
                for channel in 0..3 {
                    first_sums[index][channel] = weight * f64::from(first_mean[channel]);
                    second_sums[index][channel] = weight * f64::from(second_mean[channel]);
                }
                sample[index] = first_coverage >= 1.0 && second_coverage >= 1.0;
            }
        }
        let sigma = f64::from(TONE_LOW_FREQUENCY_SIGMA);
        let support = gaussian_cells(&weights, width, height, sigma);
        let normalise = |sums: &[[f64; 3]]| {
            gaussian_cells(sums, width, height, sigma)
                .iter()
                .zip(&support)
                .map(|(sum, support)| {
                    if support[0] > 1e-9 {
                        sum.map(|value| (value / support[0]) as f32)
                    } else {
                        [0.0; 3]
                    }
                })
                .collect::<Vec<_>>()
        };
        let (first_field, second_field) = (normalise(&first_sums), normalise(&second_sums));
        for (index, flag) in sample.iter_mut().enumerate() {
            *flag &= support[index][0] > 1e-9;
        }
        Some(Self {
            origin,
            stride,
            width,
            height,
            first: first_field,
            second: second_field,
            sample,
        })
    }

    /// Both fields at the cell holding a world position, where it is a
    /// doubly and entirely covered cell.
    fn at(&self, world_x: f64, world_y: f64) -> Option<([f32; 3], [f32; 3])> {
        let column = ((world_x - self.origin.0) / self.stride).floor();
        let row = ((world_y - self.origin.1) / self.stride).floor();
        if column < 0.0 || row < 0.0 || column >= self.width as f64 || row >= self.height as f64 {
            return None;
        }
        let index = row as usize * self.width + column as usize;
        self.sample[index].then(|| (self.first[index], self.second[index]))
    }

    /// Every doubly and entirely covered cell whose two fields both lie in
    /// the usable luminance range, `first` as the owner side (需求 9.1).
    pub(crate) fn samples(&self) -> Vec<ToneSample> {
        let usable = |pixel: [f32; 3]| {
            let luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
            (TONE_MIN_SAMPLE_LUMINANCE..=TONE_MAX_SAMPLE_LUMINANCE).contains(&luminance)
        };
        (0..self.sample.len())
            .filter(|&index| {
                self.sample[index] && usable(self.first[index]) && usable(self.second[index])
            })
            .map(|index| ToneSample {
                owner: self.first[index],
                source: self.second[index],
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ToneSample {
    /// Pixel from the already selected owner source.
    pub(crate) owner: [f32; 3],
    /// Pixel from the source whose low-frequency tone is being corrected.
    pub(crate) source: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToneSolveStatus {
    Applied,
    Identity,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToneSolve {
    pub(crate) gain: [f32; 3],
    pub(crate) offset: [f32; 3],
    /// Raw affine coefficients before either bound is applied.
    pub(crate) solved_gain: [f32; 3],
    pub(crate) solved_offset: [f32; 3],
    pub(crate) retained_samples: usize,
    pub(crate) gain_clamped: bool,
    pub(crate) status: ToneSolveStatus,
}

impl ToneSolve {
    pub(crate) fn identity(retained_samples: usize) -> Self {
        Self {
            gain: [1.0; 3],
            offset: [0.0; 3],
            solved_gain: [1.0; 3],
            solved_offset: [0.0; 3],
            retained_samples,
            gain_clamped: false,
            status: ToneSolveStatus::Identity,
        }
    }
}

/// `ln` of a normalised channel value, bounded away from zero so a channel at
/// black gives a large but finite ratio that the MAD check then rejects.
fn log_channel(value: f32) -> f64 {
    f64::from(value.max(1.0 / 65_535.0)).ln()
}

/// Per-channel `ln owner − ln source` of one sample: the log gain that maps
/// the source side of the sample onto its owner side.
fn sample_log_gain(sample: &ToneSample) -> [f64; 3] {
    std::array::from_fn(|channel| {
        log_channel(sample.owner[channel]) - log_channel(sample.source[channel])
    })
}

