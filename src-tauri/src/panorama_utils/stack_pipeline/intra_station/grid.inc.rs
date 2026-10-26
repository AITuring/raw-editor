// Intra_Station_Registrar fragment; included at parent module scope.

/// Accepted control point displacements on the refinement grid (需求 2.5).
///
/// The grid is walked in row-major order, so the 8-neighbourhood a control
/// point is compared against is a deterministic function of the photographs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ControlPointGrid {
    columns: usize,
    rows: usize,
    accepted: Vec<Option<[f64; 2]>>,
}

impl ControlPointGrid {
    pub(crate) fn new(columns: usize, rows: usize) -> Self {
        Self {
            columns,
            rows,
            accepted: vec![None; columns * rows],
        }
    }

    pub(crate) fn accept(&mut self, column: usize, row: usize, displacement: [f64; 2]) {
        if column < self.columns && row < self.rows {
            self.accepted[row * self.columns + column] = Some(displacement);
        }
    }

    pub(crate) fn accepted_count(&self) -> usize {
        self.accepted.iter().filter(|cell| cell.is_some()).count()
    }

    /// The accepted displacement of one control point, `None` when the gates
    /// rejected it or the grid does not hold that position.
    ///
    /// A query, for Property 6: the refinement itself only ever needs the
    /// neighbourhood median and the count.
    #[cfg(test)]
    pub(crate) fn accepted_at(&self, column: usize, row: usize) -> Option<[f64; 2]> {
        (column < self.columns && row < self.rows)
            .then(|| self.accepted[row * self.columns + column])
            .flatten()
    }

    /// Union this grid with another of the same shape.
    ///
    /// 需求 2.8 counts the control point cells that *contain* an accepted local
    /// match, and the refinement runs two spacing passes whose accepted matches
    /// both end up in the same displacement field.  A cell matched by the coarse
    /// pass therefore contains an accepted local match whether or not the fine
    /// pass matched it again, so the coverage is the union over passes and not
    /// whatever the last pass happened to see.
    ///
    /// The 需求 2.5 neighbourhood gate deliberately does *not* use the union: a
    /// coarse-pass displacement has already been applied to the field, so the
    /// fine pass measures what is left and the two are not comparable.
    pub(crate) fn absorb(&mut self, other: &Self) {
        if self.columns != other.columns || self.rows != other.rows {
            return;
        }
        for (cell, incoming) in self.accepted.iter_mut().zip(other.accepted.iter()) {
            if cell.is_none() {
                *cell = *incoming;
            }
        }
    }

    /// Per-axis median of the accepted 8-neighbourhood, or `None` when no
    /// neighbour has been accepted yet.
    pub(crate) fn neighbour_median(&self, column: usize, row: usize) -> Option<[f64; 2]> {
        let mut xs = Vec::with_capacity(8);
        let mut ys = Vec::with_capacity(8);
        for row_offset in -1i64..=1 {
            for column_offset in -1i64..=1 {
                if row_offset == 0 && column_offset == 0 {
                    continue;
                }
                let neighbour_row = row as i64 + row_offset;
                let neighbour_column = column as i64 + column_offset;
                if neighbour_row < 0
                    || neighbour_column < 0
                    || neighbour_row >= self.rows as i64
                    || neighbour_column >= self.columns as i64
                {
                    continue;
                }
                if let Some(displacement) =
                    self.accepted[neighbour_row as usize * self.columns + neighbour_column as usize]
                {
                    xs.push(displacement[0]);
                    ys.push(displacement[1]);
                }
            }
        }
        (!xs.is_empty()).then(|| [median_of(&mut xs), median_of(&mut ys)])
    }
}

/// Deterministic median: `total_cmp` ordering, and the lower of the two middle
/// samples for an even count, so the value never depends on sort stability.
pub(crate) fn median_of(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    values[(values.len() - 1) / 2]
}

