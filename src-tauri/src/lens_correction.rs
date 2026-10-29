use crate::AppState;
use include_dir::{Dir, include_dir};
use tauri::State;
static LENS_DB_DIR: Dir<'_> = include_dir!("$CARGO_MANIFEST_DIR/lensfun_db");

mod stack_calibration;

pub(crate) use stack_calibration::{StackLensFootprint, stack_lens_calibration};

mod lensfun_db;

pub use lensfun_db::{
    Aperture, Calibration, CalibrationElement, DetectedLensProfile, Distortion, Focal,
    LensDatabase, LensDistortionParams, MultiName, Tca, Vignetting, load_lensfun_db,
};

mod lens_profile;

pub use lens_profile::Lens;

mod lens_search;

pub use lens_search::{find_best_lens_match, resolve_lens_params};
use lens_search::{lenses_for_maker, makers_match};

#[tauri::command]
pub fn get_lensfun_makers(state: State<AppState>) -> Result<Vec<String>, String> {
    let db_guard = state
        .lens_db
        .lock()
        .map_err(|e| format!("Lock poisoned: {}", e))?;
    if let Some(db) = &*db_guard {
        let mut makers: Vec<String> = db.lenses.iter().map(|lens| lens.get_maker()).collect();
        makers.sort_unstable();
        makers.dedup();
        Ok(makers)
    } else {
        Err("Lens database not loaded".to_string())
    }
}

#[tauri::command]
pub fn get_lensfun_lenses_for_maker(
    maker: String,
    state: State<AppState>,
) -> Result<Vec<String>, String> {
    let db_guard = state
        .lens_db
        .lock()
        .map_err(|e| format!("Lock poisoned: {}", e))?;
    if let Some(db) = &*db_guard {
        let maker_lenses = lenses_for_maker(db, &maker);

        let mut models: Vec<String> = maker_lenses
            .iter()
            .map(|lens| lens.get_display_name(&maker_lenses))
            .collect();
        models.sort_unstable();
        models.dedup();
        Ok(models)
    } else {
        Err("Lens database not loaded".to_string())
    }
}

#[cfg(test)]
mod tests;

#[tauri::command]
pub fn autodetect_lens(
    maker: String,
    model: String,
    state: tauri::State<AppState>,
) -> Result<Option<(String, String)>, String> {
    let db_guard = state
        .lens_db
        .lock()
        .map_err(|e| format!("Lock poisoned: {}", e))?;
    if let Some(db) = &*db_guard {
        Ok(find_best_lens_match(db, &maker, &model))
    } else {
        Ok(None)
    }
}

#[tauri::command]
pub fn autodetect_lens_profile(
    maker: String,
    model: String,
    focal_length: f32,
    aperture: Option<f32>,
    distance: Option<f32>,
    state: tauri::State<AppState>,
) -> Result<Option<DetectedLensProfile>, String> {
    let db_guard = state
        .lens_db
        .lock()
        .map_err(|e| format!("Lock poisoned: {}", e))?;
    let Some(db) = &*db_guard else {
        return Ok(None);
    };
    let Some((detected_maker, detected_model)) = find_best_lens_match(db, &maker, &model) else {
        return Ok(None);
    };

    let params = resolve_lens_params(
        db,
        &detected_maker,
        &detected_model,
        focal_length,
        aperture,
        distance,
    );

    Ok(Some(DetectedLensProfile {
        maker: detected_maker,
        model: detected_model,
        params,
    }))
}

#[tauri::command]
pub fn get_lens_distortion_params(
    maker: String,
    model: String,
    focal_length: f32,
    aperture: Option<f32>,
    distance: Option<f32>,
    state: State<AppState>,
) -> Result<Option<LensDistortionParams>, String> {
    let db_guard = state
        .lens_db
        .lock()
        .map_err(|e| format!("Lock poisoned: {}", e))?;
    if let Some(db) = &*db_guard {
        let maker_lenses = lenses_for_maker(db, &maker);

        if let Some(lens) = maker_lenses
            .iter()
            .find(|l| l.get_display_name(&maker_lenses) == model)
        {
            return Ok(lens.get_distortion_params(focal_length, aperture, distance));
        }
    }
    Ok(None)
}
