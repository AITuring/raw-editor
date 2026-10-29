//! Finding a Lensfun profile for camera metadata.
//!
//! Maker filtering, normalised fuzzy matching of EXIF lens names and the exact
//! maker/model lookup that resolves calibration parameters.

use fuzzy_matcher::FuzzyMatcher;

use super::{Lens, LensDatabase, LensDistortionParams};

pub(super) fn lenses_for_maker<'a>(db: &'a LensDatabase, maker: &str) -> Vec<&'a Lens> {
    db.lenses
        .iter()
        .filter(|l| l.get_maker() == maker)
        .collect()
}

pub fn find_best_lens_match(
    db: &LensDatabase,
    maker: &str,
    model: &str,
) -> Option<(String, String)> {
    let clean_maker = maker.trim().trim_matches('"').to_string();
    let clean_model = model.trim().trim_matches('"').to_string();
    if clean_model.len() < 3 {
        return None;
    }

    let matcher = fuzzy_matcher::skim::SkimMatcherV2::default().ignore_case();
    let normalized_query = normalize_search_text(&clean_model);
    let query_tokens = search_tokens(&normalized_query);

    let best_match = db
        .lenses
        .iter()
        .filter_map(|lens| {
            let english_name = lens.get_full_model_name();
            let canonical_name = lens.get_canonical_model_name();

            let score_english = matcher
                .fuzzy_match(&english_name, &clean_model)
                .unwrap_or(0);
            let score_canonical = matcher
                .fuzzy_match(&canonical_name, &clean_model)
                .unwrap_or(0);
            let (score, best_name) = if score_canonical > score_english {
                (score_canonical, canonical_name)
            } else {
                (score_english, english_name)
            };
            let normalized_candidate = normalize_search_text(&best_name);
            let candidate_tokens = search_tokens(&normalized_candidate);
            let token_overlap = query_tokens
                .iter()
                .filter(|token| candidate_tokens.contains(token))
                .count() as i64;
            let distinctive_token_overlap = query_tokens
                .iter()
                .filter(|token| token.len() >= 4 && candidate_tokens.contains(token))
                .count() as i64;
            let contains_model = normalized_candidate.contains(&normalized_query)
                || normalized_query.contains(&normalized_candidate);
            let maker_match = makers_match(&clean_maker, &lens.get_maker());
            let length_penalty =
                (normalized_candidate.len() as i64 - normalized_query.len() as i64).abs() / 3;
            let adjusted_score = score - length_penalty
                + token_overlap * 12
                + distinctive_token_overlap * 40
                + if contains_model { 25 } else { 0 }
                + if maker_match { 14 } else { 0 };

            let has_model_evidence =
                token_overlap >= 2 || contains_model || (score > 0 && token_overlap > 0);

            if adjusted_score >= 24 && has_model_evidence {
                Some((adjusted_score, lens))
            } else {
                None
            }
        })
        .max_by_key(|(score, _): &(i64, _)| *score);

    if let Some((_, best_lens)) = best_match {
        let lens_maker = best_lens.get_maker();
        let maker_lenses = lenses_for_maker(db, &lens_maker);
        return Some((lens_maker, best_lens.get_display_name(&maker_lenses)));
    }

    None
}

pub(super) fn normalize_search_text(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) fn search_tokens(value: &str) -> Vec<&str> {
    value
        .split_whitespace()
        .filter(|token| token.len() >= 2)
        .collect()
}

pub(super) fn makers_match(left: &str, right: &str) -> bool {
    let normalized_left = normalize_search_text(left);
    let normalized_right = normalize_search_text(right);
    !normalized_left.is_empty()
        && !normalized_right.is_empty()
        && (normalized_left == normalized_right
            || normalized_left.contains(&normalized_right)
            || normalized_right.contains(&normalized_left))
}

pub fn resolve_lens_params(
    db: &LensDatabase,
    maker: &str,
    model: &str,
    focal_length: f32,
    aperture: Option<f32>,
    distance: Option<f32>,
) -> Option<LensDistortionParams> {
    let maker_lenses = lenses_for_maker(db, maker);
    if let Some(lens) = maker_lenses
        .iter()
        .find(|l| l.get_display_name(&maker_lenses) == model)
    {
        lens.get_distortion_params(focal_length, aperture, distance)
    } else {
        None
    }
}
