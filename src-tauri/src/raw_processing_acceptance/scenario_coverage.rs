//! Sony α7R V scenario coverage of the acceptance manifest.
//!
//! Every named scenario is covered by exactly one fixture or explicitly deferred.

use std::collections::HashSet;

use super::AcceptanceManifest;

pub(super) const SONY_A7_RV_SCENARIOS: [&str; 8] = [
    "daylight",
    "tungsten",
    "high-iso",
    "underexposed",
    "saturated-light",
    "lossy-arw",
    "lossless-l-arw",
    "uncompressed-arw",
];

pub(super) fn validate_sony_a7_rv_scenario_coverage(manifest: &AcceptanceManifest) -> Vec<String> {
    let expected: HashSet<&str> = SONY_A7_RV_SCENARIOS.into_iter().collect();
    let deferred: HashSet<&str> = manifest
        .deferred_sony_a7_rv_scenarios
        .iter()
        .map(String::as_str)
        .collect();
    assert_eq!(
        deferred.len(),
        manifest.deferred_sony_a7_rv_scenarios.len(),
        "deferred α7R V scenarios must be unique"
    );

    let mut covered = HashSet::new();
    for fixture in &manifest.files {
        for scenario in &fixture.covered_sony_a7_rv_scenarios {
            assert!(
                expected.contains(scenario.as_str()),
                "{} names unknown α7R V scenario '{scenario}'",
                fixture.id
            );
            assert!(
                covered.insert(scenario.as_str()),
                "α7R V scenario '{scenario}' is covered by more than one fixture"
            );
        }
    }

    assert!(
        deferred.is_disjoint(&covered),
        "covered α7R V scenarios must not remain deferred"
    );
    let accounted_for: HashSet<&str> = deferred.union(&covered).copied().collect();
    assert_eq!(
        accounted_for, expected,
        "every α7R V acceptance scenario must be either covered or deferred"
    );

    let mut covered: Vec<String> = covered.into_iter().map(str::to_string).collect();
    covered.sort();
    covered
}
