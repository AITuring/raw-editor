//! Acceptance harness inputs: directories, report selection and source snapshots.
//!
//! Path validation without writes, unique Stack_Report selection, frozen station
//! source lists and Source_RAW snapshots used by the ignored harnesses in the
//! facade, plus the targeted replay harness.

use super::super::virtual_tile;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use tempfile::TempDir;

pub(super) fn write_acceptance_source_fixture(source_dir: &Path) {
    fs::create_dir_all(source_dir).expect("create acceptance source fixture");
    for index in 3680..=3763 {
        fs::write(source_dir.join(format!("DSC_{index}.NEF")), [index as u8])
            .expect("write acceptance source fixture");
    }
}

#[test]
fn validate_acceptance_directories_rejects_invalid_paths_and_sources() {
    let root = TempDir::new().expect("acceptance validation tempdir");
    let source = root.path().join("source");
    let report = root.path().join("report");
    write_acceptance_source_fixture(&source);
    assert!(validate_acceptance_directories(&source, &report).is_ok());
    assert!(validate_acceptance_directories(Path::new("relative"), &report).is_err());
    assert!(validate_acceptance_directories(&source, Path::new("relative")).is_err());
    let nested_report = source.join("new/reports");
    let before_nested_rejection = acceptance_source_snapshot(&source);
    assert!(validate_acceptance_directories(&source, &nested_report).is_err());
    assert_eq!(
        acceptance_source_snapshot(&source),
        before_nested_rejection,
        "a rejected preflight must not change the source inventory"
    );
    assert!(!nested_report.exists());

    let missing_root = TempDir::new().expect("missing source tempdir");
    let missing_source = missing_root.path().join("source");
    fs::create_dir_all(&missing_source).unwrap();
    assert!(
        validate_acceptance_directories(&missing_source, &missing_root.path().join("report"))
            .is_err()
    );

    let directory_root = TempDir::new().expect("directory source tempdir");
    let directory_source = directory_root.path().join("source");
    write_acceptance_source_fixture(&directory_source);
    let invalid = directory_source.join("DSC_3680.NEF");
    fs::remove_file(&invalid).unwrap();
    fs::create_dir(&invalid).unwrap();
    assert!(
        validate_acceptance_directories(&directory_source, &directory_root.path().join("report"))
            .is_err()
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let symlink_root = TempDir::new().expect("symlink source tempdir");
        let symlink_source = symlink_root.path().join("source");
        write_acceptance_source_fixture(&symlink_source);
        let invalid = symlink_source.join("DSC_3680.NEF");
        fs::remove_file(&invalid).unwrap();
        symlink(symlink_source.join("DSC_3681.NEF"), &invalid).unwrap();
        assert!(
            validate_acceptance_directories(&symlink_source, &symlink_root.path().join("report"))
                .is_err()
        );
    }
}

#[test]
fn stack_report_selection_requires_one_matching_regular_file() {
    let directory = TempDir::new().expect("report selection tempdir");
    fs::write(directory.path().join("other.json"), b"{}").unwrap();
    assert!(select_unique_stack_report(directory.path()).is_err());
    let first = directory.path().join("stack-report-first.json");
    fs::write(&first, b"{}").unwrap();
    assert_eq!(select_unique_stack_report(directory.path()).unwrap(), first);
    fs::write(directory.path().join("stack-report-second.json"), b"{}").unwrap();
    assert!(select_unique_stack_report(directory.path()).is_err());
}

pub(super) fn resolve_acceptance_path_without_writes(path: &Path) -> Result<PathBuf, String> {
    let mut resolved = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix) => resolved.push(prefix.as_os_str()),
            std::path::Component::RootDir => resolved.push(std::path::MAIN_SEPARATOR.to_string()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                resolved.pop();
            }
            std::path::Component::Normal(name) => {
                let candidate = resolved.join(name);
                match fs::symlink_metadata(&candidate) {
                    Ok(_) => {
                        resolved = candidate.canonicalize().map_err(|error| {
                            format!("cannot canonicalize {}: {error}", candidate.display())
                        })?;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        resolved = candidate;
                    }
                    Err(error) => {
                        return Err(format!(
                            "cannot inspect acceptance path {}: {error}",
                            candidate.display()
                        ));
                    }
                }
            }
        }
    }
    Ok(resolved)
}

pub(super) fn validate_acceptance_directories(
    source_dir: &Path,
    report_dir: &Path,
) -> Result<(PathBuf, PathBuf, Vec<PathBuf>), String> {
    if !source_dir.is_absolute() || !report_dir.is_absolute() {
        return Err("acceptance source and report directories must be absolute paths".to_string());
    }
    let source_dir = source_dir.canonicalize().map_err(|error| {
        format!(
            "cannot canonicalize source {}: {error}",
            source_dir.display()
        )
    })?;
    if !source_dir.is_dir() {
        return Err(format!(
            "acceptance source is not a directory: {}",
            source_dir.display()
        ));
    }
    let report_dir = resolve_acceptance_path_without_writes(report_dir)?;
    if source_dir == report_dir
        || source_dir.starts_with(&report_dir)
        || report_dir.starts_with(&source_dir)
    {
        return Err(
            "acceptance source and report directories must be separate and non-nested".to_string(),
        );
    }
    fs::create_dir_all(&report_dir).map_err(|error| {
        format!(
            "cannot create report directory {}: {error}",
            report_dir.display()
        )
    })?;
    let report_dir = report_dir.canonicalize().map_err(|error| {
        format!(
            "cannot canonicalize report {}: {error}",
            report_dir.display()
        )
    })?;
    if !report_dir.is_dir() {
        return Err(format!(
            "acceptance report is not a directory: {}",
            report_dir.display()
        ));
    }

    let mut paths = Vec::with_capacity(84);
    for index in 3680..=3763 {
        let path = source_dir.join(format!("DSC_{index}.NEF"));
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| format!("cannot read metadata for {}: {error}", path.display()))?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(format!(
                "acceptance source must be a regular non-symlink file: {}",
                path.display()
            ));
        }
        path.to_str()
            .ok_or_else(|| format!("acceptance source path is not UTF-8: {}", path.display()))?;
        fs::File::open(&path).map_err(|error| {
            format!("cannot read acceptance source {}: {error}", path.display())
        })?;
        virtual_tile::source_file_sha256(&path).map_err(|error| {
            format!("cannot hash acceptance source {}: {error}", path.display())
        })?;
        paths.push(path);
    }
    Ok((source_dir, report_dir, paths))
}

pub(super) fn select_unique_stack_report(report_dir: &Path) -> Result<PathBuf, String> {
    let mut reports = fs::read_dir(report_dir)
        .map_err(|error| {
            format!(
                "cannot read report directory {}: {error}",
                report_dir.display()
            )
        })?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_str()?;
            let regular = fs::symlink_metadata(&path)
                .ok()
                .is_some_and(|metadata| metadata.file_type().is_file());
            (regular && name.starts_with("stack-report-") && name.ends_with(".json"))
                .then_some(path)
        })
        .collect::<Vec<_>>();
    reports.sort();
    match reports.as_slice() {
        [report] => Ok(report.clone()),
        _ => Err(format!(
            "expected exactly one regular stack-report-*.json, found {}: {:?}",
            reports.len(),
            reports
        )),
    }
}

#[test]
#[ignore = "targeted replay harness requires explicitly selected local sources"]
fn stack_targeted_replay_harness() {
    let source_dir = std::env::var_os("RAW_EDITOR_STACK_TARGETED_SOURCE_DIR")
        .map(PathBuf::from)
        .expect("RAW_EDITOR_STACK_TARGETED_SOURCE_DIR is required")
        .canonicalize()
        .expect("targeted source directory must exist");
    let report_dir = std::env::var_os("RAW_EDITOR_STACK_ACCEPTANCE_REPORT_DIR")
        .map(PathBuf::from)
        .expect("RAW_EDITOR_STACK_ACCEPTANCE_REPORT_DIR is required");
    let report_dir = resolve_acceptance_path_without_writes(&report_dir)
        .expect("targeted report path must be valid");
    assert!(
        source_dir != report_dir
            && !source_dir.starts_with(&report_dir)
            && !report_dir.starts_with(&source_dir),
        "targeted source and report directories must be separate and non-nested"
    );
    fs::create_dir_all(&report_dir).expect("create targeted report directory");
    assert!(
        fs::read_dir(&report_dir).unwrap().next().is_none(),
        "targeted report directory must be fresh"
    );
    let names = std::env::var("RAW_EDITOR_STACK_TARGETED_SOURCE_NAMES")
        .expect("RAW_EDITOR_STACK_TARGETED_SOURCE_NAMES is required");
    let paths = names
        .split(',')
        .filter(|name| !name.is_empty())
        .map(|name| source_dir.join(name))
        .collect::<Vec<_>>();
    assert!(
        paths.len() >= 2,
        "targeted replay needs at least two sources"
    );
    let before = paths
        .iter()
        .map(|path| {
            let metadata = fs::symlink_metadata(path).expect("targeted source metadata");
            assert!(metadata.file_type().is_file() && !metadata.file_type().is_symlink());
            (
                path.clone(),
                virtual_tile::source_file_sha256(path).expect("hash targeted source"),
            )
        })
        .collect::<Vec<_>>();
    unsafe {
        std::env::set_var("RAW_EDITOR_STACK_REPORT_DIR", &report_dir);
    }
    crate::sidecar_storage::initialize(Path::new(
        "/private/tmp/raw-editor-stack-targeted-replay-sidecars",
    ))
    .expect("targeted replay sidecar storage should initialize");
    let app = tauri::test::mock_app();
    let result = crate::panorama_stitching::stitch_images_with_options(
        paths
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
        app.handle().clone(),
        crate::panorama_stitching::AlignmentMode::Auto,
        crate::panorama_stitching::BlendMode::FocusStack,
        "test-stack-targeted-replay-progress",
    );
    let after = paths
        .iter()
        .map(|path| {
            (
                path.clone(),
                virtual_tile::source_file_sha256(path).expect("rehash targeted source"),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(before, after, "targeted Source_RAW bytes changed");
    let report_path = select_unique_stack_report(&report_dir)
        .expect("targeted replay must write one Stack_Report");
    println!(
        "TARGETED_REPLAY result={:?} report={}",
        result.as_ref().map(|_| "ok"),
        report_path.display()
    );
    if let Err(error) = result {
        eprintln!("targeted replay stopped after writing evidence: {error}");
    }
}

pub(super) fn frozen_station_source_paths(source_dir: &Path, station: usize) -> Vec<PathBuf> {
    // Exact members from the frozen authoritative 84-image report. These
    // names locate an ignored test fixture only; production grouping and every
    // relation still come from image evidence.
    let members: &[u16] = match station {
        0 => &[3751, 3752, 3753],
        1 => &[3750, 3749],
        2 => &[3735, 3734, 3733],
        3 => &[3713, 3712, 3711],
        4 => &[3690, 3691],
        5 => &[3732, 3731, 3730],
        6 => &[3692, 3693],
        7 => &[3681, 3680, 3682, 3683],
        8 => &[3709, 3710, 3708, 3707],
        9 => &[3754, 3755, 3756],
        10 => &[3748, 3747],
        11 => &[3716, 3717, 3715, 3714],
        12 => &[3729, 3728, 3727],
        13 => &[3736, 3738, 3737],
        14 => &[3706, 3704, 3705],
        15 => &[3696, 3695, 3694],
        16 => &[3685, 3686, 3684],
        17 => &[3757, 3758, 3759],
        18 => &[3719, 3718, 3720],
        19 => &[3744, 3745, 3746, 3743],
        20 => &[3726, 3725, 3724],
        21 => &[3701, 3700],
        22 => &[3702, 3703],
        23 => &[3739, 3742, 3741, 3740],
        24 => &[3763, 3762, 3761, 3760],
        25 => &[3722, 3723, 3721],
        26 => &[3697, 3699, 3698],
        27 => &[3689, 3688, 3687],
        _ => panic!("unknown frozen station {station}"),
    };
    members
        .iter()
        .map(|index| source_dir.join(format!("DSC_{index}.NEF")))
        .collect()
}

pub(super) fn parse_station_pair_cases(value: &str) -> Vec<(usize, usize)> {
    value
        .split(',')
        .filter(|case| !case.is_empty())
        .map(|case| {
            let (left, right) = case
                .split_once('-')
                .unwrap_or_else(|| panic!("invalid station pair `{case}`"));
            let left = left
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid station `{left}`"));
            let right = right
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("invalid station `{right}`"));
            (left.min(right), left.max(right))
        })
        .collect()
}

pub(super) fn acceptance_source_snapshot(source_dir: &Path) -> BTreeMap<String, (u64, String)> {
    let required = (3680..=3763)
        .map(|index| format!("DSC_{index}.NEF"))
        .collect::<BTreeSet<_>>();
    fs::read_dir(source_dir)
        .expect("acceptance source directory should be readable")
        .map(|entry| {
            let entry = entry.expect("every source directory entry must be readable");
            let path = entry.path();
            let name = entry
                .file_name()
                .into_string()
                .expect("source filenames must be UTF-8");
            let size = entry
                .metadata()
                .expect("source metadata must be readable")
                .len();
            // Extra files are inventoried by name/size only; their contents are never read.
            let digest = if required.contains(&name) {
                virtual_tile::hex_digest(
                    &virtual_tile::source_file_sha256(&path)
                        .expect("every required NEF must be hashable"),
                )
            } else {
                String::new()
            };
            (name, (size, digest))
        })
        .collect::<BTreeMap<_, _>>()
}
