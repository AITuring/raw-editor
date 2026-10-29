#[cfg(not(all(target_os = "windows", target_arch = "aarch64")))]
use mimalloc::MiMalloc;

#[cfg(not(all(target_os = "windows", target_arch = "aarch64")))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod adjustment_utils;
mod ai_commands;
mod ai_processing;
mod android_integration;
mod app_settings;
mod app_state;
mod cache_utils;
mod color_management;
mod crop_analysis;
mod culling;
mod denoising;
mod exif_processing;
mod export_processing;
mod file_management;
mod formats;
mod gpu_processing;
mod hdr_deghosting;
mod image_loader;
mod image_processing;
mod image_stack;
mod inpainting;
mod launch_request;
mod lens_blur;
mod lens_correction;
mod lut_processing;
mod mask_generation;
mod multi_exposure;
mod negative_conversion;
mod panorama_stitching;
mod panorama_utils;
mod preset_converter;
mod preview_protocol;
mod psd_loader;
mod raw_processing;
mod render_strategy;
mod sidecar_storage;
mod style_transfer;
mod tagging;
mod tagging_utils;
mod window_customizer;

#[path = "lib/preview_cache.rs"]
mod preview_cache;

#[cfg(test)]
use preview_cache::transformed_image_into_arc;
pub use preview_cache::{
    TransformedPreview, generate_transformed_preview, get_cached_full_warped_image,
};

#[path = "lib/preview_worker.rs"]
mod preview_worker;

#[cfg(test)]
use preview_worker::should_output_to_native_display;
use preview_worker::{start_analytics_worker, start_preview_worker};

#[path = "lib/preview_commands.rs"]
mod preview_commands;

pub use preview_commands::WgpuTransformPayload;

#[path = "lib/geometry_previews.rs"]
mod geometry_previews;

pub use geometry_previews::get_original_image;

#[path = "lib/merged_outputs.rs"]
mod merged_outputs;

#[path = "lib/logging.rs"]
mod logging;

use logging::setup_logging;

#[path = "lib/window_state.rs"]
mod window_state;

use window_state::{available_monitor_bounds, saved_window_state_is_usable};

use std::collections::HashMap;
use std::io::Cursor;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tauri::{Emitter, Manager};
use tempfile::NamedTempFile;
use tokio::sync::Mutex as TokioMutex;

use crate::cache_utils::{
    BudgetedCache, DECODED_IMAGE_CACHE_MAX_BYTES, DecodedImageCache, GEOMETRY_CACHE_MAX_BYTES,
    GEOMETRY_CACHE_MAX_ENTRIES, LUT_CACHE_MAX_BYTES, LUT_CACHE_MAX_ENTRIES, MASK_CACHE_MAX_BYTES,
    MASK_CACHE_MAX_ENTRIES, THUMBNAIL_GEOMETRY_CACHE_MAX_BYTES,
    THUMBNAIL_GEOMETRY_CACHE_MAX_ENTRIES,
};
use crate::file_management::parse_virtual_path;
use crate::image_processing::get_or_init_gpu_context;
use crate::mask_generation::get_cached_or_generate_mask;
use crate::window_customizer::PinchZoomDisablePlugin;
pub use adjustment_utils::*;
pub use android_integration::*;
pub use app_settings::*;
pub use app_state::*;
pub use launch_request::*;
use tagging_utils::{candidates, hierarchy};

#[cfg(target_os = "macos")]
extern "C" fn force_exit(_signal: libc::c_int) {
    unsafe {
        libc::_exit(0);
    }
}

#[cfg(target_os = "macos")]
pub fn register_exit_handler() {
    unsafe {
        libc::signal(libc::SIGABRT, force_exit as *const () as libc::sighandler_t);
    }
}

#[cfg(not(target_os = "macos"))]
pub fn register_exit_handler() {}

#[derive(serde::Serialize)]
struct ImageDimensions {
    width: u32,
    height: u32,
}

#[cfg(test)]
#[path = "lib/transformed_image_sharing_tests.rs"]
mod transformed_image_sharing_tests;

#[tauri::command]
fn get_image_dimensions(path: String) -> Result<ImageDimensions, String> {
    let (source_path, _) = parse_virtual_path(&path);
    image::image_dimensions(&source_path)
        .map(|(width, height)| ImageDimensions { width, height })
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn cancel_thumbnail_generation(
    state: tauri::State<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    state
        .thumbnail_cancellation_token
        .store(true, Ordering::SeqCst);

    let mut tracker = state.thumbnail_progress.lock().unwrap();
    tracker.total = 0;
    tracker.completed = 0;
    drop(tracker);

    let _ = app_handle.emit(
        "thumbnail-progress",
        serde_json::json!({ "current": 0, "total": 0 }),
    );
    Ok(())
}

#[cfg(test)]
#[path = "lib/native_preview_display_tests.rs"]
mod native_preview_display_tests;

#[tauri::command]
async fn save_temp_file(bytes: Vec<u8>) -> Result<String, String> {
    let mut temp_file = NamedTempFile::new().map_err(|e| e.to_string())?;
    temp_file.write_all(&bytes).map_err(|e| e.to_string())?;
    let (_file, path) = temp_file.keep().map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().to_string())
}

#[cfg(target_os = "linux")]
fn is_nvidia_gpu() -> bool {
    std::path::Path::new("/proc/driver/nvidia/version").exists()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = rayon::ThreadPoolBuilder::new()
        .stack_size(8 * 1024 * 1024)
        .build_global();

    let mut builder = tauri::Builder::default();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let launch_req = parse_launch_args(&args);
    let is_headless = matches!(launch_req, LaunchRequest::HeadlessExport(_));

    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    {
        if !is_headless {
            builder = builder.plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
                log::info!(
                    "New instance launched with args: {:?}. Focusing main window.",
                    argv
                );
                if let Some(window) = app.get_webview_window("main") {
                    if let Err(e) = window.unminimize() {
                        log::error!("Failed to unminimize window: {}", e);
                    }
                    if let Err(e) = window.set_focus() {
                        log::error!("Failed to set focus on window: {}", e);
                    }
                }

                let forwarded_args = argv.get(1..).unwrap_or(&[]);
                emit_launch_request(app, parse_launch_args(forwarded_args));
            }));
        }
    }

    builder
        .plugin(tauri_plugin_os::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .plugin(PinchZoomDisablePlugin)
        .on_window_event(|window, event| if let tauri::WindowEvent::Resized(size) = event {
            let state = window.state::<AppState>();
            if let Some(ctx) = state.gpu_context.lock().unwrap().as_ref()
                && let Ok(mut display_lock) = ctx.display.try_lock()
                    && let Some(display) = display_lock.as_mut() {
                        display.config.width = size.width.max(1);
                        display.config.height = size.height.max(1);
                        display.surface.configure(&ctx.device, &display.config);
                        display.render(&ctx.device, &ctx.queue);
                    }
        })
        .setup(move |app| {
            let state = app.state::<AppState>();

            #[cfg(any(windows, target_os = "linux", target_os = "macos"))]
            {
                match launch_req.clone() {
                    LaunchRequest::EditSession(session) => {
                        log::info!("Initial launch with external edit session for: {}", session.source);
                        *state.pending_edit_session.lock().unwrap() = Some(session);
                    }
                    LaunchRequest::OpenFile(path) => {
                        log::info!("Initial open: Storing path {} for later.", path);
                        *state.initial_file_path.lock().unwrap() = Some(path);
                    }
                    _ => {}
                }
            }

            let app_handle = app.handle().clone();

            let app_data_dir = app_handle
                .path()
                .app_data_dir()
                .map_err(|error| error.to_string())?;
            sidecar_storage::initialize(&app_data_dir)?;

            {
                let disks_app_handle = app_handle.clone();
                std::thread::spawn(move || {
                    let disks = sysinfo::Disks::new_with_refreshed_list();
                    let state = disks_app_handle.state::<AppState>();
                    *state.disks_cache.lock().unwrap() = Some(disks);
                });
            }

            let config_dir = app_handle.path().app_config_dir().expect("Failed to get config dir");
            let crash_flag_path = config_dir.join(".gpu_init_crash_flag");

            {
                let state = app.state::<AppState>();
                *state.gpu_crash_flag_path.lock().unwrap() = Some(crash_flag_path.clone());
            }

            let mut settings: AppSettings = load_settings(app_handle.clone()).unwrap_or_default();

            {
                let state = app.state::<AppState>();
                let cache_size = settings.image_cache_size.unwrap_or(5) as usize;
                state.decoded_image_cache.lock().unwrap().set_capacity(cache_size);
            }

            if crash_flag_path.exists() {
                log::warn!("GPU Driver crash detected on last run! Falling back to OpenGL backend.");
                settings.processing_backend = Some("gl".to_string());
                let _ = crate::save_settings(settings.clone(), app_handle.clone());
                let _ = std::fs::remove_file(&crash_flag_path);
            }

            let lens_db = lens_correction::load_lensfun_db(&app_handle);
            {
                let state = app.state::<AppState>();
                *state.lens_db.lock().unwrap() = Some(Arc::new(lens_db));
            }

            unsafe {
                if let Some(backend) = &settings.processing_backend
                    && backend != "auto" {
                        std::env::set_var("WGPU_BACKEND", backend);
                    }

                #[cfg(target_os = "linux")]
                {
                    if settings.linux_gpu_optimization.unwrap_or(false) {
                        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
                        std::env::set_var("WEBKIT_DISABLE_COMPOSITING_MODE", "1");
                        std::env::set_var("NODEVICE_SELECT", "1");
                    } else if is_nvidia_gpu() {
                        std::env::set_var("__NV_DISABLE_EXPLICIT_SYNC", "1");
                    }
                }

                #[cfg(not(target_os = "android"))]
                {
                    let resource_path = app_handle
                        .path()
                        .resolve("resources", tauri::path::BaseDirectory::Resource)
                        .expect("failed to resolve resource directory");

                    let ort_library_name = {
                        #[cfg(target_os = "windows")]
                        { "onnxruntime.dll" }
                        #[cfg(target_os = "linux")]
                        { "libonnxruntime.so" }
                        #[cfg(target_os = "macos")]
                        { "libonnxruntime.dylib" }
                        #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
                        { "libonnxruntime.so" }
                    };
                    let ort_library_path = resource_path.join(ort_library_name);
                    if ort_library_path.is_file() {
                        std::env::set_var("ORT_DYLIB_PATH", &ort_library_path);
                        println!("Set ORT_DYLIB_PATH to: {}", ort_library_path.display());
                    }
                }
            }

            setup_logging(&app_handle);

            if let Some(backend) = &settings.processing_backend
                && backend != "auto" {
                    log::info!("Applied processing backend setting: {}", backend);
                }
            #[cfg(target_os = "linux")]
            {
                if settings.linux_gpu_optimization.unwrap_or(false) {
                    log::info!("Applied Linux Compatibility Mode (forced software compositing).");
                } else if is_nvidia_gpu() {
                    log::info!("Applied Nvidia explicit-sync workaround (hardware compositing kept).");
                }
            }

            if let LaunchRequest::HeadlessExport(session) = launch_req {
                let app_handle_clone = app_handle.clone();
                tauri::async_runtime::spawn(async move {
                    match crate::export_processing::run_headless_export(session, app_handle_clone.clone()).await {
                        Ok(_) => {
                            println!("Headless export completed successfully.");
                            app_handle_clone.exit(0);
                        }
                        Err(e) => {
                            eprintln!("Headless export failed: {}", e);
                            app_handle_clone.exit(1);
                        }
                    }
                });

                return Ok(());
            }

            start_preview_worker(app_handle.clone());
            start_analytics_worker(app_handle.clone());
            file_management::start_thumbnail_workers(app_handle.clone());
            file_management::start_metadata_workers(app_handle.clone());
            jxl_oxide::integration::register_image_decoding_hook();

            let window_cfg = app.config().app.windows.first().unwrap().clone();
            let decorations = settings.decorations.unwrap_or(window_cfg.decorations);
            #[cfg(target_os = "android")]
            let _ = decorations;

            let main_window_cfg = app
                .config()
                .app
                .windows
                .iter()
                .find(|w| w.label == "main")
                .expect("Main window config not found")
                .clone();

            let mut window_builder =
                tauri::WebviewWindowBuilder::from_config(app.handle(), &main_window_cfg)
                    .unwrap();

            #[cfg(not(target_os = "android"))]
            {
                window_builder = window_builder.decorations(decorations).visible(false);
            }

            let window = window_builder.build().expect("Failed to build window");

            #[cfg(target_os = "android")]
            android_integration::initialize_android(&window);

            #[cfg(not(target_os = "android"))]
            {
                let app_state = app.state::<AppState>();
                if let Err(error) = get_or_init_gpu_context(&app_state, app.handle()) {
                    log::warn!(
                        "GPU pre-initialization failed (editing and thumbnails may be degraded): {}",
                        error
                    );
                }

                if let Ok(config_dir) = app.path().app_config_dir() {
                    let path = config_dir.join("window_state.json");
                    if let Ok(contents) = std::fs::read_to_string(&path) {
                        if let Ok(state) = serde_json::from_str::<WindowState>(&contents) {
                            let monitor_bounds = available_monitor_bounds(&window);
                            if saved_window_state_is_usable(&state, &monitor_bounds) {
                                let _ = window.set_size(tauri::Size::Physical(
                                    tauri::PhysicalSize::new(state.width, state.height),
                                ));
                                let _ = window.set_position(tauri::Position::Physical(
                                    tauri::PhysicalPosition::new(state.x, state.y),
                                ));
                            } else {
                                log::warn!(
                                    "Saved window state was unusable ({}x{} at {},{}), centering instead.",
                                    state.width,
                                    state.height,
                                    state.x,
                                    state.y
                                );
                                let _ = window.center();
                            }
                        } else {
                            let _ = window.center();
                        }
                    } else {
                        let _ = window.center();
                    }
                } else {
                    let _ = window.center();
                }

                let window_failsafe = window.clone();
                tauri::async_runtime::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(4)).await;
                    if let Ok(false) = window_failsafe.is_visible() {
                        log::warn!(
                            "Frontend failed to report ready within timeout. Forcing window visibility."
                        );
                        let _ = window_failsafe.show();
                        let _ = window_failsafe.set_focus();
                    }
                });

                let pending_window_state = Arc::new(Mutex::new(None::<WindowState>));
                let pending_state_for_saver = pending_window_state.clone();
                let app_handle_for_saver = app.handle().clone();

                tauri::async_runtime::spawn(async move {
                    loop {
                        tokio::time::sleep(Duration::from_millis(500)).await;

                        let state_to_save = {
                            let mut lock = pending_state_for_saver.lock().unwrap();
                            lock.take()
                        };

                        if let Some(state) = state_to_save
                            && let Ok(config_dir) =
                                app_handle_for_saver.path().app_config_dir()
                        {
                            let path = config_dir.join("window_state.json");
                            let _ = std::fs::create_dir_all(&config_dir);
                            if let Ok(json) = serde_json::to_string(&state) {
                                let _ = std::fs::write(&path, json);
                            }
                        }
                    }
                });

                let window_for_handler = window.clone();
                let pending_state_for_handler = pending_window_state.clone();

                window.on_window_event(move |event| match event {
                    tauri::WindowEvent::Resized(_) | tauri::WindowEvent::Moved(_) => {
                        #[cfg(any(windows, target_os = "linux"))]
                        let maximized = window_for_handler.is_maximized().unwrap_or(false);
                        #[cfg(not(any(windows, target_os = "linux")))]
                        let maximized = false;

                        #[cfg(any(windows, target_os = "linux"))]
                        let fullscreen = window_for_handler.is_fullscreen().unwrap_or(false);
                        #[cfg(not(any(windows, target_os = "linux")))]
                        let fullscreen = false;

                        if window_for_handler.is_minimized().unwrap_or(false) {
                            return;
                        }

                        let mut state = WindowState {
                            width: 1280,
                            height: 720,
                            x: 0,
                            y: 0,
                            maximized,
                            fullscreen,
                        };

                        if let Ok(position) = window_for_handler.outer_position() {
                            state.x = position.x;
                            state.y = position.y;
                        }

                        if !maximized
                            && !fullscreen
                            && let Ok(size) = window_for_handler.outer_size()
                            && size.width >= 800
                            && size.height >= 600
                        {
                            state.width = size.width;
                            state.height = size.height;
                        }

                        *pending_state_for_handler.lock().unwrap() = Some(state);
                    }
                    _ => {}
                });
            }

            crate::register_exit_handler();
            Ok(())
        })
        .manage(AppState {
            window_setup_complete: AtomicBool::new(false),
            gpu_crash_flag_path: Mutex::new(None),
            original_image: Mutex::new(None),
            cached_preview: Mutex::new(None),
            gpu_context: Mutex::new(None),
            gpu_image_cache: Mutex::new(None),
            gpu_processor: Mutex::new(None),
            ai_state: Mutex::new(None),
            ai_init_lock: TokioMutex::new(()),
            export_task_token: Arc::new(Mutex::new(None)),
            hdr_result: Arc::new(Mutex::new(None)),
            panorama_result: Arc::new(Mutex::new(None)),
            image_stack_result: Arc::new(Mutex::new(None)),
            image_stack_generation: Arc::new(AtomicUsize::new(0)),
            denoise_result: Arc::new(Mutex::new(None)),
            indexing_task_handle: Mutex::new(None),
            lut_cache: Mutex::new(BudgetedCache::new(
                LUT_CACHE_MAX_ENTRIES,
                LUT_CACHE_MAX_BYTES,
            )),
            initial_file_path: Mutex::new(None),
            pending_edit_session: Mutex::new(None),
            thumbnail_cancellation_token: Arc::new(AtomicBool::new(false)),
            thumbnail_progress: Mutex::new(ThumbnailProgressTracker { total: 0, completed: 0 }),
            preview_worker_tx: Mutex::new(None),
            analytics_worker_tx: Mutex::new(None),
            mask_cache: Mutex::new(BudgetedCache::new(
                MASK_CACHE_MAX_ENTRIES,
                MASK_CACHE_MAX_BYTES,
            )),
            patch_cache: Mutex::new(HashMap::new()),
            geometry_cache: Mutex::new(BudgetedCache::new(
                GEOMETRY_CACHE_MAX_ENTRIES,
                GEOMETRY_CACHE_MAX_BYTES,
            )),
            thumbnail_geometry_cache: Mutex::new(BudgetedCache::new(
                THUMBNAIL_GEOMETRY_CACHE_MAX_ENTRIES,
                THUMBNAIL_GEOMETRY_CACHE_MAX_BYTES,
            )),
            lens_db: Mutex::new(None),
            load_image_generation: Arc::new(AtomicUsize::new(0)),
            full_warped_cache: Mutex::new(None),
            range_mask_source_cache: Mutex::new(None),
            full_transformed_cache: Mutex::new(None),
            decoded_image_cache: Mutex::new(DecodedImageCache::new(
                5,
                DECODED_IMAGE_CACHE_MAX_BYTES,
            )),
            thumbnail_manager: ThumbnailManager::new(),
            metadata_manager: MetadataManager::new(),
            disks_cache: Mutex::new(None),
            disks_cache_refreshing: AtomicBool::new(false),
        })
        .invoke_handler(tauri::generate_handler![
            preview_commands::apply_adjustments,
            preview_commands::generate_preview_for_path,
            geometry_previews::generate_original_transformed_preview,
            preview_commands::generate_preset_preview,
            geometry_previews::generate_uncropped_preview,
            geometry_previews::analyze_crop_straighten,
            geometry_previews::analyze_crop_upright,
            geometry_previews::preview_geometry_transform,
            logging::get_log_file_path,
            logging::frontend_log,
            merged_outputs::save_collage,
            merged_outputs::merge_hdr,
            merged_outputs::save_hdr,
            lut_processing::load_and_parse_lut,
            lut_processing::list_luts,
            lut_processing::import_luts,
            lut_processing::remove_lut,
            lut_processing::generate_lut_previews,
            save_temp_file,
            style_transfer::generate_style_transfer_preview,
            style_transfer::export_style_transfer,
            get_image_dimensions,
            window_state::frontend_ready,
            cancel_thumbnail_generation,
            preview_commands::update_wgpu_transform,
            android_integration::resolve_android_content_uri_name,
            cache_utils::clear_session_caches,
            cache_utils::clear_image_caches,
            app_settings::load_settings,
            app_settings::save_settings,
            ai_commands::generate_ai_subject_mask,
            ai_commands::precompute_ai_subject_mask,
            ai_commands::generate_ai_foreground_mask,
            ai_commands::generate_ai_sky_mask,
            ai_commands::generate_ai_depth_mask,
            ai_commands::generate_full_image_depth_map,
            inpainting::invoke_generative_replace_with_mask_def,
            inpainting::generate_manual_cleanup_patch,
            denoising::apply_denoising,
            denoising::batch_denoise_images,
            denoising::save_denoised_image,
            image_loader::load_image,
            image_loader::is_image_cached,
            image_stack::process_image_stack,
            image_stack::save_image_stack,
            image_stack::image_stack_output_notices,
            panorama_stitching::stitch_panorama,
            panorama_stitching::save_panorama,
            export_processing::export_images,
            export_processing::batch_geometry_correction,
            export_processing::analyze_batch_geometry,
            export_processing::preview_batch_geometry_correction,
            export_processing::import_watermark_image,
            export_processing::cancel_export,
            export_processing::estimate_export_sizes,
            image_processing::calculate_auto_adjustments,
            mask_generation::generate_mask_overlay,
            file_management::update_exif_fields,
            file_management::get_supported_file_types,
            file_management::read_exif_for_paths,
            file_management::listing::inspect_folder_contents,
            file_management::listing::list_images_in_dir,
            file_management::listing::list_images_recursive,
            file_management::folder_tree::get_folder_tree,
            file_management::folder_tree::get_folder_children,
            file_management::folder_tree::get_pinned_folder_trees,
            file_management::thumbnail_queue::update_thumbnail_queue,
            file_management::file_operations::create_folder,
            file_management::file_operations::delete_folder,
            file_management::file_operations::copy_files,
            file_management::file_operations::move_files,
            file_management::file_operations::rename_folder,
            file_management::file_operations::rename_files,
            file_management::file_operations::duplicate_file,
            file_management::file_operations::show_in_finder,
            file_management::file_operations::delete_files_from_disk,
            file_management::file_operations::delete_files_with_associated,
            file_management::sidecar_edits::save_metadata_and_update_thumbnail,
            file_management::sidecar_edits::apply_adjustments_to_paths,
            file_management::sidecar_edits::load_metadata,
            file_management::presets::load_presets,
            file_management::presets::save_presets,
            file_management::get_or_create_internal_library_root,
            file_management::sidecar_edits::reset_adjustments_for_paths,
            file_management::sidecar_edits::apply_auto_adjustments_to_paths,
            file_management::presets::handle_import_presets_from_file,
            file_management::presets::handle_import_legacy_presets_from_file,
            file_management::presets::handle_export_presets_to_file,
            file_management::sidecar_edits::clear_all_sidecars,
            file_management::thumbnail_queue::clear_thumbnail_cache,
            file_management::sidecar_edits::set_color_label_for_paths,
            file_management::sidecar_edits::set_rating_for_paths,
            file_management::importing::import_files,
            file_management::file_operations::create_virtual_copy,
            file_management::albums::get_albums,
            file_management::albums::save_albums,
            file_management::albums::add_to_album,
            file_management::albums::get_album_images,
            tagging::start_background_indexing,
            tagging::clear_ai_tags,
            tagging::clear_all_tags,
            tagging::add_tag_for_paths,
            tagging::remove_tag_for_paths,
            culling::cull_images,
            lens_correction::get_lensfun_makers,
            lens_correction::get_lensfun_lenses_for_maker,
            lens_correction::autodetect_lens,
            lens_correction::autodetect_lens_profile,
            lens_correction::get_lens_distortion_params,
            negative_conversion::preview_negative_conversion,
            negative_conversion::convert_negatives,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(#[allow(unused_variables)] |app_handle, event| {
            match event {
                #[cfg(target_os = "macos")]
                tauri::RunEvent::Opened { urls } => {
                    if let Some(url) = urls.first()
                        && let Ok(path) = url.to_file_path()
                        && let Some(path_str) = path.to_str()
                    {
                        let state = app_handle.state::<AppState>();
                        *state.initial_file_path.lock().unwrap() = Some(path_str.to_string());
                        log::info!("macOS initial open: Stored path {} for later.", path_str);
                    }
                }
                tauri::RunEvent::ExitRequested { api, .. } => {
                    api.prevent_exit();

                    #[cfg(target_os = "macos")]
                    unsafe { libc::_exit(0); }

                    #[cfg(not(target_os = "macos"))]
                    std::process::exit(0);
                }
                tauri::RunEvent::Exit => {
                    #[cfg(target_os = "macos")]
                    unsafe { libc::_exit(0); }

                    #[cfg(not(target_os = "macos"))]
                    std::process::exit(0);
                }
                _ => {}
            }
        });
}
