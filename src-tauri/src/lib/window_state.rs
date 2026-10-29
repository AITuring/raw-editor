//! Saved window state checks and the frontend-ready command.

use crate::app_state::AppState;
use crate::app_state::WindowState;
use crate::launch_request::LaunchPayload;

#[derive(Clone, Copy, Debug)]
pub(super) struct MonitorBounds {
    pub(super) x: i32,
    pub(super) y: i32,
    pub(super) width: u32,
    pub(super) height: u32,
}

pub(super) fn saved_window_state_is_usable(
    state: &WindowState,
    monitors: &[MonitorBounds],
) -> bool {
    if state.width < 800 || state.height < 600 {
        return false;
    }

    if monitors.is_empty() {
        return true;
    }

    let window_left = state.x as i64;
    let window_top = state.y as i64;
    let window_right = window_left + state.width as i64;
    let window_bottom = window_top + state.height as i64;

    monitors.iter().any(|monitor| {
        let monitor_left = monitor.x as i64;
        let monitor_top = monitor.y as i64;
        let monitor_right = monitor_left + monitor.width as i64;
        let monitor_bottom = monitor_top + monitor.height as i64;

        let overlap_width = window_right.min(monitor_right) - window_left.max(monitor_left);
        let overlap_height = window_bottom.min(monitor_bottom) - window_top.max(monitor_top);

        overlap_width >= 100 && overlap_height >= 100
    })
}

#[cfg(not(target_os = "android"))]
pub(super) fn available_monitor_bounds(window: &tauri::WebviewWindow) -> Vec<MonitorBounds> {
    window
        .available_monitors()
        .map(|monitors| {
            monitors
                .into_iter()
                .map(|monitor| {
                    let position = monitor.position();
                    let size = monitor.size();
                    MonitorBounds {
                        x: position.x,
                        y: position.y,
                        width: size.width,
                        height: size.height,
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(target_os = "android")]
pub(super) fn available_monitor_bounds(_window: &tauri::WebviewWindow) -> Vec<MonitorBounds> {
    Vec::new()
}

#[tauri::command]
pub(super) fn frontend_ready(
    app_handle: tauri::AppHandle,
    window: tauri::Window,
    state: tauri::State<AppState>,
) -> Result<LaunchPayload, String> {
    let is_first_run = !state
        .window_setup_complete
        .swap(true, std::sync::atomic::Ordering::Relaxed);
    #[cfg(target_os = "android")]
    let _ = (is_first_run, &window, &app_handle);

    #[cfg(not(target_os = "android"))]
    {
        #[cfg(any(windows, target_os = "linux"))]
        let mut should_maximize = false;
        #[cfg(any(windows, target_os = "linux"))]
        let mut should_fullscreen = false;
        #[cfg(not(any(windows, target_os = "linux")))]
        let _ = (&app_handle, is_first_run);

        #[cfg(any(windows, target_os = "linux"))]
        if is_first_run && let Ok(config_dir) = app_handle.path().app_config_dir() {
            let path = config_dir.join("window_state.json");

            if let Ok(contents) = std::fs::read_to_string(&path)
                && let Ok(saved_state) = serde_json::from_str::<WindowState>(&contents)
            {
                #[cfg(any(windows, target_os = "linux"))]
                {
                    should_maximize = saved_state.maximized;
                    should_fullscreen = saved_state.fullscreen;
                }

                if (should_maximize || should_fullscreen)
                    && let Some(monitor) = window
                        .current_monitor()
                        .ok()
                        .flatten()
                        .or_else(|| window.primary_monitor().ok().flatten())
                        .or_else(|| {
                            window
                                .available_monitors()
                                .ok()
                                .and_then(|m| m.into_iter().next())
                        })
                {
                    let monitor_size = monitor.size();
                    let monitor_pos = monitor.position();
                    let default_width = 1280i32;
                    let default_height = 720i32;
                    let center_x = monitor_pos.x + (monitor_size.width as i32 - default_width) / 2;
                    let center_y =
                        monitor_pos.y + (monitor_size.height as i32 - default_height) / 2;

                    let _ = window.set_size(tauri::PhysicalSize::new(
                        default_width as u32,
                        default_height as u32,
                    ));
                    let _ = window.set_position(tauri::PhysicalPosition::new(center_x, center_y));
                }
            }
        }

        if let Err(e) = window.show() {
            log::error!("Failed to show window: {}", e);
        }
        if let Err(e) = window.set_focus() {
            log::error!("Failed to focus window: {}", e);
        }
        #[cfg(any(windows, target_os = "linux"))]
        if is_first_run {
            if should_maximize {
                let _ = window.maximize();
            }
            if should_fullscreen {
                let _ = window.set_fullscreen(true);
            }
        }
    }

    let open_with_file = state.initial_file_path.lock().unwrap().take();
    let edit_session = state.pending_edit_session.lock().unwrap().take();
    if let Some(path) = &open_with_file {
        log::info!("Frontend is ready, returning initial path: {}", path);
    }
    if let Some(session) = &edit_session {
        log::info!(
            "Frontend is ready, returning external edit session for: {}",
            session.source
        );
    }
    Ok(LaunchPayload {
        open_with_file,
        edit_session,
    })
}
