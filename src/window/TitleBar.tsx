import { useCallback, useState, useEffect } from 'react';
import { platform } from '@tauri-apps/plugin-os';
import { getCurrentWindow } from '@tauri-apps/api/window';
import { isTauri } from '@tauri-apps/api/core';
import { Minus, Square, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';
import { APP_NAME } from '../config/app';

const RestoreDownIcon = ({ size = 14, className = '' }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <rect x="3" y="6" width="8" height="8" rx="1.5" />
    <path d="M6 6V4.5A1.5 1.5 0 0 1 7.5 3h5A1.5 1.5 0 0 1 14 4.5v5A1.5 1.5 0 0 1 12.5 11H11" />
  </svg>
);

export default function TitleBar() {
  const { t } = useTranslation();
  const [osPlatform, setOsPlatform] = useState('');
  const [isMaximized, setIsMaximized] = useState(false);

  const tauriRuntime = isTauri();
  const appWindow = tauriRuntime ? getCurrentWindow() : null;

  useEffect(() => {
    const getPlatform = async () => {
      if (!tauriRuntime) {
        setOsPlatform(navigator.userAgent.includes('Mac') ? 'macos' : 'windows');
        return;
      }
      try {
        const p = platform();
        setOsPlatform(p);
      } catch (error) {
        console.error('Failed to get platform:', error);
        setOsPlatform('windows');
      }
    };
    getPlatform();
  }, [tauriRuntime]);

  useEffect(() => {
    if (!appWindow) return;

    const updateMaximizedState = async () => {
      try {
        const max = await appWindow.isMaximized();
        setIsMaximized(max);
      } catch (error) {
        console.error('Failed to check maximized state:', error);
      }
    };

    updateMaximizedState();

    let unlisten: () => void;
    appWindow
      .onResized(() => {
        updateMaximizedState();
      })
      .then((u) => (unlisten = u));

    return () => {
      if (unlisten) unlisten();
    };
  }, [appWindow]);

  const handleMinimize = () => appWindow?.minimize();
  const handleClose = () => appWindow?.close();

  const handleMaximize = useCallback(async () => {
    try {
      if (!appWindow) return;
      if (osPlatform === 'macos') {
        const isFullscreen = await appWindow.isFullscreen();
        appWindow.setFullscreen(!isFullscreen);
      } else {
        appWindow.toggleMaximize();
      }
    } catch (error) {
      console.error('Failed to toggle maximize:', error);
    }
  }, [osPlatform, appWindow]);

  const isMac = osPlatform === 'macos';
  const isLinux = osPlatform === 'linux';
  const isWindows = osPlatform === 'windows';
  const isMobile = osPlatform === 'android';
  if (!osPlatform || isMobile) {
    return null;
  }
  const outerDragProps = isLinux ? {} : { 'data-tauri-drag-region': 'true' };

  return (
    <div className="relative z-50 w-full border-b border-border-color bg-bg-secondary" {...outerDragProps}>
      <div
        className="h-9 bg-bg-secondary flex justify-between items-center select-none overflow-hidden"
        {...outerDragProps}
      >
        <div className="flex items-center h-full">
          {isMac && (
            <div className="flex items-center h-full px-4 space-x-2 z-10">
              <button
                aria-label={t('ui.windowControls.close')}
                className="w-3 h-3 rounded-full bg-red-500 hover:bg-red-600 transition-colors duration-150"
                onClick={handleClose}
              />
              <button
                aria-label={t('ui.windowControls.minimize')}
                className="w-3 h-3 rounded-full bg-yellow-500 hover:bg-yellow-600 transition-colors duration-150"
                onClick={handleMinimize}
              />
              <button
                aria-label={t('ui.windowControls.maximize')}
                className="w-3 h-3 rounded-full bg-green-500 hover:bg-green-600 transition-colors duration-150"
                onClick={handleMaximize}
              />
            </div>
          )}
          <div data-tauri-drag-region className={`flex items-center h-full ${isMac ? '' : 'px-4'}`}>
            <p className="text-[11px] font-medium tracking-[0.01em] text-text-secondary pointer-events-none">
              {APP_NAME}
            </p>
          </div>
        </div>
        <div data-tauri-drag-region className="flex-1 h-full" />
        <div className="flex items-center h-full z-10">
          {isLinux && (
            <div className="flex items-center gap-2 pr-2 h-full">
              <button
                aria-label={t('ui.windowControls.minimize')}
                className="w-7 h-7 rounded-full inline-flex justify-center items-center hover:bg-white/10 transition-colors duration-150"
                onClick={handleMinimize}
              >
                <Minus size={16} className="text-text-secondary" />
              </button>
              <button
                aria-label={t(isMaximized ? 'ui.windowControls.restore' : 'ui.windowControls.maximize')}
                className="w-7 h-7 rounded-full inline-flex justify-center items-center hover:bg-white/10 transition-colors duration-150"
                onClick={handleMaximize}
              >
                {isMaximized ? (
                  <RestoreDownIcon size={13} className="text-text-secondary" />
                ) : (
                  <Square size={13} className="text-text-secondary" />
                )}
              </button>
              <button
                aria-label={t('ui.windowControls.close')}
                className="w-7 h-7 rounded-full inline-flex justify-center items-center hover:bg-red-500 hover:text-white transition-colors duration-150"
                onClick={handleClose}
              >
                <X size={16} className="text-text-secondary hover:text-white" />
              </button>
            </div>
          )}

          {isWindows && <div data-tauri-drag-region className="w-36 shrink-0 h-full pointer-events-none" />}
        </div>
      </div>

      {isWindows && (
        <div className="absolute top-0 right-0 flex h-9 z-20">
          <button aria-label={t('ui.windowControls.minimize')} className="relative w-12 group" onClick={handleMinimize}>
            <div className="absolute inset-0 flex justify-center items-center group-hover:bg-white/10 group-active:bg-white/20 transition-colors duration-150">
              <Minus size={16} className="text-text-secondary" />
            </div>
          </button>

          <button
            aria-label={t(isMaximized ? 'ui.windowControls.restore' : 'ui.windowControls.maximize')}
            className="relative w-12 group"
            onClick={handleMaximize}
          >
            <div className="absolute inset-0 flex justify-center items-center group-hover:bg-white/10 group-active:bg-white/20 transition-colors duration-150">
              {isMaximized ? (
                <RestoreDownIcon size={12} className="text-text-secondary" />
              ) : (
                <Square size={12} className="text-text-secondary" />
              )}
            </div>
          </button>

          <button aria-label={t('ui.windowControls.close')} className="relative w-14 group" onClick={handleClose}>
            <div className="absolute inset-0 flex justify-center items-center group-hover:bg-red-500 group-active:bg-red-600 transition-colors duration-150">
              <X size={16} className="text-text-secondary group-hover:text-white transition-colors duration-150" />
            </div>
          </button>
        </div>
      )}
    </div>
  );
}
