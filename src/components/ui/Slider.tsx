import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { GLOBAL_KEYS } from './AppProperties';

type SliderChangeEvent =
  | React.ChangeEvent<HTMLInputElement>
  | {
      target: {
        value: number | string;
      };
    };

interface SliderProps {
  defaultValue?: number;
  disabled?: boolean;
  label: React.ReactNode;
  max: number;
  min: number;
  onChange(event: SliderChangeEvent): void;
  onDragStateChange?(state: boolean): void;
  step: number;
  value: number;
  trackClassName?: string;
  fillOrigin?: 'min' | 'default';
  suffix?: string;
}

const DOUBLE_CLICK_THRESHOLD_MS = 150;
const FINE_ADJUSTMENT_MULTIPLIER = 0.2;
const TOUCH_DRAG_THRESHOLD_PX = 10;
const TOUCH_THUMB_HIT_RADIUS_PX = 24;

const hasFineAdjustmentModifier = (event: MouseEvent | TouchEvent | React.MouseEvent | React.TouchEvent) =>
  'shiftKey' in event && (event.shiftKey || event.altKey);

const Slider = ({
  defaultValue = 0,
  disabled = false,
  label,
  max,
  min,
  onChange,
  onDragStateChange = () => {},
  step = 1,
  value,
  trackClassName,
  fillOrigin = 'default',
  suffix = '',
}: SliderProps) => {
  const { t } = useTranslation();
  const [displayValue, setDisplayValue] = useState<number>(value);
  const [isDragging, setIsDragging] = useState(false);
  const animationFrameRef = useRef<number | undefined>(undefined);
  const [isEditing, setIsEditing] = useState(false);
  const [inputValue, setInputValue] = useState<string>(String(value));
  const inputRef = useRef<HTMLInputElement | null>(null);
  const rangeInputRef = useRef<HTMLInputElement | null>(null);
  const [isLabelHovered, setIsLabelHovered] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const lastUpTime = useRef(0);
  const lastPointerXRef = useRef<number>(0);
  const accumulatedValueRef = useRef<number>(0);
  const pendingTouchRef = useRef<{
    startX: number;
    startY: number;
    latestX: number;
    startValue: number;
  } | null>(null);
  const suppressTouchChangeRef = useRef(false);
  const isWheelActivelyChangingRef = useRef(false);
  const wheelTimeoutRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    return () => {
      if (wheelTimeoutRef.current !== undefined) {
        window.clearTimeout(wheelTimeoutRef.current);
      }
    };
  }, []);

  const fillPercentage = max !== min ? ((displayValue - min) / (max - min)) * 100 : 0;
  const originPercentage = useMemo(() => {
    if (fillOrigin === 'min') {
      return 0;
    }
    return max !== min ? ((defaultValue - min) / (max - min)) * 100 : 0;
  }, [fillOrigin, defaultValue, min, max]);

  const stepStr = String(step);
  const decimalPlaces = stepStr.includes('.') ? stepStr.split('.')[1].length : 0;

  const snapToStep = useCallback(
    (val: number): number => {
      const snapped = Math.round((val - min) / step) * step + min;
      const clamped = Math.max(min, Math.min(max, snapped));
      return parseFloat(clamped.toFixed(decimalPlaces));
    },
    [min, max, step, decimalPlaces],
  );

  const onChangeRef = useRef(onChange);
  const snapToStepRef = useRef(snapToStep);
  const rangeRef = useRef({ min, max });

  onChangeRef.current = onChange;
  snapToStepRef.current = snapToStep;
  rangeRef.current = { min, max };

  useEffect(() => {
    onDragStateChange(isDragging);
  }, [isDragging, onDragStateChange]);

  useEffect(() => {
    if (!disabled) return;

    pendingTouchRef.current = null;
    suppressTouchChangeRef.current = false;
    isWheelActivelyChangingRef.current = false;

    if (wheelTimeoutRef.current !== undefined) {
      window.clearTimeout(wheelTimeoutRef.current);
      wheelTimeoutRef.current = undefined;
    }
    if (animationFrameRef.current !== undefined) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = undefined;
    }

    setIsDragging(false);
    setIsEditing(false);
    setIsLabelHovered(false);
    setDisplayValue(value);
    setInputValue(String(value));
  }, [disabled, value]);

  useEffect(() => {
    const sliderElement = containerRef.current;
    if (!sliderElement) return;

    const handleWheel = (event: WheelEvent) => {
      if (disabled || !event.shiftKey) {
        return;
      }

      event.preventDefault();
      const direction = -Math.sign(event.deltaY || event.deltaX);
      const newValue = value + direction * step;
      const roundedNewValue = parseFloat(newValue.toFixed(decimalPlaces));

      const clampedValue = Math.max(min, Math.min(max, roundedNewValue));

      if (clampedValue !== value && !isNaN(clampedValue)) {
        isWheelActivelyChangingRef.current = true;
        setDisplayValue(clampedValue);

        if (wheelTimeoutRef.current !== undefined) {
          window.clearTimeout(wheelTimeoutRef.current);
        }
        wheelTimeoutRef.current = window.setTimeout(() => {
          isWheelActivelyChangingRef.current = false;
        }, 150);

        const syntheticEvent = {
          target: {
            value: clampedValue,
          },
        };
        onChange(syntheticEvent);
      }
    };

    sliderElement.addEventListener('wheel', handleWheel, { passive: false });

    return () => {
      sliderElement.removeEventListener('wheel', handleWheel);
    };
  }, [disabled, value, min, max, step, onChange, decimalPlaces]);

  // Handle Dragging
  useEffect(() => {
    if (!isDragging || disabled) return;

    const inputEl = rangeInputRef.current;
    if (!inputEl) return;
    const sliderWidth = inputEl.getBoundingClientRect().width || 1;

    const handlePointerMove = (e: MouseEvent | TouchEvent) => {
      let clientX: number;
      let shiftKey: boolean;

      if ('touches' in e) {
        if (e.touches.length === 0) return;
        clientX = e.touches[0].clientX;
        shiftKey = hasFineAdjustmentModifier(e);
        if (e.cancelable) e.preventDefault();
      } else {
        clientX = (e as MouseEvent).clientX;
        shiftKey = hasFineAdjustmentModifier(e);
      }

      const deltaX = clientX - lastPointerXRef.current;
      const { min: curMin, max: curMax } = rangeRef.current;

      const multiplier = shiftKey ? FINE_ADJUSTMENT_MULTIPLIER : 1;
      const deltaValue = (deltaX / sliderWidth) * (curMax - curMin) * multiplier;

      const prevAccumulated = accumulatedValueRef.current;
      accumulatedValueRef.current = Math.max(curMin, Math.min(curMax, prevAccumulated + deltaValue));

      const actualDeltaValue = accumulatedValueRef.current - prevAccumulated;
      if (deltaValue !== 0) {
        lastPointerXRef.current += deltaX * (actualDeltaValue / deltaValue);
      } else {
        lastPointerXRef.current = clientX;
      }

      const snappedValue = snapToStepRef.current(accumulatedValueRef.current);

      setDisplayValue(snappedValue);
      onChangeRef.current({ target: { value: snappedValue } });
    };

    const handlePointerUp = () => {
      lastUpTime.current = Date.now();
      pendingTouchRef.current = null;
      suppressTouchChangeRef.current = false;
      setIsDragging(false);
    };

    window.addEventListener('mousemove', handlePointerMove, { passive: false });
    window.addEventListener('mouseup', handlePointerUp);
    window.addEventListener('touchmove', handlePointerMove, { passive: false });
    window.addEventListener('touchend', handlePointerUp);
    window.addEventListener('touchcancel', handlePointerUp);

    return () => {
      window.removeEventListener('mousemove', handlePointerMove);
      window.removeEventListener('mouseup', handlePointerUp);
      window.removeEventListener('touchmove', handlePointerMove);
      window.removeEventListener('touchend', handlePointerUp);
      window.removeEventListener('touchcancel', handlePointerUp);
    };
  }, [disabled, isDragging]);

  useEffect(() => {
    if (isDragging) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      return;
    }

    if (isWheelActivelyChangingRef.current) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      setDisplayValue(value);
      return;
    }

    const startValue = displayValue;
    const endValue = value;
    const duration = 300;
    let startTime: number | null = null;

    const easeInOut = (t: number) => t * t * (3 - 2 * t);

    const animate = (timestamp: number) => {
      if (!startTime) {
        startTime = timestamp;
      }

      const progress = timestamp - startTime;
      const linearFraction = Math.min(progress / duration, 1);
      const easedFraction = easeInOut(linearFraction);
      const currentValue = startValue + (endValue - startValue) * easedFraction;
      setDisplayValue(currentValue);

      if (linearFraction < 1) {
        animationFrameRef.current = requestAnimationFrame(animate);
      }
    };

    animationFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [value, isDragging]);

  useEffect(() => {
    if (!isEditing) {
      setInputValue(String(value));
    }
  }, [value, isEditing]);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [isEditing]);

  const handleReset = () => {
    if (disabled) return;

    const syntheticEvent = {
      target: {
        value: defaultValue,
      },
    };
    onChange(syntheticEvent);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled || suppressTouchChangeRef.current) {
      return;
    }

    if (!isDragging) {
      setDisplayValue(Number(e.target.value));
      onChange(e);
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLInputElement>) => {
    if (disabled) return;

    if (Date.now() - lastUpTime.current < DOUBLE_CLICK_THRESHOLD_MS) {
      e.preventDefault();
      return;
    }
    e.preventDefault();

    const rect = e.currentTarget.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const rawValue = min + fraction * (max - min);
    const snappedValue = snapToStep(rawValue);

    accumulatedValueRef.current = rawValue;
    lastPointerXRef.current = e.clientX;

    setIsDragging(true);
    setDisplayValue(snappedValue);
    onChange({ target: { value: snappedValue } });
  };

  const handleTouchStart = (e: React.TouchEvent<HTMLInputElement>) => {
    if (disabled) return;

    if (e.touches.length === 0) return;

    const touch = e.touches[0];
    suppressTouchChangeRef.current = true;

    const inputEl = rangeInputRef.current;
    if (!inputEl) return;

    const rect = inputEl.getBoundingClientRect();
    const fraction = max !== min ? (displayValue - min) / (max - min) : 0;
    const thumbX = rect.left + Math.max(0, Math.min(1, fraction)) * rect.width;

    if (Math.abs(touch.clientX - thumbX) > TOUCH_THUMB_HIT_RADIUS_PX) {
      pendingTouchRef.current = null;
      return;
    }

    pendingTouchRef.current = {
      startX: touch.clientX,
      startY: touch.clientY,
      latestX: touch.clientX,
      startValue: displayValue,
    };
  };

  const handleTouchMove = (e: React.TouchEvent<HTMLInputElement>) => {
    if (disabled) return;

    if (isDragging || !pendingTouchRef.current || e.touches.length === 0) return;

    const touch = e.touches[0];
    const pendingTouch = pendingTouchRef.current;
    pendingTouch.latestX = touch.clientX;

    const deltaX = touch.clientX - pendingTouch.startX;
    const deltaY = touch.clientY - pendingTouch.startY;

    if (Math.abs(deltaY) > TOUCH_DRAG_THRESHOLD_PX && Math.abs(deltaY) > Math.abs(deltaX)) {
      pendingTouchRef.current = null;
      return;
    }

    if (Math.abs(deltaX) < TOUCH_DRAG_THRESHOLD_PX || Math.abs(deltaX) < Math.abs(deltaY)) {
      return;
    }

    const inputEl = rangeInputRef.current;
    if (!inputEl) return;

    const rect = inputEl.getBoundingClientRect();
    const multiplier = hasFineAdjustmentModifier(e) ? FINE_ADJUSTMENT_MULTIPLIER : 1;
    const rawValue = pendingTouch.startValue + (deltaX / rect.width) * (max - min) * multiplier;
    const snappedValue = snapToStep(rawValue);

    accumulatedValueRef.current = rawValue;
    lastPointerXRef.current = touch.clientX;
    pendingTouchRef.current = null;

    if (e.cancelable) {
      e.preventDefault();
    }

    setIsDragging(true);
    setDisplayValue(snappedValue);
    onChange({ target: { value: snappedValue } });
  };

  const handleTouchEnd = () => {
    pendingTouchRef.current = null;
    suppressTouchChangeRef.current = false;
  };

  const handleValueClick = () => {
    if (disabled) return;

    setIsEditing(true);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return;

    const textVal = e.target.value;
    if (!/^[0-9.,-]*$/.test(textVal)) {
      return;
    }
    setInputValue(textVal);
    const parseableText = textVal.replace(',', '.');
    const parsedValue = parseFloat(parseableText);
    if (!isNaN(parsedValue)) {
      const clampedValue = Math.max(min, Math.min(max, parsedValue));
      onChange({
        target: {
          value: clampedValue,
        },
      });
    }
  };

  const handleInputCommit = () => {
    if (disabled) {
      setInputValue(String(value));
      setIsEditing(false);
      return;
    }

    let newValue = parseFloat(inputValue.replace(',', '.'));
    if (isNaN(newValue)) {
      newValue = value;
    } else {
      newValue = Math.max(min, Math.min(max, newValue));
    }
    const syntheticEvent = {
      target: {
        value: newValue,
      },
    };
    onChange(syntheticEvent);
    setIsEditing(false);
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (disabled) return;

    if (e.key === 'Enter') {
      handleInputCommit();
      e.currentTarget.blur();
    } else if (e.key === 'Escape') {
      setInputValue(String(value));
      setIsEditing(false);
      e.currentTarget.blur();
    } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      e.preventDefault();
      let currentNum = parseFloat(inputValue.replace(',', '.'));
      if (isNaN(currentNum)) {
        currentNum = value;
      }
      const direction = e.key === 'ArrowUp' ? 1 : -1;
      const newValue = currentNum + direction * step;
      const snappedNewValue = snapToStep(newValue);
      setInputValue(String(snappedNewValue));
      onChange({
        target: {
          value: snappedNewValue,
        },
      });
    }
  };

  const handleRangeKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.ctrlKey || e.metaKey) {
      e.currentTarget.blur();
      return;
    }
    if (GLOBAL_KEYS.includes(e.key)) {
      e.currentTarget.blur();
    }
  };

  const numericValue = isNaN(Number(value)) ? 0 : Number(value);
  const formattedValue = `${numericValue > 0 ? '+' : ''}${
    decimalPlaces > 0 && numericValue === 0 ? '0' : numericValue.toFixed(decimalPlaces)
  }`;

  return (
    <div className={`develop-slider group ${disabled ? 'opacity-45 cursor-not-allowed' : ''}`} ref={containerRef}>
      <div className="camera-raw-slider-header flex justify-between items-center gap-3">
        {typeof label === 'string' ? (
          <button
            aria-label={`${label}: ${t('ui.slider.reset')}`}
            className={`camera-raw-slider-label grid min-h-6 items-center text-left ${
              disabled ? '' : 'cursor-pointer'
            }`}
            disabled={disabled}
            onClick={handleReset}
            onMouseEnter={() => setIsLabelHovered(true)}
            onMouseLeave={() => setIsLabelHovered(false)}
            type="button"
          >
            <span
              aria-hidden={isLabelHovered}
              className={`col-start-1 row-start-1 text-[12px] font-normal leading-5 text-text-secondary select-none transition-opacity duration-150 ease-out ${
                isLabelHovered ? 'opacity-0' : 'opacity-100'
              }`}
            >
              {label}
            </span>
            <span
              aria-hidden={!isLabelHovered}
              className={`col-start-1 row-start-1 text-[12px] font-normal leading-5 text-text-primary select-none transition-opacity duration-150 ease-out pointer-events-none ${
                isLabelHovered ? 'opacity-100' : 'opacity-0'
              }`}
            >
              {t('ui.slider.reset')}
            </span>
          </button>
        ) : (
          <div className="camera-raw-slider-label grid">
            <span className="col-start-1 row-start-1 text-[12px] font-normal leading-5 text-text-secondary">
              {label}
            </span>
          </div>
        )}
        <div className="camera-raw-slider-value w-12 shrink-0 text-right tabular-nums">
          {isEditing ? (
            <input
              aria-label={typeof label === 'string' ? label : undefined}
              className="camera-raw-slider-number h-5 w-full rounded-[2px] border border-border-color bg-card-active px-1 py-0 text-right text-[12px] text-text-primary focus-visible:border-accent"
              disabled={disabled}
              max={max}
              min={min}
              onBlur={handleInputCommit}
              onChange={handleInputChange}
              onKeyDown={handleInputKeyDown}
              ref={inputRef}
              step={step}
              type="text"
              value={inputValue}
            />
          ) : (
            <button
              aria-label={typeof label === 'string' ? `${label}: ${numericValue}` : undefined}
              className={`camera-raw-slider-number block min-h-6 w-full text-right text-[12px] leading-5 text-text-primary select-none ${
                disabled ? '' : 'cursor-text'
              }`}
              disabled={disabled}
              onClick={disabled ? undefined : handleValueClick}
              onDoubleClick={disabled ? undefined : handleReset}
              data-tooltip={disabled ? undefined : t('ui.slider.clickToEdit')}
              type="button"
            >
              {formattedValue}
              {suffix && <span className="text-[10px] align-top inline-block mt-0.5 ml-0.5">{suffix}</span>}
            </button>
          )}
        </div>
      </div>

      <div className="camera-raw-slider-track relative h-3.5 w-full">
        <div
          className={`camera-raw-slider-rail absolute left-0 top-1/2 h-[3px] w-full -translate-y-1/2 rounded-full pointer-events-none ${
            trackClassName || ''
          }`}
        />
        {!trackClassName && (
          <div
            className={`camera-raw-slider-fill absolute top-1/2 h-[3px] -translate-y-1/2 rounded-full pointer-events-none ${
              isDragging ? 'is-active' : ''
            }`}
            style={{
              left: `${Math.min(fillPercentage, originPercentage)}%`,
              width: `${Math.abs(fillPercentage - originPercentage)}%`,
            }}
          />
        )}
        <input
          aria-label={typeof label === 'string' ? label : undefined}
          ref={rangeInputRef}
          className={`camera-raw-slider-input absolute left-0 top-1/2 z-10 m-0 h-5 w-full -translate-y-1/2 appearance-none bg-transparent p-0 slider-input ${
            isDragging ? 'slider-thumb-active' : ''
          } ${disabled ? 'cursor-not-allowed' : ''}`}
          disabled={disabled}
          style={{ margin: 0, touchAction: isDragging ? 'none' : 'pan-y' }}
          max={String(max)}
          min={String(min)}
          onChange={handleChange}
          onDoubleClick={handleReset}
          onKeyDown={handleRangeKeyDown}
          onMouseDown={handleMouseDown}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          onTouchCancel={handleTouchEnd}
          step={String(step)}
          type="range"
          value={displayValue}
        />
      </div>
    </div>
  );
};

export default Slider;
