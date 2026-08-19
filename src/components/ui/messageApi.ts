import { isValidElement, type ReactNode } from 'react';

export type MessageType = 'info' | 'success' | 'warning' | 'error' | 'loading';
export type MessageKey = string | number;

export interface MessageOptions {
  content: ReactNode;
  duration?: number;
  key?: MessageKey;
  onClose?: () => void;
  pauseOnHover?: boolean;
  type?: MessageType;
}

export type TypedMessageOptions = Omit<MessageOptions, 'type'>;
export type MessageInput = ReactNode | TypedMessageOptions;

export interface MessageHandle {
  close(): void;
  key: MessageKey;
}

export interface MessageItem extends Required<Pick<MessageOptions, 'pauseOnHover' | 'type'>> {
  closing: boolean;
  content: ReactNode;
  duration: number;
  key: MessageKey;
  onClose?: () => void;
}

interface Countdown {
  remainingMs: number;
  startedAt: number;
  timerId: number | null;
}

const DEFAULT_DURATION_SECONDS = 3;
const EXIT_DURATION_MS = 140;
const MAX_VISIBLE_MESSAGES = 5;

let nextMessageId = 0;
let messageItems: MessageItem[] = [];

const listeners = new Set<() => void>();
const countdowns = new Map<MessageKey, Countdown>();
const exitTimers = new Map<MessageKey, number>();

export const getMessageSnapshot = () => messageItems;

export const subscribeToMessages = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

const emitChange = () => listeners.forEach((listener) => listener());
const now = () => (typeof performance === 'undefined' ? Date.now() : performance.now());

const clearCountdown = (key: MessageKey) => {
  const countdown = countdowns.get(key);
  if (countdown?.timerId !== null && countdown?.timerId !== undefined) {
    window.clearTimeout(countdown.timerId);
  }
  countdowns.delete(key);
};

const clearExitTimer = (key: MessageKey) => {
  const timerId = exitTimers.get(key);
  if (timerId !== undefined) window.clearTimeout(timerId);
  exitTimers.delete(key);
};

const removeMessage = (key: MessageKey) => {
  const item = messageItems.find((candidate) => candidate.key === key);
  if (!item) return;
  clearCountdown(key);
  clearExitTimer(key);
  messageItems = messageItems.filter((candidate) => candidate.key !== key);
  emitChange();
  item.onClose?.();
};

const beginClose = (key: MessageKey) => {
  const item = messageItems.find((candidate) => candidate.key === key);
  if (!item || item.closing) return;
  clearCountdown(key);
  messageItems = messageItems.map((candidate) => (candidate.key === key ? { ...candidate, closing: true } : candidate));
  emitChange();
  clearExitTimer(key);
  exitTimers.set(
    key,
    window.setTimeout(() => removeMessage(key), EXIT_DURATION_MS),
  );
};

const startCountdown = (key: MessageKey, remainingMs: number) => {
  clearCountdown(key);
  if (remainingMs <= 0) return;
  countdowns.set(key, {
    remainingMs,
    startedAt: now(),
    timerId: window.setTimeout(() => beginClose(key), remainingMs),
  });
};

export const pauseMessageCountdown = (key: MessageKey) => {
  const countdown = countdowns.get(key);
  if (!countdown || countdown.timerId === null) return;
  window.clearTimeout(countdown.timerId);
  countdown.remainingMs = Math.max(0, countdown.remainingMs - (now() - countdown.startedAt));
  countdown.timerId = null;
};

export const resumeMessageCountdown = (key: MessageKey) => {
  const countdown = countdowns.get(key);
  if (!countdown || countdown.timerId !== null) return;
  if (countdown.remainingMs <= 0) {
    beginClose(key);
    return;
  }
  countdown.startedAt = now();
  countdown.timerId = window.setTimeout(() => beginClose(key), countdown.remainingMs);
};

const isMessageOptions = (input: MessageInput): input is TypedMessageOptions =>
  typeof input === 'object' && input !== null && !isValidElement(input) && 'content' in input;

const openMessage = (options: MessageOptions): MessageHandle => {
  const type = options.type ?? 'info';
  const key = options.key ?? `app-message-${Date.now()}-${++nextMessageId}`;
  const duration = Math.max(0, options.duration ?? (type === 'loading' ? 0 : DEFAULT_DURATION_SECONDS));
  const nextItem: MessageItem = {
    closing: false,
    content: options.content,
    duration,
    key,
    onClose: options.onClose,
    pauseOnHover: options.pauseOnHover ?? true,
    type,
  };

  clearExitTimer(key);
  const existingIndex = messageItems.findIndex((item) => item.key === key);
  if (existingIndex >= 0) {
    messageItems = messageItems.map((item, index) => (index === existingIndex ? nextItem : item));
  } else {
    const visibleItems = messageItems.filter((item) => !item.closing);
    if (visibleItems.length >= MAX_VISIBLE_MESSAGES) removeMessage(visibleItems[0].key);
    messageItems = [...messageItems, nextItem];
  }
  emitChange();
  if (duration > 0) startCountdown(key, duration * 1000);
  else clearCountdown(key);

  return { key, close: () => beginClose(key) };
};

const openTypedMessage = (type: MessageType, input: MessageInput, duration?: number) => {
  const options = isMessageOptions(input) ? input : { content: input };
  return openMessage({ ...options, duration: duration ?? options.duration, type });
};

export const message = {
  destroy(key?: MessageKey) {
    if (key !== undefined) {
      beginClose(key);
      return;
    }
    messageItems.filter((item) => !item.closing).forEach((item) => beginClose(item.key));
  },
  error: (input: MessageInput, duration?: number) => openTypedMessage('error', input, duration),
  info: (input: MessageInput, duration?: number) => openTypedMessage('info', input, duration),
  loading: (input: MessageInput, duration?: number) => openTypedMessage('loading', input, duration),
  open: openMessage,
  success: (input: MessageInput, duration?: number) => openTypedMessage('success', input, duration),
  warning: (input: MessageInput, duration?: number) => openTypedMessage('warning', input, duration),
};
