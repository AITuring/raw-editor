import { useSyncExternalStore } from 'react';
import { createPortal } from 'react-dom';
import { Check, Info, LoaderCircle, TriangleAlert, X } from 'lucide-react';
import {
  getMessageSnapshot,
  pauseMessageCountdown,
  resumeMessageCountdown,
  subscribeToMessages,
  type MessageType,
} from './messageApi';

interface MessageHostProps {
  topOffset?: number;
}

function MessageIcon({ type }: { type: MessageType }) {
  if (type === 'loading') {
    return (
      <span aria-hidden="true" className="app-message__icon app-message__icon--loading">
        <LoaderCircle size={17} strokeWidth={2} />
      </span>
    );
  }

  return (
    <span aria-hidden="true" className="app-message__icon">
      {type === 'info' && <Info size={12} strokeWidth={2.7} />}
      {type === 'success' && <Check size={12} strokeWidth={3} />}
      {type === 'warning' && <TriangleAlert size={12} strokeWidth={2.4} />}
      {type === 'error' && <X size={12} strokeWidth={3} />}
    </span>
  );
}

export function MessageHost({ topOffset = 16 }: MessageHostProps) {
  const items = useSyncExternalStore(subscribeToMessages, getMessageSnapshot, getMessageSnapshot);
  if (typeof document === 'undefined') return null;

  return createPortal(
    <div className="app-message-host" style={{ top: topOffset }}>
      {items.map((item) => (
        <div
          aria-atomic="true"
          aria-live={item.type === 'error' || item.type === 'warning' ? 'assertive' : 'polite'}
          className="app-message"
          data-closing={item.closing ? 'true' : 'false'}
          data-type={item.type}
          key={item.key}
          onPointerEnter={() => item.pauseOnHover && pauseMessageCountdown(item.key)}
          onPointerLeave={() => item.pauseOnHover && resumeMessageCountdown(item.key)}
          role={item.type === 'error' || item.type === 'warning' ? 'alert' : 'status'}
        >
          <MessageIcon type={item.type} />
          <div className="app-message__content">{item.content}</div>
        </div>
      ))}
    </div>,
    document.body,
  );
}
