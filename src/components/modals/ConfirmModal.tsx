import { useEffect, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import Button, { type ButtonVariant } from '../ui/Button';
import Text from '../ui/Text';
import { TextVariants } from '../../types/typography';

interface ConfirmModalProps {
  cancelText?: string;
  confirmText?: string;
  confirmVariant?: ButtonVariant;
  isOpen: boolean;
  message?: string;
  onClose(): void;
  onConfirm?(): void;
  title?: string;
}

export default function ConfirmModal({
  cancelText,
  confirmText,
  confirmVariant = 'primary',
  isOpen,
  message,
  onClose,
  onConfirm,
  title,
}: ConfirmModalProps) {
  const { t } = useTranslation();
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);

  const resolvedCancelText = cancelText || t('modals.confirm.cancel');
  const resolvedConfirmText = confirmText || t('modals.confirm.confirm');

  useEffect(() => {
    if (isOpen) {
      setIsMounted(true);
      const timer = setTimeout(() => {
        setShow(true);
      }, 10);
      return () => clearTimeout(timer);
    } else {
      setShow(false);
      const timer = setTimeout(() => {
        setIsMounted(false);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  const handleConfirm = useCallback(() => {
    if (onConfirm) {
      onConfirm();
    }
    onClose();
  }, [onConfirm, onClose]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        e.stopPropagation();
        e.nativeEvent.stopImmediatePropagation();
        handleConfirm();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        e.nativeEvent.stopImmediatePropagation();
        onClose();
      }
    },
    [handleConfirm, onClose],
  );

  if (!isMounted) {
    return null;
  }

  return (
    <div
      aria-labelledby="confirm-modal-title"
      aria-modal="true"
      className={`app-modal-backdrop ${show ? 'opacity-100' : 'opacity-0'}`}
      onClick={onClose}
      role="dialog"
    >
      <div
        className={`app-modal-surface app-modal-surface--padded max-w-md ${
          show ? 'translate-y-0 scale-100 opacity-100' : '-translate-y-2 scale-[0.98] opacity-0'
        }`}
        onClick={(e: any) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <Text variant={TextVariants.heading} id="confirm-modal-title" className="mb-3">
          {title}
        </Text>
        <Text className="whitespace-pre-wrap leading-5">{message}</Text>
        <div className="app-modal-actions">
          <Button onClick={onClose} variant="ghost" tabIndex={0}>
            {resolvedCancelText}
          </Button>
          <Button onClick={handleConfirm} variant={confirmVariant} autoFocus={true}>
            {resolvedConfirmText}
          </Button>
        </div>
      </div>
    </div>
  );
}
