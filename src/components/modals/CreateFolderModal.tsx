import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import Text from '../ui/Text';
import Button from '../ui/Button';
import Input from '../ui/Input';
import { TextVariants } from '../../types/typography';

interface FolderModalProps {
  isOpen: boolean;
  onClose(): void;
  onSave(name: string): void;
  title?: string;
  placeholder?: string;
  buttonText?: string;
}

export default function CreateFolderModal({
  isOpen,
  onClose,
  onSave,
  title,
  placeholder,
  buttonText,
}: FolderModalProps) {
  const { t } = useTranslation();
  const [name, setName] = useState('');
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setIsMounted(true);
      const timer = setTimeout(() => setShow(true), 10);
      return () => clearTimeout(timer);
    } else {
      setShow(false);
      const timer = setTimeout(() => {
        setIsMounted(false);
        setName('');
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  const handleSave = useCallback(() => {
    if (name.trim()) {
      onSave(name.trim());
    }
    onClose();
  }, [name, onSave, onClose]);

  const handleKeyDown = useCallback(
    (e: any) => {
      if (e.key === 'Enter') {
        handleSave();
      } else if (e.key === 'Escape') {
        onClose();
      }
    },
    [handleSave, onClose],
  );

  if (!isMounted) {
    return null;
  }

  return (
    <div
      aria-modal="true"
      className={`app-modal-backdrop ${show ? 'opacity-100' : 'opacity-0'}`}
      onClick={onClose}
      role="dialog"
    >
      <div
        className={`app-modal-surface app-modal-surface--padded max-w-sm ${
          show ? 'translate-y-0 scale-100 opacity-100' : '-translate-y-2 scale-[0.98] opacity-0'
        }`}
        onClick={(e: any) => e.stopPropagation()}
      >
        <Text variant={TextVariants.heading} className="mb-3">
          {title || t('modals.createFolder.title')}
        </Text>
        <Input
          autoFocus
          bgClassName="bg-bg-primary"
          onChange={(e: any) => setName(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder || t('modals.createFolder.placeholder')}
          type="text"
          value={name}
        />
        <div className="app-modal-actions">
          <Button variant="ghost" onClick={onClose}>
            {t('modals.createFolder.cancel')}
          </Button>
          <Button disabled={!name.trim()} onClick={handleSave}>
            {buttonText || t('modals.createFolder.create')}
          </Button>
        </div>
      </div>
    </div>
  );
}
