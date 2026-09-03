import { useState, useEffect, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { FILENAME_VARIABLES } from '../ui/ExportImportProperties';
import Text from '../ui/Text';
import Button from '../ui/Button';
import Input from '../ui/Input';
import { TextVariants } from '../../types/typography';

interface RenameFileModalProps {
  filesToRename: Array<string>;
  isOpen: boolean;
  onClose(): void;
  onSave(template: any): void;
}

export default function RenameFileModal({ filesToRename, isOpen, onClose, onSave }: RenameFileModalProps) {
  const { t } = useTranslation();
  const [nameTemplate, setNameTemplate] = useState('');
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);
  const nameInputRef = useRef<HTMLInputElement>(null);

  const fileCount = filesToRename.length;
  const isSingleFile = fileCount === 1;

  useEffect(() => {
    if (isOpen) {
      if (isSingleFile && filesToRename[0]) {
        const fileName = filesToRename[0].split(/[\\/]/).pop();
        const nameWithoutExt = fileName?.substring(0, fileName.lastIndexOf('.'));
        if (nameWithoutExt) {
          setNameTemplate(nameWithoutExt);
        }
      } else {
        setNameTemplate('{original_filename}');
      }
      setIsMounted(true);
      const timer = setTimeout(() => setShow(true), 10);
      return () => clearTimeout(timer);
    } else {
      setShow(false);
      const timer = setTimeout(() => {
        setIsMounted(false);
        setNameTemplate('');
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isOpen, filesToRename, isSingleFile]);

  const handleSave = useCallback(() => {
    if (nameTemplate.trim()) {
      let finalTemplate = nameTemplate.trim();
      if (!isSingleFile && !finalTemplate.includes('{sequence}') && !finalTemplate.includes('{original_filename}')) {
        finalTemplate = `${finalTemplate}_{sequence}`;
      }
      onSave(finalTemplate);
    }
    onClose();
  }, [nameTemplate, onSave, onClose, isSingleFile]);

  const handleKeyDown = useCallback(
    (e: any) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleSave();
      } else if (e.key === 'Escape') {
        onClose();
      }
    },
    [handleSave, onClose],
  );

  const handleVariableClick = (variable: string) => {
    if (!nameInputRef.current) {
      return;
    }
    const input = nameInputRef.current;
    const start = input?.selectionStart || 0;
    const end = input?.selectionEnd || 0;
    const currentValue = input.value;
    const newValue = currentValue.substring(0, start) + variable + currentValue.substring(end);
    setNameTemplate(newValue);
    setTimeout(() => {
      input.focus();
      const newCursorPos = start + variable.length;
      input.setSelectionRange(newCursorPos, newCursorPos);
    }, 0);
  };

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
        className={`app-modal-surface app-modal-surface--padded max-w-lg ${
          show ? 'translate-y-0 scale-100 opacity-100' : '-translate-y-2 scale-[0.98] opacity-0'
        }`}
        onClick={(e: any) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        <Text variant={TextVariants.heading} className="mb-4">
          {isSingleFile
            ? t('modals.renameFile.titleSingle')
            : t('modals.renameFile.titleMultiple', { count: fileCount })}
        </Text>

        <div className="text-sm">
          <div>
            <Text variant={TextVariants.heading} className="block mb-2">
              {isSingleFile ? t('modals.renameFile.newName') : t('modals.renameFile.fileNamingTemplate')}
            </Text>
            <Input
              autoFocus
              bgClassName="bg-bg-primary"
              onChange={(e: any) => setNameTemplate(e.target.value)}
              ref={nameInputRef}
              type="text"
              value={nameTemplate}
            />
            {!isSingleFile && (
              <div className="flex flex-wrap gap-2 mt-2">
                {FILENAME_VARIABLES.map((variable: string) => (
                  <button
                className="ui-surface-button min-h-7 rounded-sm bg-bg-primary px-2 py-1 text-[11px] text-text-secondary transition-colors hover:bg-card-active hover:text-text-primary"
                    key={variable}
                    onClick={() => handleVariableClick(variable)}
                  >
                    {variable}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="app-modal-actions">
          <Button variant="ghost" onClick={onClose}>
            {t('modals.renameFile.cancel')}
          </Button>
          <Button disabled={!nameTemplate.trim()} onClick={handleSave}>
            {t('modals.renameFile.save')}
          </Button>
        </div>
      </div>
    </div>
  );
}
