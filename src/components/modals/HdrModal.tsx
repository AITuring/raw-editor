import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { CheckCircle, XCircle, Loader2, Save, RefreshCw, Images } from 'lucide-react';
import { motion } from 'framer-motion';
import Button from '../ui/Button';
import TaskProgress from '../ui/TaskProgress';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';
import { getMessageTaskProgress } from '../../utils/taskProgress';

interface HdrModalProps {
  error: string | null;
  finalImageBase64: string | null;
  imageCount?: number;
  isOpen: boolean;
  isProcessing: boolean;
  loadingImageUrl?: string | null;
  onClose(): void;
  onOpenFile(path: string): void;
  onSave(): Promise<string>;
  onMerge(): void;
  progressMessage: string | null;
}

export default function HdrModal({
  error,
  finalImageBase64,
  imageCount,
  isOpen,
  isProcessing,
  loadingImageUrl,
  onClose,
  onOpenFile,
  onSave,
  onMerge,
  progressMessage,
}: HdrModalProps) {
  const { t } = useTranslation();
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const hdrProgress = useMemo(() => getMessageTaskProgress(progressMessage, 'hdr'), [progressMessage]);

  const mouseDownTarget = useRef<EventTarget | null>(null);

  useEffect(() => {
    if (isOpen) {
      setIsMounted(true);
      const timer = setTimeout(() => setShow(true), 10);
      return () => clearTimeout(timer);
    } else {
      setShow(false);
      const timer = setTimeout(() => {
        setIsMounted(false);
        setSavedPath(null);
        setIsSaving(false);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  const handleClose = useCallback(() => {
    if (isSaving) return;
    onClose();
  }, [onClose, isSaving]);

  const handleBackdropMouseDown = (e: React.MouseEvent) => {
    mouseDownTarget.current = e.target;
  };

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget && mouseDownTarget.current === e.currentTarget) {
      handleClose();
    }
    mouseDownTarget.current = null;
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const path = await onSave();
      setSavedPath(path);
    } catch (e) {
      console.error(e);
    } finally {
      setIsSaving(false);
    }
  };

  const handleOpen = () => {
    if (savedPath) {
      onOpenFile(savedPath);
      handleClose();
    }
  };

  const renderContent = () => {
    if (error) {
      return (
        <div className="flex flex-col items-center justify-center py-10 h-[460px]">
          <div className="flex items-center justify-center mb-6">
            <XCircle className="w-12 h-12 text-status-error" />
          </div>
          <Text variant={TextVariants.title} className="mb-2 text-center">
            {t('modals.hdr.failed')}
          </Text>
          <Text className="text-center p-4 rounded-lg bg-bg-primary max-w-md mt-2 leading-relaxed">
            {String(error)}
          </Text>
        </div>
      );
    }

    if (finalImageBase64 && !isProcessing) {
      return (
        <div className="w-full">
          <div className="flex max-h-[500px] w-full items-center justify-center overflow-hidden rounded-lg border border-border-color bg-[#111]">
            <img src={finalImageBase64} alt="Merged HDR" className="w-full h-full object-contain max-h-[500px]" />
          </div>
          {savedPath && (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
              <Text
                as="div"
                variant={TextVariants.heading}
                color={TextColors.success}
                className="flex items-center justify-center gap-2 mt-4"
              >
                <CheckCircle className="w-5 h-5" />
                <span>{t('modals.hdr.savedSuccess')}</span>
              </Text>
            </motion.div>
          )}
        </div>
      );
    }

    if (isProcessing) {
      return (
        <div className="flex h-[460px] overflow-hidden rounded-lg border border-border-color">
          <div className="w-2/5 relative overflow-hidden shrink-0 bg-[#0a0a0a] flex items-center justify-center">
            {loadingImageUrl ? (
              <img src={loadingImageUrl} alt="Source preview" className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full bg-surface/50" />
            )}
          </div>
          <div className="flex-1 flex flex-col items-center justify-center px-12 bg-bg-primary">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1, duration: 0.4 }}
              className="flex flex-col items-center w-full"
            >
              <Text variant={TextVariants.title} className="mb-2 text-center">
                {t('modals.hdr.merging')}
              </Text>
              <TaskProgress
                ariaLabel={t('modals.hdr.merging')}
                className="mt-5 max-w-sm"
                indeterminate={hdrProgress.value === null}
                label={progressMessage || t('modals.hdr.initializing')}
                showPercentage={hdrProgress.exact}
                value={hdrProgress.value}
              />

              <Text variant={TextVariants.small} className="mt-6 text-center max-w-xs opacity-60">
                {t('modals.hdr.speedNotice')}
              </Text>
            </motion.div>
          </div>
        </div>
      );
    }

    return (
      <div className="flex flex-col items-center justify-center h-[460px]">
        <div className="flex items-center justify-center mb-6">
          <Images className="w-12 h-12 text-accent" />
        </div>
        <Text variant={TextVariants.title} className="mb-3 text-center">
          {t('modals.hdr.title')}
        </Text>
        <Text className="text-center max-w-md leading-relaxed">
          {imageCount ? t('modals.hdr.descriptionWithCount', { count: imageCount }) : t('modals.hdr.description')}
        </Text>
      </div>
    );
  };

  const renderButtons = () => {
    if (error) {
      return (
        <Button onClick={handleClose} className="w-full">
          {t('modals.hdr.close')}
        </Button>
      );
    }

    if (savedPath) {
      return (
        <>
          <Button onClick={handleClose} variant="secondary">
            {t('modals.hdr.close')}
          </Button>
          <Button onClick={handleOpen}>{t('modals.hdr.openInEditor')}</Button>
        </>
      );
    }

    const disabled = isProcessing || isSaving;

    return (
      <div className={`w-full flex items-center justify-end gap-2 ${disabled ? 'opacity-50 pointer-events-none' : ''}`}>
        <Button onClick={handleClose} variant="secondary">
          {finalImageBase64 ? t('modals.hdr.close') : t('modals.hdr.cancel')}
        </Button>

        <Button onClick={onMerge} disabled={isProcessing} variant={finalImageBase64 ? 'secondary' : 'primary'}>
          {isProcessing ? (
            <Loader2 className="animate-spin mr-2" size={16} />
          ) : finalImageBase64 ? (
            <RefreshCw className="mr-2" size={16} />
          ) : (
            <Images className="mr-2" size={16} />
          )}
          {finalImageBase64 ? t('modals.hdr.retry') : t('modals.hdr.start')}
        </Button>

        {finalImageBase64 && (
          <Button onClick={handleSave} disabled={isSaving || isProcessing}>
            {isSaving ? <Loader2 className="animate-spin mr-2" size={16} /> : <Save className="mr-2" size={16} />}
            {t('modals.hdr.save')}
          </Button>
        )}
      </div>
    );
  };

  if (!isMounted) return null;

  return (
    <div
      className={`app-modal-backdrop ${show ? 'opacity-100' : 'opacity-0'}`}
      onMouseDown={handleBackdropMouseDown}
      onClick={handleBackdropClick}
    >
      <div
        className={`app-modal-surface app-modal-surface--padded max-w-4xl ${
          show ? 'translate-y-0 scale-100 opacity-100' : '-translate-y-2 scale-[0.98] opacity-0'
        }`}
        onClick={(e) => e.stopPropagation()}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex flex-col">
          {renderContent()}
          {isSaving && (
            <TaskProgress
              ariaLabel={t('modals.hdr.save')}
              className="mt-4"
              compact
              indeterminate
              label={t('modals.hdr.save')}
            />
          )}
          <div className={`app-modal-footer ${savedPath ? 'is-result' : ''}`}>{renderButtons()}</div>
        </div>
      </div>
    </div>
  );
}
