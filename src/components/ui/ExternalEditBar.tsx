import { Check, Loader } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import Button from './Button';
import TaskProgress from './TaskProgress';
import { ExternalEditSession, useProcessStore } from '../../store/useProcessStore';

interface ExternalEditBarProps {
  session: ExternalEditSession;
  isFinishing: boolean;
  errorMessage: string;
  onDone: () => void;
}

export default function ExternalEditBar({ session, isFinishing, errorMessage, onDone }: ExternalEditBarProps) {
  const { t } = useTranslation();
  const outputName = session.output.split(/[\\/]/).pop() || session.output;
  const progress = useProcessStore((state) => state.exportState.progress);

  return (
    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-40 flex w-[min(28rem,calc(100vw-2rem))] flex-col gap-2 bg-bg-secondary border border-surface rounded-lg shadow-lg px-4 py-2">
      <div className="flex items-center gap-3">
        <span className="min-w-0 truncate text-sm text-text-secondary whitespace-nowrap">
          {t('editor.externalEdit.savesTo')} <span className="text-text-primary">{outputName}</span>
        </span>
        {errorMessage && <span className="text-sm text-red-400 max-w-xs truncate">{errorMessage}</span>}
        <Button onClick={onDone} disabled={isFinishing} className="ml-auto shrink-0 py-1.5">
          {isFinishing ? <Loader size={16} className="animate-spin" /> : <Check size={16} />}
          {isFinishing ? t('editor.externalEdit.exporting') : t('editor.externalEdit.done')}
        </Button>
      </div>
      {isFinishing && (
        <TaskProgress
          ariaLabel={t('editor.externalEdit.exporting')}
          compact
          current={progress.current}
          indeterminate={progress.total <= 0 || (progress.total === 1 && progress.current === 0)}
          label={t('editor.externalEdit.exporting')}
          total={progress.total}
        />
      )}
    </div>
  );
}
