import { useState, useEffect, useCallback, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { useTranslation } from 'react-i18next';
import { CheckCircle, XCircle, Loader2, Users, Trash2, Star, Tag } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import { CullingSettings, CullingSuggestions, Invokes, Progress } from '../ui/AppProperties';
import Button from '../ui/Button';
import Switch from '../ui/Switch';
import Slider from '../ui/Slider';
import Dropdown from '../ui/Dropdown';
import TaskProgress from '../ui/TaskProgress';
import Text from '../ui/Text';
import { TextColors, TextVariants } from '../../types/typography';

interface CullingModalProps {
  isOpen: boolean;
  onClose(): void;
  progress: Progress | null;
  suggestions: CullingSuggestions | null;
  error: string | null;
  imagePaths: string[];
  thumbnails: Record<string, string>;
  onApply(action: 'reject' | 'rate_zero' | 'delete', paths: string[]): void;
  onError(error: string): void;
}

type CullAction = 'reject' | 'rate_zero' | 'delete';

function ImageThumbnail({ path, thumbnails, isSelected, onToggle, children }: any) {
  const thumbnailUrl = thumbnails[path];
  return (
    <div
      className={`relative group rounded-md overflow-hidden border-2 transition-colors cursor-pointer ${
        isSelected ? 'border-accent' : 'border-transparent hover:border-surface'
      }`}
      onClick={onToggle}
    >
      <img
        src={thumbnailUrl}
        alt={path}
        className={`w-full h-full object-cover transition-opacity ${isSelected ? 'opacity-100' : 'opacity-75 group-hover:opacity-100'}`}
      />
      <div
        className={`absolute inset-0 bg-black/50 transition-opacity ${
          isSelected ? 'opacity-0' : 'opacity-100 group-hover:opacity-0'
        }`}
      />
      <div className="absolute top-2 right-2">{isSelected && <CheckCircle size={16} className="text-accent" />}</div>
      {children && (
        <Text
          as="div"
          variant={TextVariants.small}
          color={TextColors.white}
          className="absolute bottom-0 left-0 right-0 p-1 bg-black/60"
        >
          {children}
        </Text>
      )}
    </div>
  );
}

export default function CullingModal({
  isOpen,
  onClose,
  progress,
  suggestions,
  error,
  imagePaths,
  thumbnails,
  onApply,
  onError,
}: CullingModalProps) {
  const { t } = useTranslation();
  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);
  const [stage, setStage] = useState<'settings' | 'progress' | 'results'>('settings');

  const [settings, setSettings] = useState<CullingSettings>({
    groupSimilar: true,
    similarityThreshold: 28,
    filterBlurry: true,
    blurThreshold: 100.0,
  });

  const [selectedRejects, setSelectedRejects] = useState<Set<string>>(new Set());
  const [action, setAction] = useState<CullAction>('reject');
  const [activeTab, setActiveTab] = useState<'similar' | 'blurry'>('similar');

  const CULL_ACTIONS = useMemo(
    () => [
      {
        value: 'reject' as const,
        label: t('modals.culling.actionReject'),
        icon: <Tag size={16} className="text-status-error" />,
      },
      { value: 'rate_zero' as const, label: t('modals.culling.actionRateZero'), icon: <Star size={16} /> },
      { value: 'delete' as const, label: t('modals.culling.actionDelete'), icon: <Trash2 size={16} /> },
    ],
    [t],
  );

  useEffect(() => {
    if (isOpen) {
      setIsMounted(true);
      const timer = setTimeout(() => setShow(true), 10);
      return () => clearTimeout(timer);
    } else {
      setShow(false);
      const timer = setTimeout(() => {
        setIsMounted(false);
        setStage('settings');
        setSelectedRejects(new Set());
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  useEffect(() => {
    if (suggestions || error) {
      setStage('results');
    } else if (progress) {
      setStage('progress');
    } else if (isOpen) {
      setStage('settings');
    }
  }, [progress, suggestions, error, isOpen]);

  useEffect(() => {
    if (stage === 'results' && suggestions) {
      const initialRejects = new Set<string>();
      suggestions.similarGroups.forEach((group) => {
        group.duplicates.forEach((dup) => initialRejects.add(dup.path));
      });
      suggestions.blurryImages.forEach((img) => initialRejects.add(img.path));
      setSelectedRejects(initialRejects);
    }
  }, [stage, suggestions]);

  const handleStartCulling = useCallback(async () => {
    try {
      await invoke(Invokes.CullImages, { paths: imagePaths, settings });
    } catch (err) {
      console.error('Culling failed to start:', err);
      onError(String(err));
    }
  }, [imagePaths, settings, onError]);

  const handleToggleReject = (path: string) => {
    setSelectedRejects((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(path)) {
        newSet.delete(path);
      } else {
        newSet.add(path);
      }
      return newSet;
    });
  };

  const handleApply = () => {
    onApply(action, Array.from(selectedRejects));
  };

  const numSimilar = suggestions?.similarGroups.reduce((acc, group) => acc + group.duplicates.length, 0) || 0;
  const numBlurry = suggestions?.blurryImages.length || 0;

  const renderSettings = () => (
    <>
      <div className="flex items-center justify-center mb-4">
        <Users className="w-12 h-12 text-accent" />
      </div>
      <Text variant={TextVariants.title} className="mb-6 text-center">
        {t('modals.culling.title')}
      </Text>
      <div className="space-y-6 text-sm">
        <div>
          <Switch
            label={t('modals.culling.groupSimilar')}
            checked={settings.groupSimilar}
            onChange={(v) => setSettings((s) => ({ ...s, groupSimilar: v }))}
          />
          {settings.groupSimilar && (
            <div className="mt-2 pl-4 border-l-2 border-border-color ml-1">
              <Slider
                label={t('modals.culling.similarityThreshold')}
                min={1}
                max={64}
                step={1}
                value={settings.similarityThreshold}
                defaultValue={28}
                onChange={(e) => setSettings((s) => ({ ...s, similarityThreshold: Number(e.target.value) }))}
                fillOrigin="min"
              />
              <Text variant={TextVariants.small} className="mt-1">
                {t('modals.culling.similarityThresholdDesc')}
              </Text>
            </div>
          )}
        </div>
        <div>
          <Switch
            label={t('modals.culling.filterBlurry')}
            checked={settings.filterBlurry}
            onChange={(v) => setSettings((s) => ({ ...s, filterBlurry: v }))}
          />
          {settings.filterBlurry && (
            <div className="mt-2  pl-4 border-l-2 border-border-color ml-1">
              <Slider
                label={t('modals.culling.blurThreshold')}
                min={25}
                max={500}
                step={25}
                value={settings.blurThreshold}
                defaultValue={100.0}
                onChange={(e) => setSettings((s) => ({ ...s, blurThreshold: Number(e.target.value) }))}
                fillOrigin="min"
              />
              <Text variant={TextVariants.small} className="mt-1">
                {t('modals.culling.blurThresholdDesc')}
              </Text>
            </div>
          )}
        </div>
      </div>
      <div className="flex justify-end gap-3 mt-8">
        <Button variant="secondary" onClick={onClose}>
          {t('modals.culling.cancel')}
        </Button>
        <Button onClick={handleStartCulling}>{t('modals.culling.startCulling')}</Button>
      </div>
    </>
  );

  const renderProgress = () => (
    <div className="flex flex-col items-center justify-center h-48">
      <Loader2 className="w-8 h-8 text-status-info animate-spin" />
      <TaskProgress
        ariaLabel={progress?.stage || t('modals.culling.starting')}
        className="mt-5 max-w-xl"
        current={progress?.current}
        indeterminate={!progress || progress.total <= 0}
        label={progress?.stage || t('modals.culling.starting')}
        total={progress?.total}
      />
    </div>
  );

  const renderResults = () => {
    if (error) {
      return (
        <div className="flex flex-col items-center justify-center h-48">
          <XCircle className="w-16 h-16 text-status-error" />
          <Text variant={TextVariants.heading} className="mt-4 text-center">
            {t('modals.culling.cullingFailed')}
          </Text>
          <Text>{error}</Text>
          <div className="mt-6">
            <Button onClick={onClose}>{t('modals.culling.close')}</Button>
          </div>
        </div>
      );
    }

    if (!suggestions) return null;

    const totalSuggestions = numSimilar + numBlurry;
    if (totalSuggestions === 0) {
      return (
        <div className="flex flex-col items-center justify-center h-48">
          <CheckCircle className="w-16 h-16 text-status-success" />
          <Text variant={TextVariants.heading} className="mt-4">
            {t('modals.culling.noIssuesFound')}
          </Text>
          <Text>{t('modals.culling.noIssuesDesc')}</Text>
          <div className="mt-6">
            <Button onClick={onClose}>{t('modals.culling.done')}</Button>
          </div>
        </div>
      );
    }

    return (
      <>
        <Text variant={TextVariants.title} className="mb-4">
          {t('modals.culling.cullingSuggestions')}
        </Text>
        <div className="mb-4 border-b border-border-color">
          <nav className="-mb-px flex space-x-4" aria-label={t('modals.culling.title')}>
            {numSimilar > 0 && (
              <button
                aria-pressed={activeTab === 'similar'}
                onClick={() => setActiveTab('similar')}
                className={`ui-tab-action ${activeTab === 'similar' ? 'is-active' : ''}`}
                type="button"
              >
                {t('modals.culling.similarGroupsTab')} <span className="ui-count-badge">{numSimilar}</span>
              </button>
            )}
            {numBlurry > 0 && (
              <button
                aria-pressed={activeTab === 'blurry'}
                onClick={() => setActiveTab('blurry')}
                className={`ui-tab-action ${activeTab === 'blurry' ? 'is-active' : ''}`}
                type="button"
              >
                {t('modals.culling.blurryImagesTab')} <span className="ui-count-badge">{numBlurry}</span>
              </button>
            )}
          </nav>
        </div>

        <div className="bg-bg-primary rounded-lg p-2 h-[50vh] overflow-y-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
            >
              {activeTab === 'similar' && (
                <div className="space-y-4">
                  {suggestions.similarGroups.map((group, index) => (
                    <div key={index} className="bg-surface rounded-lg p-3">
                      <Text variant={TextVariants.heading} className="mb-2">
                        {t('modals.culling.groupHeader', { index: index + 1 })}
                      </Text>
                      <div className="grid grid-cols-[1fr_3fr] gap-3">
                        <div>
                          <Text variant={TextVariants.label} className="mb-1">
                            {t('modals.culling.bestImage')}
                          </Text>
                          <div className="relative rounded-md overflow-hidden border-2 border-green-500">
                            <img
                              src={thumbnails[group.representative.path]}
                              alt="Representative"
                              className="w-full h-full object-cover"
                            />
                            <Text
                              as="div"
                              variant={TextVariants.small}
                              color={TextColors.white}
                              className="absolute bottom-0 left-0 right-0 p-1 bg-black/60"
                            >
                              {t('modals.culling.score', { score: group.representative.qualityScore.toFixed(2) })}
                            </Text>
                          </div>
                        </div>
                        <div>
                          <Text variant={TextVariants.label} className="mb-1">
                            {t('modals.culling.duplicatesHeader', { count: group.duplicates.length })}
                          </Text>
                          <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2">
                            {group.duplicates.map((dup) => (
                              <ImageThumbnail
                                key={dup.path}
                                path={dup.path}
                                thumbnails={thumbnails}
                                isSelected={selectedRejects.has(dup.path)}
                                onToggle={() => handleToggleReject(dup.path)}
                              >
                                {t('modals.culling.score', { score: dup.qualityScore.toFixed(2) })}
                              </ImageThumbnail>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {activeTab === 'blurry' && (
                <div className="grid grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-2">
                  {suggestions.blurryImages.map((img) => (
                    <ImageThumbnail
                      key={img.path}
                      path={img.path}
                      thumbnails={thumbnails}
                      isSelected={selectedRejects.has(img.path)}
                      onToggle={() => handleToggleReject(img.path)}
                    >
                      {t('modals.culling.sharpness', { sharpness: img.sharpnessMetric.toFixed(0) })}
                    </ImageThumbnail>
                  ))}
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>

        <div className="app-modal-footer culling-modal-footer">
          <div className="flex-1">
            <Dropdown
              options={CULL_ACTIONS.map(({ value, label }) => ({ value, label }))}
              value={action}
              onChange={(newValue: CullAction) => setAction(newValue)}
              className="w-full"
            />
          </div>
          <div className="flex gap-3">
            <Button variant="secondary" onClick={onClose}>
              {t('modals.culling.cancel')}
            </Button>
            <Button onClick={handleApply} disabled={selectedRejects.size === 0}>
              {t('modals.culling.applyButton', { count: selectedRejects.size })}
            </Button>
          </div>
        </div>
      </>
    );
  };

  const renderContent = () => {
    switch (stage) {
      case 'settings':
        return renderSettings();
      case 'progress':
        return renderProgress();
      case 'results':
        return renderResults();
      default:
        return null;
    }
  };

  if (!isMounted) return null;

  return (
    <div
      className={`app-modal-backdrop ${show ? 'opacity-100' : 'opacity-0'}`}
      onClick={onClose}
      role="dialog"
      aria-modal="true"
    >
      <div
        className={`app-modal-surface app-modal-surface--padded max-w-3xl ${
          show ? 'translate-y-0 scale-100 opacity-100' : '-translate-y-2 scale-[0.98] opacity-0'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        {renderContent()}
      </div>
    </div>
  );
}
