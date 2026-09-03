import StyleTransferModal from '../modals/StyleTransferModal';

interface StyleTransferViewProps {
  onBack(): void;
}

/**
 * The style-transfer workflow is a first-class app view. The implementation
 * component still owns the image pipeline, while this view keeps routing and
 * presentation separate from the library's modal collection.
 */
export default function StyleTransferView({ onBack }: StyleTransferViewProps) {
  return <StyleTransferModal fullPage isOpen onClose={onBack} />;
}
