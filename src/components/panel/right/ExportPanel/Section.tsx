import Text from '../../../ui/Text';
import { TextVariants } from '../../../../types/typography';

interface SectionProps {
  children: any;
  title: string;
}

export function Section({ title, children }: SectionProps) {
  return (
    <div>
      <Text variant={TextVariants.heading} className="mb-2">
        {title}
      </Text>
      <div className="space-y-2">{children}</div>
    </div>
  );
}
