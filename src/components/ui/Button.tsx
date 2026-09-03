import clsx from 'clsx';
import type { ButtonHTMLAttributes, ReactNode } from 'react';

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'destructive';

interface ButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children' | 'size'> {
  children: ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'icon';
  variant?: ButtonVariant;
}

const Button = ({ children, onClick, disabled, className = '', size = 'md', variant, ...props }: ButtonProps) => {
  const resolvedVariant = variant
    ? variant
    : className.includes('bg-transparent') || className.includes('bg-bg-primary')
      ? 'ghost'
      : className.includes('bg-surface')
        ? 'secondary'
        : 'primary';
  const combinedClasses = clsx(
    'ui-button',
    `ui-button--${resolvedVariant}`,
    size !== 'md' && `ui-button--${size}`,
    className,
  );

  return (
    <button onClick={onClick} disabled={disabled} className={combinedClasses} {...props}>
      {children}
    </button>
  );
};

export default Button;
