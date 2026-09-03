import React from 'react';
import clsx from 'clsx';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  bgClassName?: string;
}

/**
 * A reusable text input component that matches the application's design system.
 * It uses theme variables for colors and consistent styling for focus and disabled states.
 *
 * @param {string} className - Additional classes to apply to the input.
 * @param {string} type - The type of the input (e.g., 'text', 'password', 'email').
 * @param {object} props - Other standard input props (value, onChange, placeholder, etc.).
 */
const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type = 'text', bgClassName, ...props }, ref) => {
    return (
      <input
        className={clsx(
          'ui-input',
          'file:border-0 file:bg-transparent file:text-sm file:font-medium',
          bgClassName,
          className,
        )}
        ref={ref}
        type={type}
        {...props}
      />
    );
  },
);

Input.displayName = 'Input';

export default Input;
