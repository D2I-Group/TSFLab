// Unified segmented toggle button. One active treatment (filled accent) and
// one shape (rounded-md) across every "pick one" control on the page; only the
// size varies — lg (primary categories), md (tracks/views), sm (filters).
export function Seg({
  active,
  onClick,
  disabled = false,
  size = "sm",
  className = "",
  children,
}: {
  active: boolean;
  onClick: () => void;
  disabled?: boolean;
  size?: "lg" | "md" | "sm";
  className?: string;
  children: React.ReactNode;
}) {
  const sizes = {
    lg: "px-5 py-2 text-sm font-medium",
    md: "px-3 py-1.5 text-sm font-medium",
    sm: "px-2.5 py-1 text-xs font-medium",
  } as const;
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`rounded-md border transition-colors ${sizes[size]} ${
        active
          ? "border-accent bg-accent text-accent-fg shadow-sm"
          : "border-border bg-surface text-muted hover:text-ink hover:border-accent/50 disabled:opacity-50 disabled:cursor-not-allowed"
      } ${className}`}
    >
      {children}
    </button>
  );
}
