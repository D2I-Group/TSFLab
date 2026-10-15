// Small presentational pieces for the results table header + rank column.

// Top-3 get a gold-tinted badge; everyone else a plain muted rank. No emoji —
// the medal is carried by the accent, on-brand with the rest of the site.
export function RankBadge({ rank }: { rank: number }) {
  if (rank <= 3) {
    return (
      <span className="inline-flex h-6 w-6 items-center justify-center rounded-full border border-accent/50 bg-accent-soft text-xs font-semibold text-accent [font-variant-numeric:tabular-nums]">
        {rank}
      </span>
    );
  }
  return (
    <span className="text-sm text-muted [font-variant-numeric:tabular-nums]">{rank}</span>
  );
}

export function Th({
  children,
  className = "",
  title,
}: {
  children: React.ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <th
      title={title}
      className={`border-b border-border bg-surface px-5 py-2.5 text-left text-[0.7rem] font-medium uppercase tracking-[0.04em] text-faint ${
        title ? "cursor-help" : ""
      } ${className}`}
    >
      {children}
    </th>
  );
}

export function SortTh({
  label,
  active,
  dir,
  onClick,
}: {
  label: string;
  active: boolean;
  dir: number;
  onClick: () => void;
}) {
  return (
    <th className="border-b border-border bg-surface px-5 py-2.5 text-left text-[0.7rem] font-medium uppercase tracking-[0.04em]">
      <button
        type="button"
        onClick={onClick}
        className={`inline-flex items-center gap-1 uppercase tracking-[0.04em] transition-colors ${
          active ? "text-accent" : "text-faint hover:text-ink"
        }`}
      >
        {label}
        {active ? (dir > 0 ? "↑" : "↓") : ""}
      </button>
    </th>
  );
}
