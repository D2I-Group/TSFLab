// Metric column definitions + value formatters for the results table.

// Numeric, sortable metric columns. `corr` is higher-is-better; the rest are
// errors (lower-is-better) — captured by `betterDir` so the default sort and
// the arrow glyph stay honest.
export const METRICS = [
  { key: "mse", betterDir: 1 },
  { key: "mae", betterDir: 1 },
  { key: "wape", betterDir: 1 },
  { key: "rse", betterDir: 1 },
  { key: "corr", betterDir: -1 },
] as const;

// betterDir is the sort direction that puts the BEST value first: 1 = ascending
// (lower is better), -1 = descending (higher is better). For quant: returns /
// Sharpe / win-rate are higher-better; max drawdown is reported negative, so a
// value closer to 0 (higher) is better; turnover is lower-better.
export const QUANT_METRICS = [
  { key: "total_return", label: "Total Return", betterDir: -1 },
  { key: "annualized_return", label: "Annualized", betterDir: -1 },
  { key: "sharpe", label: "Sharpe", betterDir: -1 },
  { key: "max_drawdown", label: "Max DD", betterDir: -1 },
  { key: "win_rate", label: "Win Rate", betterDir: -1 },
  { key: "avg_turnover", label: "Avg Turnover", betterDir: 1 },
] as const;

export type SortKey =
  | (typeof METRICS)[number]["key"]
  | (typeof QUANT_METRICS)[number]["key"];

// Best-first sort direction for any metric key, across both metric sets.
export function betterDirOf(key: string): number {
  return (
    (METRICS as readonly { key: string; betterDir: number }[]).find((m) => m.key === key)?.betterDir ??
    (QUANT_METRICS as readonly { key: string; betterDir: number }[]).find((m) => m.key === key)?.betterDir ??
    1
  );
}

export function fmt(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : v.toFixed(4);
}

// Percentage display for ratio-style metrics (returns, drawdown, win rate).
export function fmtPct(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : `${(v * 100).toFixed(2)}%`;
}

// Compact decimal for unit-less ratios (Sharpe, turnover).
export function fmtRatio(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : v.toFixed(2);
}

// Per-column formatter for the quant view.
export const QUANT_FORMAT: Record<string, (v: number | null | undefined) => string> = {
  total_return: fmtPct,
  annualized_return: fmtPct,
  sharpe: fmtRatio,
  max_drawdown: fmtPct,
  win_rate: fmtPct,
  avg_turnover: fmtRatio,
};

// Financial colour semantics. Gain vs loss colours depend on convention:
// Western (redIsUp=false) → green = gain, red = loss; Chinese (redIsUp=true) →
// red = gain (涨), green = loss (跌). Applied to the return/Sharpe columns;
// drawdown reads as a loss; win-rate splits at 50%; turnover stays neutral.
export function quantColor(key: string, v: number | null | undefined, redIsUp = false): string {
  if (v === null || v === undefined) return "text-muted";
  const green = "text-emerald-600 dark:text-emerald-400";
  const red = "text-rose-600 dark:text-rose-400";
  const up = redIsUp ? red : green; // gain
  const down = redIsUp ? green : red; // loss
  switch (key) {
    case "total_return":
    case "annualized_return":
    case "sharpe":
      return v > 0 ? up : v < 0 ? down : "text-muted";
    case "max_drawdown":
      return v < 0 ? down : "text-muted";
    case "win_rate":
      return v >= 0.5 ? up : down;
    default:
      return "text-muted";
  }
}

// Air-quality pollutant labels with proper chemical subscripts.
const POLLUTANT_LABELS: Record<string, string> = {
  pm2_5: "PM2.5",
  pm10: "PM10",
  ozone: "O₃",
  no2: "NO₂",
  so2: "SO₂",
  co: "CO",
};

export function formatPollutant(h: string): string {
  return POLLUTANT_LABELS[h.toLowerCase()] ?? h.toUpperCase().replace("_", ".");
}
