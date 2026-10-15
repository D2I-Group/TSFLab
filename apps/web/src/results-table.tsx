"use client";

import type { LeaderboardDict } from "./types";
import type { LeaderRow } from "./lib/leaderboard-data";
import { METRICS, QUANT_METRICS, QUANT_FORMAT, quantColor, fmt, type SortKey } from "./lib/metrics";
import { RankBadge, Th, SortTh } from "./ui/table-bits";

type Props = {
  rows: LeaderRow[];
  track: string;
  view: "regression" | "quant";
  sortKey: SortKey;
  sortDir: number;
  onSort: (k: SortKey) => void;
  copy: LeaderboardDict;
  // Chinese convention flips gain/loss colours (red = up, green = down).
  redIsUp?: boolean;
};

export function ResultsTable({ rows, track, view, sortKey, sortDir, onSort, copy, redIsUp = false }: Props) {
  if (rows.length === 0) {
    return <p className="px-5 py-12 text-center text-sm text-muted">{copy.noMatch}</p>;
  }

  const isQuant = track === "stock" && view === "quant";
  const showTsCols = track === "time_series";

  // Bar width = min–max normalised magnitude of the active sort metric across
  // the visible rows (handles negative quant values gracefully).
  const sortVals = rows
    .map((r) => r[sortKey])
    .filter((v): v is number => typeof v === "number");
  const minV = sortVals.length ? Math.min(...sortVals) : 0;
  const maxV = sortVals.length ? Math.max(...sortVals) : 1;
  const barWidth = (v: number | null | undefined) =>
    typeof v === "number" && maxV > minV ? `${(((v - minV) / (maxV - minV)) * 100).toFixed(0)}%` : "0%";

  return (
    <div className="max-h-[34rem] overflow-auto">
      <table className="w-full border-collapse [font-variant-numeric:tabular-nums]">
        <thead className="sticky top-0 z-10 bg-surface">
          <tr>
            <Th className="w-12" title={copy.quant.rankTip}>
              #
            </Th>
            <Th>{copy.cols.model}</Th>

            {isQuant ? (
              QUANT_METRICS.map((m) => (
                <SortTh
                  key={m.key}
                  label={m.label}
                  active={sortKey === m.key}
                  dir={sortDir}
                  onClick={() => onSort(m.key)}
                />
              ))
            ) : (
              <>
                <SortTh label={copy.cols.mse} active={sortKey === "mse"} dir={sortDir} onClick={() => onSort("mse")} />
                <SortTh label={copy.cols.mae} active={sortKey === "mae"} dir={sortDir} onClick={() => onSort("mae")} />
                {showTsCols && (
                  <>
                    <SortTh label="WAPE" active={sortKey === "wape"} dir={sortDir} onClick={() => onSort("wape")} />
                    <SortTh label="RSE" active={sortKey === "rse"} dir={sortDir} onClick={() => onSort("rse")} />
                  </>
                )}
                <SortTh label={copy.cols.corr} active={sortKey === "corr"} dir={sortDir} onClick={() => onSort("corr")} />
              </>
            )}
            <Th>{copy.cols.runs}</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const isBaseline = r.modelType === "baseline";
            const displayName = isBaseline
              ? isQuant
                ? copy.quant.buyAndHold
                : copy.quant.lastValueCopy
              : r.model;

            const dataCell = (key: SortKey, value: string, colorClass: string) => (
              <td
                key={key}
                className={`relative px-5 py-2.5 font-mono text-sm ${colorClass} ${sortKey === key ? "font-semibold" : ""}`}
              >
                {value}
                {sortKey === key && (
                  <span
                    aria-hidden="true"
                    className="absolute inset-x-0 bottom-0 h-px bg-accent/40"
                    style={{ width: barWidth(r[key]) }}
                  />
                )}
              </td>
            );

            return (
              <tr
                key={r.model}
                className={`border-b border-border last:border-0 transition-colors ${
                  isBaseline ? "bg-accent/10 hover:bg-accent/15" : "hover:bg-paper-2"
                }`}
              >
                <td className="px-5 py-2.5">
                  {r.displayRank ? <RankBadge rank={r.displayRank} /> : <span className="text-sm text-muted">—</span>}
                </td>
                <td className="px-5 py-2.5 text-sm font-semibold">
                  <span className={isBaseline ? "text-accent" : "text-ink"}>{displayName}</span>
                  {isBaseline && (
                    <span className="ml-2 rounded border border-accent/40 bg-accent-soft px-1.5 py-0.5 text-[0.6rem] font-medium uppercase tracking-wide text-accent">
                      {copy.quant.baselineTag}
                    </span>
                  )}
                </td>

                {isQuant
                  ? QUANT_METRICS.map((m) => dataCell(m.key, QUANT_FORMAT[m.key](r[m.key]), quantColor(m.key, r[m.key], redIsUp)))
                  : [
                      dataCell("mse", fmt(r.mse), sortKey === "mse" ? "text-accent" : "text-muted"),
                      dataCell("mae", fmt(r.mae), sortKey === "mae" ? "text-accent" : "text-muted"),
                      ...(showTsCols
                        ? [
                            dataCell("wape", fmt(r.wape), sortKey === "wape" ? "text-accent" : "text-muted"),
                            dataCell("rse", fmt(r.rse), sortKey === "rse" ? "text-accent" : "text-muted"),
                          ]
                        : []),
                      dataCell("corr", fmt(r.corr), sortKey === "corr" ? "text-accent" : "text-muted"),
                    ]}
                <td className="px-5 py-2.5 text-sm text-muted">{isBaseline ? "—" : r.n_runs}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// METRICS is re-exported for the card's sort logic convenience.
export { METRICS };
