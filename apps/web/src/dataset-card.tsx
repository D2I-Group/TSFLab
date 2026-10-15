"use client";

import { useEffect, useMemo, useState } from "react";
import type { LeaderboardDict } from "./types";
import type { DatasetBlock } from "./lib/leaderboard-data";
import { getModelType, STOCK_VIZ_MODELS } from "./lib/model-types";
import { formatPollutant, betterDirOf, type SortKey } from "./lib/metrics";
import { Seg } from "./ui/seg";
import { ResultsTable } from "./results-table";
import { QuantVisualization, TradingStrategyDescription } from "./quant-visualization";

export function DatasetCard({
  track,
  name,
  ds,
  copy,
  query,
  hideTitle = false,
  view: viewProp,
  quantConfig: quantConfigProp,
  onQuantConfigChange,
  locale = "en",
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  visualizationData,
}: {
  track: string;
  name: string;
  ds: DatasetBlock;
  copy: LeaderboardDict;
  query: string;
  hideTitle?: boolean;
  view?: "regression" | "quant";
  quantConfig?: "conservative" | "balanced" | "aggressive";
  onQuantConfigChange?: (cfg: "conservative" | "balanced" | "aggressive") => void;
  locale?: "en" | "zh";
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  visualizationData?: any;
}) {
  const horizons = useMemo(
    () => Object.keys(ds.horizons).sort((a, b) => Number(a) - Number(b)),
    [ds],
  );
  const [horizon, setHorizon] = useState(horizons[0]);

  const view = viewProp ?? "regression";
  const hasQuantData = track === "stock" && ds.quant && Object.keys(ds.quant).length > 0;

  const defaultSortKey = (view === "quant" ? "total_return" : "mse") as SortKey;
  const [sortKey, setSortKey] = useState<SortKey>(defaultSortKey);
  const [sortDir, setSortDir] = useState(() => betterDirOf(defaultSortKey));

  useEffect(() => {
    const k = (view === "quant" ? "total_return" : "mse") as SortKey;
    setSortKey(k);
    setSortDir(betterDirOf(k));
  }, [view]);

  const quantConfig = quantConfigProp ?? "conservative";
  const setQuantConfig = onQuantConfigChange ?? (() => {});

  // Model type filter — at least one must stay selected.
  const [showTimeSeries, setShowTimeSeries] = useState(true);
  const [showSpatiotemporal, setShowSpatiotemporal] = useState(true);
  const [showAirQuality, setShowAirQuality] = useState(true);
  const toggleTimeSeries = () => {
    if (!showTimeSeries || showSpatiotemporal || showAirQuality) setShowTimeSeries(!showTimeSeries);
  };
  const toggleSpatiotemporal = () => {
    if (!showSpatiotemporal || showTimeSeries || showAirQuality) setShowSpatiotemporal(!showSpatiotemporal);
  };
  const toggleAirQuality = () => {
    if (!showAirQuality || showTimeSeries || showSpatiotemporal) setShowAirQuality(!showAirQuality);
  };

  function toggleSort(k: SortKey) {
    if (k === sortKey) setSortDir((d) => -d);
    else {
      setSortKey(k);
      setSortDir(betterDirOf(k)); // start "best-first" for the chosen metric
    }
  }

  // Rows for the selected horizon + view.
  const all = (() => {
    if (view === "quant" && ds.quant) {
      const key = `backtest_91d_${quantConfig}` as keyof typeof ds.quant;
      return ds.quant[key] || ds.horizons[horizon];
    }
    return ds.horizons[horizon] || [];
  })();

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = q ? all.filter((r) => r.model.toLowerCase().includes(q)) : all;
    const withTypes = filtered.map((r) => ({ ...r, modelType: getModelType(r.model) }));

    const allSorted = withTypes.slice().sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      return (av - bv) * sortDir;
    });

    const isNaturalOrder = sortDir === betterDirOf(sortKey);

    // Ranks for non-baseline models only (baseline shows "—" and doesn't shift others).
    let rankCounter = 0;
    const withRanks = allSorted.map((row) => {
      if (row.modelType === "baseline") return { ...row, displayRank: undefined, categoryRank: undefined };
      rankCounter++;
      return { ...row, displayRank: isNaturalOrder ? rankCounter : 135 - rankCounter + 1 };
    });

    const typeFiltered = withRanks.filter((r) => {
      if (r.modelType === "baseline") return true;
      if (r.modelType === "st" && !showSpatiotemporal) return false;
      if (r.modelType === "ts" && !showTimeSeries) return false;
      if (r.modelType === "aq" && !showAirQuality) return false;
      return true;
    });

    return typeFiltered;
  }, [all, query, sortKey, sortDir, showTimeSeries, showSpatiotemporal, showAirQuality, track]);

  // Baselines are shown but excluded from the model count.
  const nBaselines = all.filter((r) => getModelType(r.model) === "baseline").length;
  const showCount = copy.showing
    .replace("{n}", String(rows.filter((r) => r.modelType !== "baseline").length))
    .replace("{total}", String(all.length - nBaselines));

  const chip = (text: string) => (
    <span className="rounded-md border border-border bg-paper-2 px-2.5 py-1 text-xs text-muted">{text}</span>
  );

  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-surface">
      {/* Header: title + filters (left) · metadata chips (right) */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-3.5">
        <div className="flex flex-wrap items-center gap-3">
          {!hideTitle && <span className="font-serif text-lg tracking-[-0.01em] text-ink">{name}</span>}

          {(track === "stock" || track === "air_quality") && (
            <div className="flex items-center gap-1.5">
              <span className="mr-1 text-xs text-muted">{copy.quant.modelType}:</span>
              <Seg active={showTimeSeries} disabled={showTimeSeries && !showSpatiotemporal && !showAirQuality} onClick={toggleTimeSeries}>
                {copy.tracks.time_series}
              </Seg>
              <Seg active={showSpatiotemporal} disabled={showSpatiotemporal && !showTimeSeries && !showAirQuality} onClick={toggleSpatiotemporal}>
                {copy.tracks.spatiotemporal}
              </Seg>
              {track === "air_quality" && (
                <Seg active={showAirQuality} disabled={showAirQuality && !showTimeSeries && !showSpatiotemporal} onClick={toggleAirQuality}>
                  {copy.tracks.air_quality}
                </Seg>
              )}
            </div>
          )}

          {track === "stock" && view === "quant" && hasQuantData && (
            <div className="flex items-center gap-1.5">
              <span className="mr-1 text-xs text-muted">{copy.quant.strategy}:</span>
              {(["conservative", "balanced", "aggressive"] as const).map((cfg) => (
                <Seg key={cfg} active={quantConfig === cfg} onClick={() => setQuantConfig(cfg)}>
                  {copy.quant.configs[cfg]}
                </Seg>
              ))}
            </div>
          )}

          <span className="text-xs text-faint [font-variant-numeric:tabular-nums]">{showCount}</span>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {track === "stock" && chip("2026/01/01 ~ 05/31")}
          {track === "air_quality" && chip("2026/01 ~ 05/31")}
          {track === "time_series" ? (
            <>
              {chip("seq_len: 336")}
              {chip("pred_len: 192")}
            </>
          ) : track === "air_quality" ? (
            <>
              {chip("seq_len: 24")}
              {chip("pred_len: 24")}
            </>
          ) : (
            <>
              {chip("seq_len: 20")}
              {view !== "quant" && chip(`pred_len: ${horizons[0]}`)}
            </>
          )}
          {/* Horizon selector — only when there's more than one (single horizon
              already shows as the pred_len chip). Air quality uses its own row.
              Hidden in quant view where horizons don't apply. */}
          {track !== "air_quality" &&
            view !== "quant" &&
            horizons.length > 1 &&
            horizons.map((h) => (
              <Seg key={h} active={h === horizon} onClick={() => setHorizon(h)}>
                {`${copy.horizon} ${h}`}
              </Seg>
            ))}
        </div>
      </div>

      {/* Pollutant selector — air quality only, on its own labelled row */}
      {track === "air_quality" && horizons.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 border-b border-border bg-paper-2/40 px-5 py-2.5">
          <span className="mr-1 text-[0.7rem] font-medium uppercase tracking-[0.04em] text-faint">{copy.quant.pollutant}</span>
          {horizons.map((h) => (
            <Seg key={h} active={h === horizon} onClick={() => setHorizon(h)}>
              {formatPollutant(h)}
            </Seg>
          ))}
        </div>
      )}

      {/* Trends chart above the table (stock track only) */}
      {track === "stock" && visualizationData && (
        <QuantVisualization data={visualizationData} availableModels={STOCK_VIZ_MODELS} view={view} copy={copy} quantConfig={quantConfig} />
      )}

      <ResultsTable
        rows={rows}
        track={track}
        view={view}
        sortKey={sortKey}
        sortDir={sortDir}
        onSort={toggleSort}
        copy={copy}
        redIsUp={locale === "zh"}
      />

      {track === "stock" && view === "quant" && <TradingStrategyDescription copy={copy} />}
    </div>
  );
}
