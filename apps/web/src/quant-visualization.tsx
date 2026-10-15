"use client";

import { useEffect, useRef, useState, useMemo } from "react";
import { useTheme } from "next-themes";
import type { LeaderboardDict } from "./types";
import {
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  ReferenceLine,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

type VisualizationData = {
  log_return: {
    dates: string[];
    true_avg: number[];
    models: Record<string, number[]>;
  };
  cumulative_return: {
    dates: string[];
    baseline: Record<string, number[]>;
    models: Record<string, number[]>;
  };
};

type Props = {
  data: VisualizationData;
  availableModels: string[];
  view?: "regression" | "quant";
  copy: LeaderboardDict;
  quantConfig?: "conservative" | "balanced" | "aggressive";
};

const COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"];

// Resolve a model's series in the viz data, tolerating st_/ts_ prefixes.
function resolveModel<T>(models: Record<string, T>, model: string, suffix = ""): T | null {
  const keys = [`${model}${suffix}`, `st_${model}${suffix}`, `ts_${model}${suffix}`];
  for (const k of keys) if (models[k]) return models[k];
  return null;
}

export function QuantVisualization({ data, availableModels, view = "quant", copy, quantConfig: quantConfigProp }: Props) {
  const bhLabel = copy.viz.buyHold;
  const mode: "cumulative" | "scatter" = view === "regression" ? "scatter" : "cumulative";
  const config = quantConfigProp ?? "conservative";

  const [selectedModels, setSelectedModels] = useState<string[]>(
    [availableModels[0], availableModels[1]].filter(Boolean),
  );
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pickerQuery, setPickerQuery] = useState("");
  const pickerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!pickerOpen) return;
    const handler = (e: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) setPickerOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [pickerOpen]);

  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";
  const gridStroke = isDark ? "#2f2f2f" : "#e5e7eb";
  const axisColor = isDark ? "#9ca3af" : "#6b7280";
  const baselineStroke = isDark ? "#d1d5db" : "#111111";

  const toggleModel = (model: string) => {
    setSelectedModels((prev) => {
      if (prev.includes(model)) return prev.filter((m) => m !== model);
      if (prev.length < 5) return [...prev, model];
      return prev;
    });
  };

  const removeModel = (model: string) => {
    setSelectedModels((prev) => prev.filter((m) => m !== model));
  };

  // ---- Cumulative-return (Quant view) line data ----
  const cumulativeData = useMemo(() => {
    return data.cumulative_return.dates.map((date, idx) => {
      const point: Record<string, number | string> = { date };
      point[bhLabel] = data.cumulative_return.baseline[config][idx];
      selectedModels.forEach((model) => {
        const series = resolveModel(data.cumulative_return.models, model, `_${config}`);
        if (series) point[model] = series[idx];
      });
      return point;
    });
  }, [selectedModels, config, data, bhLabel]);

  // ---- Predicted-vs-actual (Regression view) scatter data ----
  const scatterSeries = useMemo(() => {
    return selectedModels.map((model) => {
      const series = resolveModel(data.log_return.models, model);
      const points = series
        ? data.log_return.dates.map((date, idx) => ({
            date,
            actual: data.log_return.true_avg[idx],
            predicted: series[idx],
          }))
        : [];
      return { model, points };
    });
  }, [selectedModels, data]);

  // Symmetric square domain so the 45° "perfect prediction" line reads true.
  const scatterDomain = useMemo(() => {
    let m = 0;
    scatterSeries.forEach((s) =>
      s.points.forEach((p) => {
        m = Math.max(m, Math.abs(p.actual), Math.abs(p.predicted));
      }),
    );
    m = m > 0 ? m * 1.05 : 0.05;
    return [-m, m] as [number, number];
  }, [scatterSeries]);

  const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;
  const fmtCum = (v: number, digits = 1) => `${Number(v).toFixed(digits)}%`;

  const pickerFiltered = availableModels.filter(
    (m) => !pickerQuery.trim() || m.toLowerCase().includes(pickerQuery.trim().toLowerCase()),
  );

  return (
    <div className="border-b border-border px-5 py-5">
      {/* Title row + compact model chips */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h3 className="mr-2 font-serif text-base tracking-[-0.01em] text-ink">
          {mode === "cumulative" ? copy.viz.cumulativeTitle : copy.viz.scatterTitle}
        </h3>
        {selectedModels.map((model, idx) => (
          <span key={model} className="inline-flex items-center gap-1 rounded-md border border-border bg-paper-2 px-2 py-0.5 text-xs font-medium text-ink">
            <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
            {model}
            <button type="button" onClick={() => removeModel(model)} className="ml-0.5 text-muted hover:text-ink">&times;</button>
          </span>
        ))}
        {selectedModels.length < 5 && (
          <div className="relative" ref={pickerRef}>
            <button
              type="button"
              onClick={() => { setPickerOpen(!pickerOpen); setPickerQuery(""); }}
              className="rounded-md border border-dashed border-border px-2 py-0.5 text-xs font-medium text-accent hover:border-accent/50"
            >
              {copy.viz.addModel}
            </button>
            {pickerOpen && (
              <div className="absolute left-0 top-full z-20 mt-1 w-56 rounded-lg border border-border bg-surface shadow-lg">
                <div className="border-b border-border p-2">
                  <input
                    type="search"
                    value={pickerQuery}
                    onChange={(e) => setPickerQuery(e.target.value)}
                    placeholder={copy.viz.searchModels}
                    autoFocus
                    className="w-full rounded-md border border-border bg-paper-2 px-2.5 py-1.5 text-xs text-ink placeholder:text-faint focus:border-accent focus:outline-none"
                  />
                </div>
                <div className="max-h-48 overflow-y-auto p-1">
                  {pickerFiltered.map((model) => {
                    const isSelected = selectedModels.includes(model);
                    return (
                      <button
                        key={model}
                        type="button"
                        disabled={!isSelected && selectedModels.length >= 5}
                        onClick={() => toggleModel(model)}
                        className={`w-full rounded px-2.5 py-1.5 text-left text-xs ${
                          isSelected
                            ? "bg-accent/10 font-medium text-accent"
                            : "text-muted hover:bg-paper-2 hover:text-ink disabled:opacity-40"
                        }`}
                      >
                        {model}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
        <span className="text-xs text-faint">{copy.viz.maxSelected.replace("{n}", String(selectedModels.length))}</span>
      </div>

      {/* Chart */}
      <div className="h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          {mode === "cumulative" ? (
            <LineChart data={cumulativeData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 12, fill: axisColor }}
                stroke={gridStroke}
                tickFormatter={(val) => String(val).slice(5)}
              />
              <YAxis
                tick={{ fontSize: 12, fill: axisColor }}
                stroke={gridStroke}
                tickFormatter={(val) => fmtCum(val)}
              />
              <Tooltip
                contentStyle={
                  isDark ? { backgroundColor: "#1c1c1c", border: `1px solid ${gridStroke}`, color: "#e5e7eb" } : undefined
                }
                formatter={(value) => fmtCum(Number(value), 2)}
                labelFormatter={(label) => `${label}`}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey={bhLabel}
                stroke={baselineStroke}
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
              />
              {selectedModels.map((model, idx) => (
                <Line
                  key={model}
                  type="monotone"
                  dataKey={model}
                  stroke={COLORS[idx % COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          ) : (
            <ScatterChart margin={{ top: 10, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis
                type="number"
                dataKey="actual"
                domain={scatterDomain}
                tick={{ fontSize: 12, fill: axisColor }}
                stroke={gridStroke}
                tickFormatter={(val) => pct(val)}
                label={{ value: copy.viz.actual, position: "insideBottom", offset: -10, fontSize: 12, fill: axisColor }}
              />
              <YAxis
                type="number"
                dataKey="predicted"
                domain={scatterDomain}
                tick={{ fontSize: 12, fill: axisColor }}
                stroke={gridStroke}
                tickFormatter={(val) => pct(val)}
                label={{ value: copy.viz.predicted, angle: -90, position: "insideLeft", fontSize: 12, fill: axisColor }}
              />
              <ReferenceLine
                stroke={baselineStroke}
                strokeDasharray="5 5"
                segment={[
                  { x: scatterDomain[0], y: scatterDomain[0] },
                  { x: scatterDomain[1], y: scatterDomain[1] },
                ]}
                ifOverflow="extendDomain"
              />
              <Tooltip
                cursor={{ stroke: gridStroke }}
                contentStyle={
                  isDark ? { backgroundColor: "#1c1c1c", border: `1px solid ${gridStroke}`, color: "#e5e7eb" } : undefined
                }
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                content={({ payload }: any) => {
                  const p = payload?.[0]?.payload;
                  if (!p) return null;
                  return (
                    <div className="rounded-md border border-border bg-paper px-3 py-2 text-xs shadow-sm">
                      <div className="font-medium text-ink">{p.date}</div>
                      <div className="text-muted">{copy.viz.actual}: {pct(p.actual, 2)}</div>
                      <div className="text-muted">{copy.viz.predicted}: {pct(p.predicted, 2)}</div>
                    </div>
                  );
                }}
              />
              <Legend verticalAlign="top" align="center" wrapperStyle={{ paddingBottom: 8 }} />
              {scatterSeries.map((s, idx) => (
                <Scatter
                  key={s.model}
                  name={s.model}
                  data={s.points}
                  fill={COLORS[idx % COLORS.length]}
                  fillOpacity={0.6}
                  isAnimationActive={false}
                />
              ))}
            </ScatterChart>
          )}
        </ResponsiveContainer>
      </div>
      <p className="mt-2 text-center text-xs text-faint">
        {mode === "cumulative" ? copy.viz.cumulativeFootnote : copy.viz.scatterFootnote}
        {mode === "scatter" && <> · — — — {copy.viz.perfectLine}</>}
      </p>
    </div>
  );
}

export function TradingStrategyDescription({ copy }: { copy: LeaderboardDict }) {
  const s = copy.strategy;
  return (
    <details className="border-t border-border">
      <summary className="cursor-pointer select-none px-5 py-3.5 text-sm font-medium text-muted hover:text-ink">
        {s.heading}
      </summary>
      <div className="space-y-4 border-t border-border px-5 py-5 text-sm leading-relaxed text-muted">
        <p>{s.intro}</p>

        <div className="rounded-md bg-paper-2 p-4">
          <h4 className="mb-2 font-medium text-ink">{s.coreTitle}</h4>
          <ul className="space-y-1.5 text-sm">
            {s.core.map((item) => (
              <li key={item.label}>
                • <strong>{item.label}</strong>：{item.value}
              </li>
            ))}
          </ul>
        </div>

        <div className="space-y-1">
          <p className="font-medium text-ink">{s.scoringLabel}</p>
          <code className="block rounded bg-paper-2 px-3 py-2 text-xs font-mono">
            {s.scoringFormula}
          </code>
        </div>

        <div className="space-y-3">
          {s.items.map((item) => (
            <div key={item.title} className="border-l-2 border-accent/30 pl-4">
              <h4 className="mb-1 font-medium text-ink">{item.title}</h4>
              <p>{item.body}</p>
            </div>
          ))}
        </div>

        <div className="mt-4 space-y-2 border-t border-border pt-4">
          <div className="text-xs text-muted">
            <p>{s.backtestPeriod}</p>
            <p>{s.marketCondition}</p>
            <p>{s.initialCapital}</p>
          </div>
          <p className="mt-3 text-xs italic text-faint">
            ⚠️ <strong>{s.disclaimerLabel}</strong>：{s.disclaimer}
          </p>
        </div>
      </div>
    </details>
  );
}
