"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useTheme } from "next-themes";
import { Seg } from "./ui/seg";
import { orderDatasets } from "./lib/datasets";
import type { LeaderboardDict } from "./types";
import modelMeta from "@/data/model-meta.json";

// echarts-for-react touches `window` on init, so load it client-only.
const ReactECharts = dynamic(() => import("echarts-for-react"), { ssr: false });

type Meta = Record<string, { year: number | null; venue: string | null; arxiv: string | null }>;
const META = modelMeta as Meta;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Datasets = Record<string, { horizons: Record<string, any[]> }>;

type Point = {
  year: number; // true publication year
  x: number; // logical column (years < 2020 collapse to 2019 → "<2020")
  jx: number; // plotted x with horizontal fan-out so equal-MSE dots don't stack
  mse: number;
  model: string;
  venue: string | null;
  isTop: boolean;
};

const PRE_CUTOFF = 2020;
const xOf = (year: number) => (year < PRE_CUTOFF ? PRE_CUTOFF - 1 : year);
const PICK_COLORS = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"];
const MAX_PICKS = 5;

// Case-insensitive lookup into the TSFLab metadata map.
const metaLookup = (() => {
  const lower: Record<string, keyof Meta> = {};
  for (const k of Object.keys(META)) lower[k.toLowerCase()] = k;
  return (model: string) => META[lower[model.toLowerCase()]];
})();

/**
 * "Method evolution" chart for static datasets: publication year (x) vs MSE (y),
 * one point per model, rendered with ECharts for smooth animated transitions,
 * hover-to-enlarge + name, and a gold best-so-far (SOTA) frontier line.
 */
export function EvolutionChart({ datasets, copy }: { datasets: Datasets; copy: LeaderboardDict }) {
  const names = orderDatasets(Object.keys(datasets));
  const [dataset, setDataset] = useState(names[0]);
  const [yScale, setYScale] = useState<"overview" | "zoom">("overview");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const toggleModel = (name: string) => {
    setSelectedModels((prev) =>
      prev.includes(name) ? prev.filter((m) => m !== name) : prev.length < MAX_PICKS ? [...prev, name] : prev,
    );
  };
  const removeModel = (name: string) => setSelectedModels((prev) => prev.filter((m) => m !== name));
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
  const isZoom = yScale === "zoom";

  const model = useMemo(() => {
    const block = datasets[dataset] ?? datasets[names[0]];
    const h = block ? Object.keys(block.horizons)[0] : undefined;
    const rows = (h && block ? block.horizons[h] : []) ?? [];

    const all: Point[] = rows
      .filter((r) => typeof r.mse === "number")
      .map((r) => {
        const m = metaLookup(r.model);
        return m && typeof m.year === "number"
          ? { year: m.year, x: xOf(m.year), jx: xOf(m.year), mse: r.mse as number, model: r.model, venue: m.venue, isTop: false }
          : null;
      })
      .filter((p): p is Point => p !== null);

    const sortedMse = all.map((p) => p.mse).sort((a, b) => a - b);
    const q = (f: number) => (sortedMse.length ? sortedMse[Math.min(sortedMse.length - 1, Math.floor(f * sortedMse.length))] : 1);
    const cap = sortedMse.length ? Math.max(q(0.9), q(0.5) * 3) : 1;
    const floor = sortedMse.length ? sortedMse[0] : 0.1;
    const zoomCap = sortedMse.length ? q(0.75) * 1.15 : cap;

    // Best-so-far frontier + the methods that set each new SOTA (labelled).
    const xsSorted = Array.from(new Set(all.map((p) => p.x))).sort((a, b) => a - b);
    let best = Infinity;
    const frontier: [number, number][] = [];
    const setters = new Set<string>();
    for (const x of xsSorted) {
      const champ = all.filter((p) => p.x === x).reduce((a, b) => (b.mse < a.mse ? b : a));
      if (champ.mse < best) {
        best = champ.mse;
        setters.add(champ.model);
      }
      frontier.push([x, best]);
    }
    all.forEach((p) => (p.isTop = setters.has(p.model)));

    // Fan points out horizontally within each column (ordered by name to decorrelate from MSE).
    const FAN = 0.72;
    for (const x of xsSorted) {
      const g = all.filter((p) => p.x === x).sort((a, b) => (a.model < b.model ? -1 : 1));
      const n = g.length;
      g.forEach((p, i) => (p.jx = n <= 1 ? x : x - FAN / 2 + (FAN * i) / (n - 1)));
    }

    const hasPre = all.some((p) => p.year < PRE_CUTOFF);
    const xsAll = all.map((p) => p.x);
    const minX = xsAll.length ? Math.min(...xsAll) : PRE_CUTOFF - 1;
    const maxX = xsAll.length ? Math.max(...xsAll) : 2026;
    return { all, frontier, cap, floor, zoomCap, hasPre, minX, maxX };
  }, [datasets, dataset, names]);

  const grid = isDark ? "#2f2f2f" : "#e5e7eb";
  const axis = isDark ? "#9ca3af" : "#6b7280";
  const ink = isDark ? "#e5e7eb" : "#1f2937";
  const gold = isDark ? "#d6b25e" : "#8c6f24";
  const dim = isDark ? "#6b7280" : "#9ca3af";

  const availableModelNames = useMemo(
    () => Array.from(new Set(model.all.map((p) => p.model))).sort(),
    [model],
  );
  const pickerFiltered = availableModelNames.filter(
    (m) => !pickerQuery.trim() || m.toLowerCase().includes(pickerQuery.trim().toLowerCase()),
  );

  const toItem = (p: Point) => ({ value: [p.jx, p.mse], name: p.model, venue: p.venue, year: p.year, mse: p.mse });

  const option = useMemo(() => {
    const { all, frontier, cap, floor, zoomCap, hasPre, minX, maxX } = model;
    const selSet = new Set(selectedModels);
    const rest = all.filter((p) => !p.isTop && !selSet.has(p.model)).map(toItem);
    const top = all.filter((p) => p.isTop && !selSet.has(p.model)).map(toItem);
    const linLo = Math.max(0, floor - (cap - floor) * 0.06);
    const zoomLo = floor * 0.95;

    return {
      animationDuration: 600,
      animationDurationUpdate: 500,
      animationEasing: "cubicOut",
      grid: { left: 52, right: 22, top: 18, bottom: 46 },
      legend: { show: false },
      tooltip: {
        trigger: "item",
        backgroundColor: isDark ? "#1c1c1c" : "#ffffff",
        borderColor: grid,
        borderWidth: 1,
        padding: [7, 11],
        textStyle: { color: ink, fontSize: 13, fontWeight: 500 },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        formatter: (p: any) => {
          const d = p.data || {};
          const venue = d.venue ? `<div style="color:${axis};font-size:11px">${d.venue} · ${d.year}</div>` : `<div style="color:${axis};font-size:11px">${d.year}</div>`;
          return `<b>${d.name}</b>${venue}<div style="color:${gold};font-family:monospace;margin-top:2px">MSE ${Number(d.mse).toFixed(4)}</div>`;
        },
      },
      xAxis: {
        type: "value",
        min: minX - 1,
        max: maxX + 1,
        interval: 1,
        name: copy.overview.xAxis,
        nameLocation: "middle",
        nameGap: 30,
        nameTextStyle: { color: axis, fontSize: 13, fontWeight: 600 },
        axisLine: { lineStyle: { color: grid } },
        axisTick: { show: false },
        splitLine: { show: true, lineStyle: { color: grid, type: "dashed" } },
        axisLabel: {
          color: axis,
          fontSize: 13,
          fontWeight: 500,
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          formatter: (v: any) => {
            const n = Number(v);
            if (n < minX || n > maxX) return "";
            return hasPre && n === PRE_CUTOFF - 1 ? `<${PRE_CUTOFF}` : String(n);
          },
        },
      },
      yAxis: {
        type: "value",
        name: "MSE",
        nameLocation: "end",
        nameGap: 10,
        nameTextStyle: { color: axis, fontSize: 13, fontWeight: 600, align: "left" },
        ...(isZoom
          ? { min: zoomLo, max: zoomCap }
          : { min: linLo, max: cap }),
        axisLine: { lineStyle: { color: grid } },
        axisTick: { show: false },
        splitLine: { show: true, lineStyle: { color: grid, type: "dashed" } },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        axisLabel: { color: axis, fontSize: 13, fontWeight: 500, formatter: (v: any) => Number(v).toFixed(2) },
      },
      series: [
        {
          name: "frontier",
          type: "line",
          step: "end",
          data: frontier,
          showSymbol: false,
          silent: true,
          lineStyle: { color: gold, width: 1.5, type: "dashed" },
          z: 1,
        },
        {
          name: "methods",
          type: "scatter",
          data: rest,
          symbolSize: 9,
          itemStyle: { color: dim, opacity: 0.7 },
          emphasis: {
            scale: 2.2,
            itemStyle: { opacity: 1 },
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            label: { show: true, formatter: (p: any) => p.data.name, position: "top", color: ink, fontSize: 13, fontWeight: 700 },
          },
          z: 2,
        },
        {
          name: "sota",
          type: "scatter",
          data: top,
          symbolSize: 12,
          itemStyle: { color: gold },
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          label: { show: true, formatter: (p: any) => p.data.name, position: "top", color: ink, fontSize: 12, fontWeight: 600 },
          emphasis: { scale: 1.6, label: { show: true, color: ink, fontSize: 13, fontWeight: 700 } },
          z: 3,
        },
        ...selectedModels.flatMap((name, i) => {
          const point = all.find((p) => p.model === name);
          if (!point) return [];
          return [{
            name,
            type: "scatter",
            data: [toItem(point)],
            symbolSize: 14,
            itemStyle: { color: PICK_COLORS[i % PICK_COLORS.length], opacity: 1 },
            label: { show: true, formatter: () => name, position: "top", color: ink, fontSize: 12, fontWeight: 600 },
            emphasis: { scale: 1.5, label: { show: true, color: ink, fontSize: 13, fontWeight: 700 } },
            z: 4,
          }];
        }),
      ],
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, isDark, isZoom, copy, selectedModels]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const onEvents = {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    click: (p: any) => {
      const d = p?.data;
      if (d?.name) toggleModel(d.name);
    },
  };

  return (
    <div className="mb-6 rounded-2xl border border-border bg-surface p-5">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-serif text-lg tracking-[-0.01em] text-ink">{copy.overview.title}</h3>
        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-xs text-muted">{copy.overview.dataset}:</span>
          {names.map((n) => (
            <Seg key={n} active={dataset === n} onClick={() => setDataset(n)}>
              {n}
            </Seg>
          ))}
        </div>
      </div>
      <p className="mb-3 text-xs text-faint">{copy.overview.caption.replace("{dataset}", dataset)}</p>

      <div className="mb-3 flex items-center gap-1.5">
        <span className="mr-1 text-xs text-muted">{copy.overview.scale}:</span>
        <Seg active={!isZoom} onClick={() => setYScale("overview")}>{copy.overview.scaleOverview}</Seg>
        <Seg active={isZoom} onClick={() => setYScale("zoom")}>{copy.overview.scaleZoom}</Seg>
      </div>

      <div className="mb-3 flex min-h-[1.5rem] flex-wrap items-center gap-2 text-xs">
        {selectedModels.length > 0 ? (
          selectedModels.map((name, idx) => {
            const point = model.all.find((p) => p.model === name);
            return (
              <span key={name} className={`inline-flex items-center gap-1.5 rounded-md border border-border bg-paper-2 px-2.5 py-1${point ? "" : " opacity-40"}`}>
                <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: PICK_COLORS[idx % PICK_COLORS.length] }} />
                <span className="font-medium text-ink">{name}</span>
                {point && <span className="font-mono text-accent">MSE {point.mse.toFixed(4)}</span>}
                <button type="button" onClick={() => removeModel(name)} className="ml-0.5 text-muted hover:text-ink">&times;</button>
              </span>
            );
          })
        ) : (
          <span className="text-faint">{copy.overview.clickHint}</span>
        )}
        {selectedModels.length < MAX_PICKS && (
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
                  {pickerFiltered.map((name) => {
                    const isSelected = selectedModels.includes(name);
                    return (
                      <button
                        key={name}
                        type="button"
                        disabled={!isSelected && selectedModels.length >= MAX_PICKS}
                        onClick={() => toggleModel(name)}
                        className={`w-full rounded px-2.5 py-1.5 text-left text-xs ${
                          isSelected
                            ? "bg-accent/10 font-medium text-accent"
                            : "text-muted hover:bg-paper-2 hover:text-ink disabled:opacity-40"
                        }`}
                      >
                        {name}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
        <span className="ml-auto inline-flex items-center gap-1.5 text-faint">
          <span className="inline-block h-0.5 w-5" style={{ backgroundColor: gold }} />
          {copy.overview.frontier}
        </span>
      </div>

      <ReactECharts
        key={isZoom ? "zoom" : "overview"}
        option={option}
        onEvents={onEvents}
        style={{ height: 360, width: "100%" }}
        opts={{ renderer: "canvas" }}
        notMerge
        lazyUpdate
      />
    </div>
  );
}
