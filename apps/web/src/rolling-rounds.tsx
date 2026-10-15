"use client";

import type { RealtimeTrackMeta, RollingSummary } from "./lib/leaderboard-data";
import { RankBadge, Th } from "./ui/table-bits";

export interface RollingCopy {
  title: string;
  intro: string;
  rounds: string;
  stability: string;
  meanRank: string;
  meanMse: string;
  empty: string;
  submit: string;
  awaiting?: string;
}

const fmt = (v: number | null | undefined, digits = 4) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);

// Domain ids come from the track id prefix; a few differ from the dictionary keys.
const DOMAIN_KEY: Record<string, string> = { air: "air_quality" };

const titleCase = (s: string) => s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

interface Entry { meta: RealtimeTrackMeta; summary?: RollingSummary }

/** Weekly rounds: forecasts are submitted before their truth exists, then scored.
 *  Tracks come from `tracks` (generated from configs/realtime) unioned with the
 *  summaries in `data`, grouped by domain; tracks without rounds are "awaiting". */
export function RollingRounds({ data, tracks, copy, domainLabels = {} }: {
  data: Record<string, RollingSummary>;
  tracks?: RealtimeTrackMeta[];
  copy: RollingCopy;
  domainLabels?: Record<string, string>;
}) {
  const byId = new Map<string, Entry>();
  for (const meta of tracks ?? []) byId.set(meta.id, { meta });
  for (const [id, summary] of Object.entries(data)) {
    const tid = summary.track ?? id;
    const prev = byId.get(tid);
    byId.set(tid, { meta: prev?.meta ?? { id: tid, domain: tid.split("_")[0] }, summary });
  }
  const groups = new Map<string, Entry[]>();
  for (const e of byId.values()) {
    const list = groups.get(e.meta.domain) ?? [];
    list.push(e);
    groups.set(e.meta.domain, list);
  }
  const domains = [...groups.keys()].sort(
    (a, b) => Number(groups.get(b)!.some((e) => e.summary?.methods.length)) -
              Number(groups.get(a)!.some((e) => e.summary?.methods.length)) || a.localeCompare(b),
  );
  const awaiting = copy.awaiting ?? "Awaiting first release";
  return (
    <section className="mt-8 rounded-xl border border-border bg-surface p-5">
      <h2 className="text-base font-semibold text-ink">{copy.title}</h2>
      <p className="mt-1 max-w-3xl text-sm text-muted">
        {copy.intro}{" "}
        <a className="text-accent underline-offset-2 hover:underline"
           href="https://github.com/D2I-Group/TSFLab/blob/main/apps/web/SUBMITTING.md#real-time-rounds">
          {copy.submit}
        </a>
      </p>
      {byId.size === 0 && <p className="mt-4 text-sm text-faint">{copy.empty}</p>}
      {domains.map((domain) => (
        <div key={domain} className="mt-6">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-muted">
            {domainLabels[DOMAIN_KEY[domain] ?? domain] ?? titleCase(domain)}
          </h3>
          {groups.get(domain)!.map(({ meta, summary }) => (
            <TrackBlock key={meta.id} meta={meta} summary={summary} copy={copy} awaiting={awaiting} />
          ))}
        </div>
      ))}
    </section>
  );
}

function TrackBlock({ meta, summary, copy, awaiting }: {
  meta: RealtimeTrackMeta; summary?: RollingSummary; copy: RollingCopy; awaiting: string;
}) {
  const rounds = summary?.scored_rounds.length ?? 0;
  const detail = [meta.freq && `freq ${meta.freq}`, meta.horizon && `horizon ${meta.horizon}`].filter(Boolean).join(" · ");
  return (
    <div className="mt-4">
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-sm">
        <span className="font-medium text-ink" title={meta.title}>{meta.id}</span>
        {meta.title && <span className="text-faint">{meta.title}</span>}
        {detail && <span className="text-faint">{detail}</span>}
        {summary && rounds > 0 ? (
          <>
            <span className="text-muted">{copy.rounds}: {rounds}</span>
            <span className="text-muted">{copy.stability}: {fmt(summary.rank_stability.mean, 2)}</span>
          </>
        ) : (
          <span className="rounded-full border border-border px-2 py-0.5 text-xs text-faint">{awaiting}</span>
        )}
      </div>
      {summary && summary.methods.length > 0 && (
        <div className="mt-2 overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                <Th>#</Th>
                <Th>Model</Th>
                <Th>{copy.rounds}</Th>
                <Th>{copy.meanRank}</Th>
                <Th>{copy.meanMse}</Th>
              </tr>
            </thead>
            <tbody>
              {summary.methods.map((m, i) => (
                <tr key={m.model} className="border-b border-border/60">
                  <td className="px-5 py-2"><RankBadge rank={i + 1} /></td>
                  <td className="px-5 py-2 text-ink">{m.model}</td>
                  <td className="px-5 py-2 text-muted [font-variant-numeric:tabular-nums]">{m.rounds}</td>
                  <td className="px-5 py-2 text-muted [font-variant-numeric:tabular-nums]">{fmt(m.mean_rank, 2)}</td>
                  <td className="px-5 py-2 text-muted [font-variant-numeric:tabular-nums]">{fmt(m.mean_mse)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
