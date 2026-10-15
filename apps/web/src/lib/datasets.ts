// Canonical display order for static (time-series) datasets, independent of the
// key order in leaderboard.json. ETT variants lead as H1, M1, H2, M2; the rest
// follow. Unlisted datasets sort after, alphabetically.
const DATASET_ORDER = [
  "ETTh1",
  "ETTm1",
  "ETTh2",
  "ETTm2",
  "electricity",
  "solar",
  "traffic",
  "weather",
];

export function datasetRank(name: string): number {
  const i = DATASET_ORDER.indexOf(name);
  return i === -1 ? DATASET_ORDER.length + 1 : i;
}

export function orderDatasets(names: string[]): string[] {
  return names.slice().sort((a, b) => datasetRank(a) - datasetRank(b) || a.localeCompare(b));
}
