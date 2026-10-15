// Shape of data/leaderboard.json (produced by the TSFLab Leaderboard build pipeline).
import type { ModelType } from "./model-types";

export interface LeaderRow {
  model: string;
  mse: number | null;
  mae: number | null;
  rmse?: number | null;
  wape?: number | null;
  rse?: number | null;
  corr: number | null;
  n_runs: number;
  submission_ids: string[];
  rank?: number;
  displayRank?: number;
  categoryRank?: number;
  modelType?: ModelType;
  // Quant (trading) metrics — present on stock-track quant rows.
  total_return?: number | null;
  annualized_return?: number | null;
  sharpe?: number | null;
  max_drawdown?: number | null;
  win_rate?: number | null;
  avg_turnover?: number | null;
}

export interface DatasetBlock {
  horizons: Record<string, LeaderRow[]>;
  // Stock track only: backtest_91d_{conservative|balanced|aggressive} → rows.
  quant?: Record<string, LeaderRow[]>;
}

export interface TrackBlock {
  datasets: Record<string, DatasetBlock>;
}

export interface RollingMethod {
  model: string;
  rounds: number;
  mean_mse: number;
  mean_rank: number;
}

export interface RollingSummary {
  track: string;
  scored_rounds: string[];
  methods: RollingMethod[];
  rank_stability: { consecutive_kendall_tau: number[]; mean: number | null };
}

/** Metadata of a real-time track, generated from configs/realtime/*.toml. */
export interface RealtimeTrackMeta {
  id: string;
  domain: string;
  title?: string;
  mode?: string;
  freq?: string;
  seq_len?: number;
  horizon?: number;
  submission_hours?: number;
}

export interface LeaderboardData {
  schema_version: string;
  generated_at: string;
  primary_metric: string;
  n_submissions: number;
  n_rejected: number;
  tracks: Record<string, TrackBlock>;
  /** Rolling real-time rounds, keyed by real-time track id (schema >= 1.1). */
  realtime?: Record<string, RollingSummary>;
  /** Every declared real-time track (schema >= 1.2); tracks without rounds still appear. */
  realtime_tracks?: RealtimeTrackMeta[];
}
