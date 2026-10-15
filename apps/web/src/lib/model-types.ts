// Model-type classification, derived from the TSFLab work_dirs structure.
// Used to tag each leaderboard row (ts / st / aq / baseline) for filtering and
// the per-category rank column.

export const SPATIOTEMPORAL_MODELS = new Set([
  "AGCRN", "BiST", "BigST", "D2STGNN", "DCRNN", "DFDGCN", "DGCRN", "DSTAGNN",
  "GTS", "GWNet", "HimNet", "LSTM", "MAGE", "MTGNN", "MegaCRN",
  "RPMixer", "STAEformer", "STDN", "STGCN", "STGODE", "STID", "STNorm",
  "STOP", "STPGNN", "STTN", "STWave", "StemGNN", "Sumba",
]);

// Air-quality-specific models (covariate mode).
export const AIR_QUALITY_MODELS = new Set([
  "AirCade", "AirFormer", "CauAir", "DeepAir", "PM25GNN",
]);

// Baseline models — always shown regardless of the type filter.
export const BASELINE_MODELS = ["baseline", "HL"];

export const TIME_SERIES_MODELS = new Set([
  "AMRC", "APN", "Amplifier", "Aurora", "Autoformer", "BiMamba", "CARD", "CATS",
  "CMoS", "COSA", "CoRA", "CrossGNN", "CrossLinear", "Crossformer", "CycleNet",
  "DLinear", "DSFormer", "DTAF", "DUET", "DeepAR", "DistDF", "DynamicTMoE",
  "ETSformer", "FEDformer", "FITS", "FTP", "FeTS", "FiLM", "FreTS", "Fredformer",
  "GOTSF", "GTR", "HDMixer", "HMformer", "HN_MVTS", "ImplicitForecaster",
  "Informer", "InterPDN", "Koopa", "Kronos", "LatentTSF", "LightTS", "Linear",
  "MAFS", "MICN", "MMPD", "MSGNet", "MTSMixer", "MambaSimple", "MixLinear",
  "MoFo", "ModernTCN", "MultiPatchFormer", "NBeats", "NHiTS", "NLinear",
  "NSTransformer", "OLinear", "OccamVTS", "PAttn", "PHAT", "PMDformer", "PULSE",
  "PWS", "PaiFilter", "PatchMLP", "PatchTST", "Pathformer", "PhaseFormer",
  "Pyraformer", "RLinear", "Reformer", "S4", "SCINet", "SEMPO", "SOFTS", "SRSNet",
  "SVTime", "S_Mamba", "SegRNN", "Sonnet", "SparseTSF", "Sumba", "SymTime",
  "TSMixer", "TSRAG", "TexFilter", "TiDE", "TiRex", "TimeAlign", "TimeBase",
  "TimeBridge", "TimeCAP", "TimeEmb", "TimeFilter", "TimeKAN", "TimeMixer",
  "TimeMosaic", "TimeO1", "TimePerceiver", "TimeXer", "TimesNet", "Transformer",
  "UMixer", "WPMixer", "WaveNet", "iTransformer", "xPatch",
]);

export type ModelType = "ts" | "st" | "aq" | "baseline";

export function getModelType(modelName: string): ModelType {
  if (BASELINE_MODELS.includes(modelName)) return "baseline";
  // Strip a trailing hardware suffix like " (Pro6000)".
  const base = modelName.replace(/\s*\([^)]+\)\s*$/, "");
  if (AIR_QUALITY_MODELS.has(base)) return "aq";
  if (SPATIOTEMPORAL_MODELS.has(base)) return "st";
  if (TIME_SERIES_MODELS.has(base)) return "ts";
  return "ts"; // unknown → treat as time series
}

// Models selectable in the stock-track trends chart.
export const STOCK_VIZ_MODELS = [
  "AGCRN", "AMRC", "APN", "Amplifier", "Aurora", "Autoformer", "BiMamba", "BiST",
  "BigST", "CARD", "CATS", "CMoS", "COSA", "CoRA", "CrossGNN", "CrossLinear",
  "Crossformer", "CycleNet", "D2STGNN", "DCRNN", "DFDGCN", "DGCRN", "DLinear",
  "DSFormer", "DSTAGNN", "DTAF", "DUET", "DeepAR", "DistDF", "DynamicTMoE",
  "ETSformer", "FEDformer", "FITS", "FTP", "FeTS", "FiLM", "FreTS", "Fredformer",
  "GOTSF", "GTR", "GTS", "GWNet", "HDMixer", "HL", "HMformer", "HN_MVTS",
  "HimNet", "ImplicitForecaster", "Informer", "InterPDN", "Koopa", "Kronos",
  "LSTM", "LatentTSF", "LightTS", "Linear", "MAFS", "MAGE", "MICN", "MMPD",
  "MSGNet", "MTGNN", "MTSMixer", "MambaSimple", "MixLinear", "MoFo", "ModernTCN",
  "MultiPatchFormer", "NBeats", "NHiTS", "NLinear", "NSTransformer", "OLinear",
  "OccamVTS", "PAttn", "PHAT", "PMDformer", "PULSE", "PWS", "PaiFilter",
  "PatchMLP", "PatchTST", "Pathformer", "PhaseFormer", "Pyraformer", "RLinear",
  "RPMixer", "Reformer", "S4", "SCINet", "SEMPO", "SOFTS", "SRSNet", "STAEformer",
  "STDN", "STGCN", "STGODE", "STID", "STNorm", "STOP", "STPGNN", "STTN", "STWave",
  "SVTime", "S_Mamba", "SegRNN", "Sonnet", "SparseTSF", "StemGNN", "Sumba",
  "SymTime", "TSMixer", "TSRAG", "TexFilter", "TiDE", "TiRex", "TimeAlign",
  "TimeBase", "TimeBridge", "TimeCAP", "TimeEmb", "TimeFilter", "TimeKAN",
  "TimeMixer", "TimeMosaic", "TimeO1", "TimePerceiver", "TimeXer", "TimesNet",
  "Transformer", "UMixer", "WPMixer", "WaveNet", "iTransformer", "xPatch",
];
