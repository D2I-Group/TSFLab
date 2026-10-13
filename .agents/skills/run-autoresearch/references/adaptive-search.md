# Adaptive method discovery

Use when the objective is to discover an improved method rather than tune one
supplied model. Fixed candidate tasks retain their supplied search space.

- Define the comparison panel, horizons, metric aggregation, confirmation seeds,
  compute budget and acceptance rule before results. A claim covers only this
  panel and protocol; beating one weak baseline is not beating the whole panel.
- Read one train/validation profile. Choose a few distinct full architectures from
  matching cards; include temporal and channel hypotheses when justified.
  Do not silently restrict discovery to one previously unsuccessful architecture.
- For a compact first-stage search, batch a small set of distinct controls and one
  profile-supported composition in one request. Return compact metrics, select the
  incumbent across all completed models, freeze, and evaluate its saved checkpoint.
  Record the reduced search breadth; do not present a supplied hypothesis panel as
  unrestricted autonomous discovery. Bind resource policy to the execution request,
  rather than only writing a policy file.
- Start with representative horizons at the final training fidelity. Cheap screens
  reject hypotheses; they cannot support a full-horizon superiority claim. Expand
  promising branches to all required horizons before confirmation. Record every
  cost, including rejected branches and failures.
- Retain the strongest protocol-matched baseline checkpoint as an eligible final
  incumbent. Select the lowest validation loss across controls and derived candidates;
  a worse derived model must not displace a stronger baseline merely to supply a new
  method name. If the baseline wins, report a baseline fallback and no discovery gain.
- Keep a small frontier: best aggregate validation result, non-dominated horizon
  tradeoffs and one distinct promising branch. A losing first candidate may still
  reveal a useful component. Avoid spending the whole budget on width changes.
- With code authorization, make a hypothesis-driven component or architecture
  change; preserve the full nonlinear backbone and label the derived method.
  Inspect the relevant component interface, check shapes/gradients, and change
  one factor first. Combine individually supported factors only with an explicit
  interaction hypothesis. Preserve the original baseline implementations.
- Keep a compact decision record containing candidate/config identity, hypothesis,
  native completion path, validation verdict and next action. Reuse successful
  records and checkpoints with matching protocol/config/seed; a framework update
  alone is not a reason to retrain. Reserve confirmation and frozen evaluation
  claims before expanding another branch. Register saved validation baselines
  with `tsf research reference ROUND RUN_DIR...` before screening in a new round;
  references add no training claims but retain their original cost attribution.
- Evaluate every fixed comparison-panel method under the same protocol before a
  panel-wide claim. Confirm the selected method with the declared seeds against
  matched baselines. Use native research screen for strict matched MSE/MAE gates;
  apply its failed-stop only when the declared search budget/patience ends.
- Freeze the candidate and comparison panel before test access. Evaluate saved
  checkpoints once; no subsequent search may use test outcomes. Report all
  horizons, failures and a failed objective honestly. Do not describe a valid
  no-winner workflow as achieving a method-improvement objective.
