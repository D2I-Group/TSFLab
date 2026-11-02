/* TSFLab — page-specific tables and the strings they build.
   Numbers are transcribed from the arXiv PDF (v1); nothing is computed here
   beyond best/second marking, which reproduces the paper's own marking. */
(function () {
  "use strict";

  const { $, ZH, tr, el, registerRender, rankMarks } = MC;

  /* ------------------------------------------------- runtime strings (zh) */
  Object.assign(ZH, {
    "Setting": "设置",
    "Completed ↑": "完成数 ↑",
    "Time (h) ↓": "时间（小时）↓",
    "Tokens (M) ↓": "Token（百万）↓",
    "Calls ↓": "调用次数 ↓",
    "Paper reproduction": "论文复现",
    "Benchmarking on new data": "新数据上的基准评测",
    "No workbench": "无工作台",
    "Full TSFLab": "完整 TSFLab",
    "Workbench module ablation": "工作台模块消融",
    "Agent ablation": "智能体消融",
    "w/o TSF-Domain Guide": "去掉 TSF-Domain Guide",
    "w/o Knowledge Structurer": "去掉 Knowledge Structurer",
    "w/o Experiment Workflower": "去掉 Experiment Workflower",
    "Copy": "复制",
    "Copied": "已复制"
  });

  /* Columns: [label, better, decimals]. Completed is shown as n/total and
     ranked on n (every group shares one total). */
  const COLS = [
    ["Completed ↑", "higher", null],
    ["Time (h) ↓", "lower", 2],
    ["Tokens (M) ↓", "lower", 3],
    ["Calls ↓", "lower", 2],
    ["ΔLOC ↓", "lower", 2],
    ["ΔFuncs ↓", "lower", 2],
    ["ΔCC ↓", "lower", 2]
  ];

  /* ============================================= main results (Table 2) */
  /* Each group is marked on its own, as in the paper. */
  const MAIN = [
    { group: "Paper reproduction", total: 12, rows: [
      ["TSFLab",       [12, 1.96, 0.434, 159.67, 148.08,  8.25, 25.83], true],
      ["TSLib",        [10, 2.61, 0.533, 164.60, 160.67,  9.67, 31.33]],
      ["TFB",          [ 7, 2.46, 0.487, 131.86, 153.42, 11.58, 36.00]],
      ["No workbench", [ 8, 3.50, 0.706, 212.75, 131.00, 11.00, 34.00]]
    ]},
    { group: "Benchmarking on new data", total: 48, rows: [
      ["TSFLab",       [42, 0.05, 0.022,  9.05,   0.00,  0.00,  0.00], true],
      ["TSLib",        [30, 0.06, 0.035, 12.50, 180.33,  9.17, 52.17]],
      ["TFB",          [16, 0.08, 0.065, 18.88, 345.67, 30.33, 76.50]],
      ["No workbench", [26, 0.08, 0.051, 15.85, 144.00, 16.33, 50.17]]
    ]}
  ];

  /* ================================================= ablations (Table 3) */
  /* Marked across the whole table, as in the paper. */
  const ABLATION = [
    { group: null, total: 12, rows: [
      ["Full TSFLab", [12, 1.96, 0.434, 159.67, 148.08, 8.25, 25.83], true]
    ]},
    { group: "Workbench module ablation", total: 12, rows: [
      ["w/o TSF-Domain Guide",      [3, 5.73, 1.129, 348.33, 147.33, 11.33, 39.42]],
      ["w/o Knowledge Structurer",  [8, 2.08, 0.438, 118.75, 171.42,  9.42, 31.58]],
      ["w/o Experiment Workflower", [5, 2.24, 0.533, 153.00, 178.75, 10.42, 36.42]]
    ]},
    { group: "Agent ablation", total: 12, rows: [
      ["Claude Code, Sonnet 5.5", [11, 2.38, 0.195, 78.00, 212.17, 13.50, 36.25]],
      ["Codex CLI, GPT-6 Sol",    [12, 2.88, 0.394, 82.25, 132.17,  9.25, 23.67]]
    ]}
  ];

  function cell(v, j, total) {
    return COLS[j][2] === null ? v + "/" + total : v.toFixed(COLS[j][2]);
  }

  function header(table) {
    const head = table.querySelector("thead");
    head.innerHTML = "";
    const hr = document.createElement("tr");
    hr.appendChild(el("th", "name", tr("Setting")));
    COLS.forEach(([label]) => hr.appendChild(el("th", null, tr(label))));
    head.appendChild(hr);
  }

  function marksFor(rows) {
    return COLS.map((c, j) => rankMarks(rows.map((r) => r[1][j]), c[1]));
  }

  function body(table, groups, markEachGroup) {
    const tbody = table.querySelector("tbody");
    tbody.innerHTML = "";
    const all = groups.flatMap((g) => g.rows);
    const global = markEachGroup ? null : marksFor(all);
    let offset = 0;
    groups.forEach((g) => {
      if (g.group) {
        const sr = el("tr", "section-row");
        const td = el("td", null, tr(g.group));
        td.colSpan = COLS.length + 1;
        sr.appendChild(td);
        tbody.appendChild(sr);
      }
      const local = markEachGroup ? marksFor(g.rows) : null;
      g.rows.forEach(([name, vals, ours], i) => {
        const row = document.createElement("tr");
        row.appendChild(el("td", "name", tr(name)));
        vals.forEach((v, j) => {
          const mark = markEachGroup ? local[j][i] : global[j][offset + i];
          const cls = [mark, ours ? "ours" : ""].filter(Boolean).join(" ");
          row.appendChild(el("td", cls || null, cell(v, j, g.total)));
        });
        tbody.appendChild(row);
      });
      offset += g.rows.length;
    });
  }

  registerRender(function renderMain() {
    const t = $("#main-table");
    if (!t) return;
    header(t);
    body(t, MAIN, true);
  });

  registerRender(function renderAblation() {
    const t = $("#ablation-table");
    if (!t) return;
    header(t);
    body(t, ABLATION, false);
  });

  MC.start();   /* always last */
})();
