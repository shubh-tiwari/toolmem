"""Turn the benchmark results into a markdown table, significance tests and two PNG charts.

    python benchmark/report.py [results_dir]   # writes README_table.md, stats.md and *.png into results/

Per-condition metrics are recomputed from the ``runs.*.jsonl`` records (the source of truth), so
metrics added to ``run.summarize`` apply to earlier runs without new model calls.
"""
from __future__ import annotations

import json
import random
import sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
sys.path.insert(0, str(HERE))
from run import summarize  # noqa: E402

# Categorical palette (fixed slot order, validated for CVD safety) and text/surface tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


# One color per model family (fixed order, never cycled). Jev's server-then-tool routing shares
# Jev's color and is told apart by its marker, which keeps the scatter at three hues.
MODEL_COLOR = {"deepseek/deepseek-v4-flash": SERIES[0], "qwen/qwen3.7-flash": SERIES[1],
               "jev-latest": SERIES[2], "jev-latest-route": SERIES[2]}
MODEL_NAME = {"deepseek/deepseek-v4-flash": "DeepSeek V4 Flash", "qwen/qwen3.7-flash": "Qwen3.7 Flash",
              "jev-latest": "Jev", "jev-latest-route": "Jev, server then tool",
              "cascade-jev-deepseek-v4-flash": "Jev, then DeepSeek if unsure",
              "cascade-jev-qwen3.7-flash": "Jev, then Qwen if unsure"}


def charted(model: str) -> bool:
    """Cascades are table-only: they would add hues past the three the scatter can carry."""
    return not model.startswith("cascade-")
BEST = "toolmem_aug@20"  # the retrieval setup the charts feature


def short_model(model: str) -> str:
    return MODEL_NAME.get(model, model.split("/")[-1])


def color_of(model: str, i: int) -> str:
    return MODEL_COLOR.get(model, SERIES[i % len(SERIES)])


N_TOOLS = None  # set from summary.json in main(), so "all@all" renders as "all tools (454)"


def cond_label(cond: str) -> str:
    kind, _, arg = cond.partition("@")
    if kind == "all" and arg == "all" and N_TOOLS:
        arg = str(N_TOOLS)
    return {"all": f"all tools ({arg})", "toolmem": f"toolmem top-{arg}",
            "toolmem_aug": f"toolmem top-{arg} + augmented",
            "toolmem_shuffled": f"toolmem top-{arg}, shuffled",
            "toolmem_aug_shuffled": f"toolmem top-{arg} + augmented, shuffled"}[kind]


def size_of(cond: str, n_tools: int) -> int | None:
    kind, _, arg = cond.partition("@")
    if kind != "all":
        return None
    return n_tools if arg == "all" else min(int(arg), n_tools)


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT2)
    ax.yaxis.label.set_color(TEXT2)


def pct_axis(ax, lo=0.5, hi=1.02):
    ax.set_ylim(lo, hi)
    ticks = [t / 10 for t in range(int(lo * 10), 11)]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.0%}" for t in ticks])


def chart_accuracy_vs_size(summary: dict, out: Path) -> None:
    """Accuracy of the model picking from N tools, per model, with toolmem top-20 as reference lines."""
    n_tools = summary["meta"]["n_tools"]
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=160, facecolor=SURFACE)
    style(ax)
    handles = []
    for i, (model, entry) in enumerate(summary["models"].items()):
        if not charted(model):
            continue
        color = color_of(model, i)
        conds = entry["conditions"]
        if model == "jev-latest-route":
            s = conds.get("all@all")
            if s and s["accuracy"] is not None:
                h = ax.scatter([n_tools], [s["accuracy"]], marker="D", s=60, color=color, edgecolors=SURFACE,
                               linewidths=2, zorder=4, label=f"{short_model(model)} over all {n_tools}")
                handles.append(h)
            continue
        pts = sorted((size_of(c, n_tools), s["accuracy"]) for c, s in conds.items()
                     if size_of(c, n_tools) and s["accuracy"] is not None)
        if pts:
            xs, ys = zip(*pts)
            (h,) = ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, markerfacecolor=SURFACE,
                           markeredgewidth=2, label=f"{short_model(model)}: tools sent in full")
            handles.append(h)
        s = conds.get(BEST)
        if s and s["accuracy"] is not None:
            h = ax.axhline(s["accuracy"], color=color, linewidth=1.6, dashes=(4, 2),
                           label=f"{short_model(model)}: toolmem top-20 of {n_tools}")
            handles.append(h)
    ax.set_xscale("log")
    sizes = sorted({size_of(c, n_tools) for e in summary["models"].values() for c in e["conditions"]} - {None})
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    pct_axis(ax, 0.7)
    ax.set_xlabel("tools shown to the picker (log scale); dashed lines: toolmem shortlists 20 of the 454")
    ax.set_ylabel("tool-selection accuracy")
    ax.set_title(f"Accuracy vs. tools shown  ({summary['meta']['n_queries']} queries, {n_tools}-tool catalog)",
                 loc="left", fontsize=11, color=TEXT)
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, labelcolor=TEXT2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


SCATTER_CONDS = ("all@10", "all@200", "all@all", "toolmem_aug@5", "toolmem_aug@20")


def chart_tokens_vs_accuracy(summary: dict, out: Path) -> None:
    """Input tokens per request vs accuracy; one point per model and setup."""
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=160, facecolor=SURFACE)
    style(ax)
    import math
    placed: list[tuple[str, float, float]] = []
    for i, (model, entry) in enumerate(summary["models"].items()):
        if not charted(model):
            continue
        color = color_of(model, i)
        for cond, s in entry["conditions"].items():
            if cond not in SCATTER_CONDS or s["accuracy"] is None or not s["mean_input_tokens"]:
                continue
            route = model == "jev-latest-route"
            filled = route or not cond.startswith("all")
            ax.scatter(s["mean_input_tokens"], s["accuracy"], s=70 if not route else 60, marker="D" if route else "o",
                       color=color if filled else SURFACE, edgecolors=color if not route else SURFACE,
                       linewidths=2, zorder=3)
            label = "Jev, server then tool" if route else cond_label(cond).replace(" + augmented", "")
            x, y = math.log10(s["mean_input_tokens"]), s["accuracy"]
            if any(l == label and abs(px - x) < 0.05 and abs(py - y) < 0.015 for l, px, py in placed):
                continue  # same label already drawn on an overlapping point
            placed.append((label, x, y))
            ax.annotate(label, (s["mean_input_tokens"], s["accuracy"]), textcoords="offset points",
                        xytext=(-7, -3) if route else (7, -3), ha="right" if route else "left",
                        fontsize=7, color=TEXT2)
        if model != "jev-latest-route":
            ax.scatter([], [], color=color, label=short_model(model))
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    pct_axis(ax, 0.7, 1.02)
    ax.set_xlabel("mean input tokens per request (log scale)")
    ax.set_ylabel("tool-selection accuracy")
    ax.set_title("Tokens vs. accuracy (filled = toolmem shortlist, hollow = tools sent in full)",
                 loc="left", fontsize=11, color=TEXT)
    ax.legend(fontsize=8, frameon=False, loc="lower left", labelcolor=TEXT2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def markdown_table(summary: dict) -> str:
    n_tools = summary["meta"]["n_tools"]
    lines = ["| model | condition | accuracy | target first | confusable acc. | no tool | cut off | tools/req "
             "| input tok/req | list $/1k | billed $/1k | p50 latency |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def pct(x):
        return f"{x:.1%}" if x is not None else "n/a"

    for model, entry in summary["models"].items():
        for cond, s in entry["conditions"].items():
            if s["accuracy"] is None:
                continue
            conf = s["by_kind"].get("confusable", {}).get("accuracy")
            billed = s.get("billed_per_1k_requests_usd")
            lines.append(f"| {short_model(model)} | {cond_label(cond)} | {pct(s['accuracy'])} | "
                         f"{pct(s.get('target_accuracy'))} | {pct(conf)} | {pct(s['no_tool_rate'])} | "
                         f"{pct(s.get('cut_off_rate'))} | {s['tools_per_request']:.0f} | {s['mean_input_tokens']:,} | "
                         f"{s['cost_per_1k_requests_usd']:.2f} | {f'{billed:.2f}' if billed is not None else 'n/a'} | "
                         f"{s['latency_p50_s']:.1f}s |")
    lines.append("")
    lines.append("Retrieval only (does the expected tool appear in the top-k?):")
    lines.append("")
    lines.append("| embedder | mode | hit@1 | hit@3 | hit@5 | hit@10 | MRR |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for emb, modes in summary["retrieval"].items():
        for mode, r in modes.items():
            lines.append(f"| {emb} | {mode} | {r['hit@1']:.1%} | {r['hit@3']:.1%} | {r['hit@5']:.1%} | "
                         f"{r['hit@10']:.1%} | {r['mrr']:.3f} |")
    lines.append("")
    lines.append(f"{n_tools} tools, {summary['meta']['n_queries']} queries, embedder `{summary['meta']['embedder']}`.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- run records and stats
def load_runs(summary: dict, results: Path) -> dict[str, dict[str, dict[int, dict]]]:
    """{model: {condition: {query uid: record}}} from the runs files next to summary.json."""
    runs: dict = {}
    for model, entry in summary["models"].items():
        path = results / f"runs.{entry['provider']}.{model.replace('/', '_')}.jsonl"
        if not path.exists():
            continue
        by: dict = {}
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                by.setdefault(r["condition"], {})[r["i"]] = r
        runs[model] = by
    return runs


def refresh_from_runs(summary: dict, runs: dict) -> None:
    """Recompute every condition's summary from its records, keeping the recorded prices."""
    for model, conds in runs.items():
        entry = summary["models"][model]["conditions"]
        for cond, recs in conds.items():
            prices = entry.get(cond, {}).get("prices_per_mtok") or [0, 0, 0, 0]
            entry[cond] = summarize(list(recs.values()), prices)


def mcnemar(a: dict[int, dict], b: dict[int, dict]) -> tuple[int, int, int, float]:
    """Exact two-sided McNemar test on the queries both runs share: (n, a only, b only, p)."""
    ids = sorted(set(a) & set(b))
    a_only = sum(1 for i in ids if a[i]["correct"] and not b[i]["correct"])
    b_only = sum(1 for i in ids if b[i]["correct"] and not a[i]["correct"])
    n = a_only + b_only
    p = min(1.0, 2 * sum(comb(n, j) for j in range(min(a_only, b_only) + 1)) / 2 ** n) if n else 1.0
    return len(ids), a_only, b_only, p


def paired_ci(a: dict[int, dict], b: dict[int, dict], reps: int = 10_000, seed: int = 7) -> tuple[float, float, float]:
    """Accuracy difference a - b with a 95% paired bootstrap interval."""
    ids = sorted(set(a) & set(b))
    d = [a[i]["correct"] - b[i]["correct"] for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(reps))
    return sum(d) / len(d), boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1]


def holm(ps: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, for reading many tests at once."""
    order = sorted(range(len(ps)), key=lambda k: ps[k])
    adj, running = [0.0] * len(ps), 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[k]))
        adj[k] = running
    return adj


DS, QW, JEV, ROUTE = "deepseek/deepseek-v4-flash", "qwen/qwen3.7-flash", "jev-latest", "jev-latest-route"
# The comparisons the README's findings rest on: (question, model A, condition A, model B, condition B).
COMPARISONS = [
    ("accuracy falls with more tools", DS, "all@10", DS, "all@all"),
    ("accuracy falls with more tools", QW, "all@10", QW, "all@all"),
    ("top-20 shortlist vs all tools", DS, "toolmem_aug@20", DS, "all@all"),
    ("top-20 shortlist vs all tools", QW, "toolmem_aug@20", QW, "all@all"),
    ("top-5 vs top-20 shortlist", DS, "toolmem_aug@5", DS, "toolmem_aug@20"),
    ("top-5 vs top-20 shortlist", QW, "toolmem_aug@5", QW, "toolmem_aug@20"),
    ("top-50 vs top-20 shortlist", DS, "toolmem_aug@50", DS, "toolmem_aug@20"),
    ("top-50 vs top-20 shortlist", QW, "toolmem_aug@50", QW, "toolmem_aug@20"),
    ("rewritten vs original descriptions, top-5", DS, "toolmem_aug@5", DS, "toolmem@5"),
    ("rewritten vs original descriptions, top-5", QW, "toolmem_aug@5", QW, "toolmem@5"),
    ("Jev on top-20 vs LLM on top-20", JEV, "toolmem_aug@20", DS, "toolmem_aug@20"),
    ("Jev on top-20 vs LLM on all tools", JEV, "toolmem_aug@20", DS, "all@all"),
    ("Jev on top-20 vs LLM on all tools", JEV, "toolmem_aug@20", QW, "all@all"),
    ("200 random tools", JEV, "all@200", DS, "all@200"),
    ("200 random tools", JEV, "all@200", QW, "all@200"),
    ("best match first vs shuffled shortlist", DS, "toolmem_aug@20", DS, "toolmem_aug_shuffled@20"),
    ("best match first vs shuffled shortlist", QW, "toolmem_aug@20", QW, "toolmem_aug_shuffled@20"),
    ("best match first vs shuffled shortlist", JEV, "toolmem_aug@20", JEV, "toolmem_aug_shuffled@20"),
    ("Jev server-then-tool vs LLM on all tools", ROUTE, "all@all", DS, "all@all"),
    ("Jev server-then-tool vs LLM on all tools", ROUTE, "all@all", QW, "all@all"),
]
GATE_CONDS = ("toolmem_aug@20", "toolmem_aug@50")
GATE_THRESHOLDS = (0.5, 0.7, 0.8, 0.9)


def stats_markdown(summary: dict, runs: dict) -> str:
    from toolmem import calibration
    lines = ["# Significance tests and confidence gating", "",
             "Generated by `report.py` from the `runs.*.jsonl` records. Each test is an exact McNemar test "
             "on the queries both runs share; the interval is a 95% paired bootstrap of the accuracy "
             "difference (A minus B). With this many tests, read the Holm-adjusted column before calling "
             "a single result significant.", "",
             "| question | A | B | acc. A | acc. B | A - B [95% CI] | A only / B only | p | Holm p |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for q, ma, ca, mb, cb in COMPARISONS:
        a, b = runs.get(ma, {}).get(ca), runs.get(mb, {}).get(cb)
        if not a or not b:
            continue
        n, a_only, b_only, p = mcnemar(a, b)
        diff, lo, hi = paired_ci(a, b)
        acc_a = sum(r["correct"] for r in a.values()) / len(a)
        acc_b = sum(r["correct"] for r in b.values()) / len(b)
        rows.append((q, f"{short_model(ma)}, {cond_label(ca)}", f"{short_model(mb)}, {cond_label(cb)}",
                     acc_a, acc_b, diff, lo, hi, a_only, b_only, p))
    for row, adj in zip(rows, holm([r[-1] for r in rows])):
        q, la, lb, acc_a, acc_b, diff, lo, hi, a_only, b_only, p = row
        lines.append(f"| {q} | {la} | {lb} | {acc_a:.1%} | {acc_b:.1%} | {diff:+.1%} [{lo:+.1%}, {hi:+.1%}] | "
                     f"{a_only} / {b_only} | {p:.3f} | {adj:.3f} |")

    jev = runs.get(JEV, {})
    gated = [c for c in GATE_CONDS if c in jev]
    if gated:
        lines += ["", "## Jev confidence gating", "",
                  "Share of queries whose Jev pick reaches each confidence, and the accuracy of those picks. "
                  "ECE is the expected calibration error (0 = confidences match accuracy exactly).", "",
                  "| condition | threshold | coverage | accuracy of covered picks |", "|---|---:|---:|---:|"]
        for cond in gated:
            recs = list(jev[cond].values())
            confs = [(json.loads(r["text"] or "{}").get("confidence") or 0.0) for r in recs]
            ok = [bool(r["correct"]) for r in recs]
            for t in GATE_THRESHOLDS:
                idx = [k for k, c in enumerate(confs) if c >= t]
                acc = f"{sum(ok[k] for k in idx) / len(idx):.1%}" if idx else "n/a"
                lines.append(f"| {cond_label(cond)} | {t:.1f} | {len(idx) / len(recs):.0%} | {acc} |")
            ece, _ = calibration(confs, ok)
            lines.append(f"| {cond_label(cond)} | ECE | {ece:.3f} | |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    results = Path(argv[0]) if argv else RESULTS
    summary = json.load(open(results / "summary.json"))
    runs = load_runs(summary, results)
    refresh_from_runs(summary, runs)
    global N_TOOLS
    N_TOOLS = summary["meta"]["n_tools"]
    chart_accuracy_vs_size(summary, results / "accuracy_vs_tools.png")
    chart_tokens_vs_accuracy(summary, results / "tokens_vs_accuracy.png")
    table = markdown_table(summary)
    (results / "README_table.md").write_text(table + "\n")
    stats = stats_markdown(summary, runs)
    (results / "stats.md").write_text(stats + "\n")
    print(table)
    print()
    print(stats)
    print(f"\nwrote {results / 'accuracy_vs_tools.png'}, {results / 'tokens_vs_accuracy.png'}, "
          f"{results / 'README_table.md'}, {results / 'stats.md'}")


if __name__ == "__main__":
    main(sys.argv[1:])
