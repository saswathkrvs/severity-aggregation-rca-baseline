"""The paper's figures, drawn from the analysis output (``evaluation.analysis.compute``) and the per-case records.

    python -m evaluation.analysis --results results --figures figs

Writes PDF and PNG (300 dpi) files. Colours: a fixed categorical order (blue, orange, aqua, yellow, magenta) with distinct
markers, so that no figure depends on colour alone; text is black or grey.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SUITE_LABEL = {"H1": "RE2-H1 (dev.)", "H2": "RE2-H2", "RE1": "RE1", "RE3": "RE3", "PS": "PetShop", "NZ": "Nezha"}


def _style(plt):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42})


def _save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(out / (name + ".png"), dpi=300, bbox_inches="tight")


def forest(res, out):
    """Paired Top-1 difference of the log-compressed sum against each comparator, per held-out suite, in
    percentage points of the suite, with 95 % cluster-bootstrap intervals."""
    import matplotlib.pyplot as plt
    _style(plt)
    suites = ["RE1", "RE3", "H2", "PS", "NZ"]
    comps = [("agg:max", "maximum aggregation", BLUE, "o"), ("cmp:nsigma", "N-Sigma", ORANGE, "s"),
             ("cmp:baro", "BARO scorer", AQUA, "D"), ("cmp:simplerca", "SimpleRCA metric rules", YELLOW, "^")]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    y = 0.0
    yticks, ylabels = [], []
    for s in reversed(suites):
        n = res["n"][s]
        for j, (m, label, color, marker) in enumerate(comps):
            p = res["paired"][s][m]
            yy = y + (len(comps) - 1 - j) * 0.17
            lo, hi = 100.0 * p["ci95"][0] / n, 100.0 * p["ci95"][1] / n
            ax.plot([lo, hi], [yy, yy], color=color, lw=1.6, solid_capstyle="round")
            ax.plot([100.0 * p["diff"] / n], [yy], marker=marker, color=color, ms=4.5, mec="white", mew=0.6,
                    label=label if s == suites[-1] else None, linestyle="none")
        yticks.append(y + 0.255)
        ylabels.append("%s (n = %d)" % (SUITE_LABEL[s], n))
        y += 1.0
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Top-1 difference, log-compressed sum minus comparator (percentage points)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, handletextpad=0.3,
              columnspacing=1.2)
    _save(fig, out, "fig_forest")
    plt.close(fig)


def robustness(res, out):
    """Top-1 (share of the suite) against onset error and against the anomalous window length, RE1 and RE3."""
    import matplotlib.pyplot as plt
    _style(plt)
    methods = [("v11", "log-compressed sum", BLUE, "o", "-"),
               ("v11sd", "log-compressed sum, SD scale", MAGENTA, "o", (0, (3.5, 1.8))),
               ("v1", "maximum aggregation", ORANGE, "s", "-"),
               ("nsigma", "N-Sigma", AQUA, "D", "-"), ("baro", "BARO scorer", YELLOW, "^", "-")]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.0), sharey=True)
    shifts = [-120, -60, -30, 0, 30, 60, 120]
    lengths = [60, 120, 180, 300]
    for col, s in enumerate(("RE1", "RE3")):
        sens = res["sensitivity"][s]
        n = res["n"][s]
        ax = axes[2 * col]
        for m, label, color, marker, ls in methods:
            vals = [sens["default"][m] if x == 0 else sens["shift:%+d" % x][m] for x in shifts]
            ax.plot(shifts, [100.0 * v / n for v in vals], color=color, marker=marker, ms=3.5, lw=1.4, ls=ls,
                    label=label, mec=color if ls != "-" else "white", mfc="white" if ls != "-" else color, mew=0.7)
        ax.set_xticks([-120, -60, 0, 60, 120])
        ax.set_xlabel("onset error (s)")
        ax.set_title("(%s) %s: onset" % ("ac"[col], s), loc="left", color=INK)
        ax = axes[2 * col + 1]
        for m, label, color, marker, ls in methods:
            vals = [sens["win:300-%d" % x][m] if x != 300 else sens["default"][m] for x in lengths]
            ax.plot(lengths, [100.0 * v / n for v in vals], color=color, marker=marker, ms=3.5, lw=1.4, ls=ls,
                    mec=color if ls != "-" else "white", mfc="white" if ls != "-" else color, mew=0.7)
        ax.set_xticks(lengths)
        ax.set_xlabel("anomalous window (s)")
        ax.set_title("(%s) %s: window" % ("bd"[col], s), loc="left", color=INK)
    axes[0].set_ylabel("Top-1 (% of incidents)")
    for ax in axes:
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(2.35, 1.42), ncol=5, frameon=False, handlelength=3.0,
                   columnspacing=1.0)
    _save(fig, out, "fig_robustness")
    plt.close(fig)


def mechanism(res, results_root, out):
    """Breadth against peak on the RCAEval suites: for every case where the maximum and the log-compressed sum
    disagree at Top-1, the labelled service's number of series with s >= 1 and its largest s, each minus the
    strongest competitor's (the other service with the largest single series score)."""
    import matplotlib.pyplot as plt
    from evaluation import analysis as A
    _style(plt)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    pts = {"gain": [], "loss": []}
    for s in ("RE1", "RE3", "H2"):
        for r in A.load_study(results_root, s):
            a, b = A.hit(r, A.V11), A.hit(r, "agg:max")
            if a == b:
                continue
            nodes = r["nodes"]
            label = r["service"] if s in ("PS", "NZ") else r["service"].replace("-db", "")

            def svc(n):
                return n if s in ("PS", "NZ") else n.split("_")[0].replace("-db", "")
            lab = [n for n in nodes if svc(n) == label]
            comp = [n for n in nodes if svc(n) != label]
            if not lab or not comp:
                continue
            ln = sorted(lab, key=lambda n: (-nodes[n][4], n))[0]
            cn = sorted(comp, key=lambda n: (-nodes[n][3], n))[0]
            db = nodes[ln][1] - nodes[cn][1]
            dp = np.log10(1 + nodes[ln][3]) - np.log10(1 + nodes[cn][3])
            pts["gain" if a else "loss"].append((db, dp))
    rng = np.random.default_rng(0)
    for key, label, color, marker in (("gain", "only the log-compressed sum correct", BLUE, "o"),
                                      ("loss", "only the maximum correct", ORANGE, "s")):
        xy = np.array(pts[key]) if pts[key] else np.zeros((0, 2))
        jitter = rng.uniform(-0.18, 0.18, size=len(xy))
        ax.scatter(xy[:, 0] + jitter, xy[:, 1], s=13, color=color, marker=marker, edgecolors="white",
                   linewidths=0.4, label="%s (%d)" % (label, len(xy)), zorder=3)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.axvline(0, color=MUTED, lw=0.7)
    ax.set_xlabel("breadth: series with s ≥ 1, label minus competitor")
    ax.set_ylabel("peak: log10(1 + largest s), label minus competitor")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=1)
    _save(fig, out, "fig_mechanism")
    plt.close(fig)


REF_LINES = [("mad_logsum", "log-compressed sum, MAD scale (v1.1)", BLUE, "o", "-"),
             ("sd_logsum", "log-compressed sum, standard-deviation scale", MAGENTA, "o", (0, (3.5, 1.8))),
             ("mad_max", "maximum, MAD scale (v1)", ORANGE, "s", "-"),
             ("sd_max", "maximum, standard-deviation scale", VIOLET, "v", (0, (1.2, 1.4))),
             ("sd_dc_max", "N-Sigma's configuration (standard-deviation scale, maxima)", AQUA, "D", "-")]


def reference_length(res, out):
    """Top-1 (share of the suite) against the length of the normal window, everything else fixed, per held-out
    suite (``evaluation.controls``, ``reflen``). RCAEval: the last L seconds before the onset (dotted line: the
    study's 300 s). PetShop: L contiguous 5-minute rows of the fault-free period. Nezha, offline: the L fault-free
    minutes of the day nearest the onset on either side, ending with the study's day-wide reference; Nezha, real
    time: the L most recent fault-free minutes before the onset."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
    _style(plt)
    day = 1300
    panels = [("RE1", "(a) RE1", "seconds before the onset", None),
              ("RE3", "(b) RE3", "seconds before the onset", None),
              ("H2", "(c) RE2-H2", "seconds before the onset", None),
              ("PS", "(d) PetShop", "5-minute rows of the fault-free period", None),
              ("NZ", "(e) Nezha, offline reference", "fault-free minutes nearest the onset", "nearest:"),
              ("NZ", "(f) Nezha, real-time reference", "fault-free minutes before the onset", "preonset:")]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.8), sharey=True)
    for ax, (s, title, unit, prefix) in zip(axes.flat, panels):
        ref = res["controls"][s]["reflen"]
        n = res["n"][s]
        if prefix:
            keys = [k for k in ref if k.startswith(prefix)] + (["day"] if prefix == "nearest:" else [])
            xs = [day if k == "day" else int(k.split(":")[1]) for k in keys]
            labels = ["day" if k == "day" else k.split(":")[1] for k in keys]
        else:
            keys = list(ref)
            xs = [int(k) for k in keys]
            labels = [str(x) for x in xs]
        for m, label, color, marker, ls in REF_LINES:
            ax.plot(xs, [100.0 * ref[k][m] / n for k in keys], color=color, marker=marker, ms=3.4, lw=1.3, ls=ls,
                    label=label, mec=color if ls != "-" else "white", mfc="white" if ls != "-" else color, mew=0.7)
        if s in ("RE1", "RE3", "H2"):
            ax.axvline(300, color=MUTED, lw=0.7, ls=":")
            ax.text(300, 3, " study", color=MUTED, fontsize=6.5, ha="left", va="bottom")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(xs))
        ax.xaxis.set_major_formatter(FixedFormatter(labels))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel(unit, fontsize=7)
        ax.set_ylim(0, 100)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Top-1 (% of incidents)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               handlelength=3.0, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.885), h_pad=1.2)
    _save(fig, out, "fig_reference")
    plt.close(fig)


FAULT_LABEL = {"cpu": "CPU", "mem": "memory", "disk": "disk", "delay": "delay", "loss": "packet loss",
               "socket": "socket", "f1": "code fault F1", "f2": "code fault F2", "f3": "code fault F3",
               "f4": "code fault F4", "f5": "code fault F5", "error": "error", "overload": "overload",
               "cpu_consumed": "CPU consumption", "cpu_contention": "CPU contention", "exception": "exception",
               "network_delay": "network delay", "return": "return value"}
FAULT_LINES = [("v11", "log-compressed sum (v1.1)", BLUE, "o"), ("v1", "maximum (v1)", ORANGE, "s"),
               ("nsigma", "N-Sigma", AQUA, "D"), ("torai_stage", "TORAI severity stage", VIOLET, "v"),
               ("simplerca", "SimpleRCA metric rules", YELLOW, "^")]


def by_fault(res, out):
    """Top-1 (share of the fault type's incidents) per held-out suite and fault type."""
    import matplotlib.pyplot as plt
    _style(plt)
    panels = [("RE1", "RE3"), ("H2", "PS", "NZ")]
    fig = plt.figure(figsize=(7.1, 3.7))
    boxes = [(0.165, 0.11, 0.31, 0.78), (0.69, 0.11, 0.295, 0.78)]
    for box, suites in zip(boxes, panels):
        ax = fig.add_axes(box)
        rows = []
        for s in suites:
            for f in [f for f in res["by_fault"][s] if not f.startswith("_")]:
                rows.append((s, f, res["by_fault"][s][f]))
        y = np.arange(len(rows))[::-1]
        for (m, label, color, marker), off in zip(FAULT_LINES, np.linspace(-0.26, 0.26, len(FAULT_LINES))):
            ax.scatter([100.0 * r[2][m] / r[2]["n"] for r in rows], y + off, s=13, color=color, marker=marker,
                       edgecolors="white", linewidths=0.4, label=label, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(["%s %s (%d)" % (SUITE_LABEL[s], FAULT_LABEL.get(f, f), r["n"]) for s, f, r in rows],
                           fontsize=6.8)
        for i in range(1, len(rows)):
            if rows[i][0] != rows[i - 1][0]:
                ax.axhline(y[i] + 0.5, color=MUTED, lw=0.6)
        ax.set_xlim(-3, 103)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Top-1 (% of the fault type's incidents)")
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if suites[0] == "RE1":
            handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False,
               columnspacing=1.2, handletextpad=0.3)
    _save(fig, out, "fig_faults")
    plt.close(fig)


def make_all(results_root, out, res):
    forest(res, out)
    robustness(res, out)
    mechanism(res, results_root, out)
    if res.get("controls"):
        reference_length(res, out)
    by_fault(res, out)
