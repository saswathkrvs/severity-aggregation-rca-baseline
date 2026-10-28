# Severity aggregation for service-level root-cause ranking on RCAEval

Code and per-case results for the paper *A Training-Free Severity-Aggregation Baseline for Service-Level
Root-Cause Ranking on RCAEval*.

The ranker scores every metric (and, where available, trace) series of an incident with a one-sided robust
n-sigma statistic — the 90th percentile of the anomalous window's z-scores against the window before the onset —
and ranks services by aggregating their series scores. Version 1.1 sums `log(1 + s)` over a service's usable
series; version 1 takes the maximum. No training, no graph and no labels are needed.

## Contents

| Path | |
| --- | --- |
| `src/aiops_kg/graph_rank.py` | v1: windows, series scores, maximum aggregation, optional personalised PageRank over a graph |
| `src/aiops_kg/graph_rank_v11.py` | v1.1: the compressed sum (everything else is v1) |
| `evaluation/datasets.py` | case discovery; RE1/RE3 cases written into the common layout (`metrics.json` decoded with RCAEval's reader) |
| `evaluation/rankers.py` | the in-process methods: v1, v1.1, graph variants, aggregation alternatives, sensitivity checks |
| `evaluation/comparators.py` | N-Sigma, BARO, random order, RCD, CIRCA, ε-Diagnosis (RCAEval) and the TORAI severity stage |
| `evaluation/pinned_runner.py` | TORAI and the BARO scorer on RE2 through RCAEval's own runner |
| `evaluation/run.py` | runs methods on a suite and writes the per-case records |
| `evaluation/stats.py`, `evaluation/tables.py` | measures, cluster bootstrap and exact tests; every table of the paper |
| `evaluation/re2_heldout_splits.json` | the RE2 development cases and held-out splits H1 and H2 |
| `results/<suite>/<method>.jsonl` | one record per case and method (below) |
| `tests/` | unit tests, and a test that every published number is reproduced from `results/` |

## Reproduce the tables (no benchmark data needed)

```
pip install -r requirements.txt
python -m evaluation.tables --check
```

This prints Tables 1–5 and every number quoted in the text, computed from `results/`, and compares each with
the published value. `python -m unittest discover -s tests` runs the tests (with `src` on `PYTHONPATH`).

## Rerun the methods

Data (not redistributed here):

* **RE1, RE3**: `RCAEval-v2.zip` of the RCAEval benchmark, Figshare article 31048672
  (doi:10.6084/m9.figshare.31048672.v1); 375 RE1 and 90 RE3 cases.
* **RE2**: the TORAI data release (doi:10.6084/m9.figshare.31925976.v1, `torai-data.zip`); 18 development cases,
  held-out splits H1 (162 cases) and H2 (90 cases).

```
python -m evaluation.external                      # RCAEval bb48c5aa and its comparators' sources into external/
python -m evaluation.datasets --archive <unpacked RCAEval-v2> --suite RE1 --out prepared/RE1 \
    --rcaeval-src external/RCAEval-src --extra-path external/tqdm-src            # likewise RE3

export PYTHONPATH=src
python -m evaluation.run --suite RE1 --cases prepared/RE1 --methods v11,v1,v1_inferred,sum_all,count3_max
python -m evaluation.run --suite RE1 --cases prepared/RE1 --methods nsigma,baro,dummy,torai_stage,rcd,circa,e_diagnosis
python -m evaluation.run --suite H2 --cases <unpacked torai-data> --methods v11,v1,v1_inferred,v11_inferred,v11_mean,v11_no_reference,nsigma,baro,dummy
python -m evaluation.run --suite H1 --cases <unpacked torai-data> --methods v1_placebo,v11_placebo
python -m evaluation.tables --results results
```

Records are written to `results/<suite>/<method>.jsonl`; finished cases are skipped, so an interrupted run
continues where it stopped (`--out` writes elsewhere, `--only` limits a run to listed cases). The rankers are
deterministic; the comparators run with seed 0 in a fresh process each, one numerical thread, and a time limit of
180 s (N-Sigma, BARO, random order) or 600 s (RCD, CIRCA, ε-Diagnosis). TORAI and the BARO scorer on RE2 were run
with RCAEval's own `main.py` in its Python 3.8 environment (`requirements-runner.txt`):

```
python -m evaluation.pinned_runner --suite H2 --cases <unpacked torai-data> --methods torai,baro \
    --rcaeval <RCAEval checkout at bb48c5aa> --python <that environment's python>
```

## Methods and records

| Method | Suites | |
| --- | --- | --- |
| `v11` | RE1, RE3, H1, H2 | v1.1, compressed sum |
| `v1` | RE1, RE3, H1, H2 | v1, maximum |
| `v1_inferred`, `v11_inferred` | RE1, RE3, H1, H2 / H1, H2 | personalised PageRank over the inferred correlation graph |
| `sum_all`, `count3_max` | RE1, RE3 | aggregation alternatives: plain sum; number of series with s ≥ 3, then maximum |
| `v11_mean`, `v11_no_reference` | H1, H2 | sensitivity checks: mean instead of sum; normal-window self-score removed |
| `v1_placebo`, `v11_placebo` | H1 | both windows before the fault (onset moved 300 s earlier) |
| `nsigma`, `baro`, `dummy` | RE1, RE3, H1, H2 | RCAEval N-Sigma, BARO scorer, random order (seed 0) |
| `torai_stage` | RE1, RE3 | the metric severity stage of the pinned TORAI implementation only |
| `rcd`, `circa`, `e_diagnosis` | RE1, RE3 | RCAEval RCD, CIRCA, ε-Diagnosis |
| `torai`, `baro_runner` | H1, H2 | TORAI and the BARO scorer through RCAEval's runner |

Every record holds the case (`suite`, `system`, `case`, `service`, `fault`, `instance`), the `method`, a `status`
and, when completed, `service_result`: the rank of the labelled service, Top-1/3/5 hits, Avg@5, the number of
ranked services and whether the labelled service was among them. Ranker records add the windows, the candidate
nodes, the first 10 services and 15 nodes of the ranking with their scores and series counts, and the usable series
per node; comparator records add the raw ranking (`raw_ranks`), the services derived from it, the windows and
timings (`function_seconds` is the method call alone, `process_seconds` includes start-up and imports). A failed,
timed-out or invalid run scores 0 and stays in the denominator.

Comparator adapters, recorded in each record: RCAEval's version-routed imports are bypassed by setting
`RCAEval.utility.is_py312` to return True; CIRCA's `@rca` wrapper is bypassed so that an exception is recorded as
a failure rather than replaced by the unranked column order (`called_undecorated_inner_function`), and its unused
statsmodels imports are satisfied by an import-only stub (`statsmodels_import_stub`). A later RCAEval fix
(29 September 2026) concerns that wrapper and a duplicated `time` column; no record here contains either.

## Reproduction check

The published records were computed under Python 3.11. A rerun under Python 3.13 with current numpy, pandas, scikit-learn and scipy reproduced them: the decoded inputs were byte-identical; the rankers' records were identical on all 717 cases; the TORAI severity stage on 465 of 465; N-Sigma, BARO and the random order on every case of RE1, RE3, H1 and H2; RCD and ε-Diagnosis on seven sampled cases. For CIRCA on seven sampled cases, the rank of the labelled service and every scored measure were identical, but in four of them the raw ranking (`raw_ranks`) ordered near-tied metrics differently, an effect of the numerical library versions.

## License

Code: MIT (`LICENSE`). Results in `results/`: CC BY 4.0. The benchmark data belong to their publishers and are
not included.
