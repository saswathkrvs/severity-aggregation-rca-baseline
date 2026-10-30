# Evidence aggregation for training-free root-cause ranking

Code and per-case results for the paper *Breadth or Peak? Evidence Aggregation in Training-Free Root-Cause
Ranking across Five Microservice Benchmark Suites*.

The ranker scores every metric (and, where available, trace) series of an incident with a one-sided robust
statistic — the 90th percentile of the anomalous window's z-scores against the median and MAD of the window
before the onset — and ranks services by aggregating their series scores. Version 1.1 sums `log(1 + s)` over a
service's usable series; version 1 takes the maximum. No training, no graph and no labels are needed. The study
compares fifteen aggregation rules, a factorial over scale estimator, statistic and aggregation, and external
comparators on identical inputs, on five suites: RCAEval RE1, RE3 and RE2 (development split H1, held-out split
H2, the third repetitions of the development scenarios), PetShop and Nezha. Controls examine instrumentation
density (equal series counts, the fault-free noise floor, a score with that floor subtracted) and the normal window
(its length varied within each suite; on Nezha also a reference available in real time, before the onset). The
analysis separates the RCAEval scenarios that occur in development from those that do not, and repeats the choice of
the aggregation rule with groups of scenarios held out.

## Contents

| Path | |
| --- | --- |
| `src/aiops_kg/graph_rank.py` | v1: windows, series scores, maximum aggregation, optional personalised PageRank over a graph |
| `src/aiops_kg/graph_rank_v11.py` | v1.1: the log-compressed sum (everything else is v1) |
| `evaluation/engine.py` | the study's scoring engine: series statistics, the aggregation rules, rankings, and the comparators re-implemented from RCAEval (N-Sigma, BARO scorer, TORAI severity stage) and SimpleRCA (metric rules) |
| `evaluation/bocpd.py` | BARO's onset detector (multivariate Bayesian online change-point detection), for the full BARO pipeline |
| `evaluation/newdata.py` | the PetShop and Nezha cases and their windows |
| `evaluation/study.py` | runs every method and variant on a suite and writes one record per case |
| `evaluation/controls.py` | the controls: equal series counts, the fault-free noise floor, normal windows of different lengths, the maximum over metric series only; run times of every method under one protocol |
| `evaluation/analysis.py`, `evaluation/figures.py` | every table, number and figure of the paper, from the records |
| `evaluation/datasets.py`, `rankers.py`, `comparators.py`, `pinned_runner.py`, `run.py` | the original runs: case discovery and decoding, the in-process rankers, the pinned RCAEval comparators (including RCD, CIRCA, ε-Diagnosis) and TORAI on RE2 |
| `evaluation/stats.py`, `evaluation/tables.py` | measures and the tables of the original runs |
| `evaluation/re2_heldout_splits.json` | the RE2 development cases and the splits H1 and H2 |
| `results/<suite>/study.jsonl` | the study's records (below); suites `RE1`, `RE3`, `H1`, `H2`, `PS` (PetShop), `NZ` (Nezha) |
| `results/<suite>/bocpd.jsonl` | the BARO pipeline with its own onset (RCAEval suites) |
| `results/PS/placebo.jsonl`, `results/NZ/study_notolerance.jsonl` | PetShop fault-free windows; Nezha without the spread tolerance |
| `results/<suite>/controls.jsonl`, `results/<suite>/timing.jsonl` | the controls (held-out suites and RE2-H1); run times (RE1, RE3) |
| `results/<suite>/<method>.jsonl` | the original runs' records, one per case and method |
| `tests/` | unit tests, and tests that every published number is reproduced from `results/` |

## Reproduce the paper's numbers (no benchmark data needed)

```
pip install -r requirements.txt
python -m evaluation.analysis --check --figures figs
```

This recomputes the paper's tables and reported results from `results/`, compares each with the published value,
and draws the figures. `--json analysis.json` also writes everything the analysis computes, including Top-3, Top-5
and MRR per method, suite and fault type. `python -m evaluation.tables --check` does the same for the tables of the
original runs.
`python -m unittest discover -s tests` runs the tests (with `src` on `PYTHONPATH`).

## Rerun the methods

Data (not redistributed here):

* **RE1, RE3**: `RCAEval-v2.zip` of the RCAEval benchmark, Figshare article 31048672
  (doi:10.6084/m9.figshare.31048672.v1); 375 RE1 and 90 RE3 cases.
* **RE2**: the TORAI data release (doi:10.6084/m9.figshare.31925976.v1, `torai-data.zip`); 18 development cases,
  splits H1 (162 cases) and H2 (90 cases).
* **PetShop**: the `dataset/` folder of https://github.com/amazon-science/petshop-root-cause-analysis (CC BY 4.0);
  68 issues.
* **Nezha**: the `rca_data/` folder of https://github.com/IntelligentDDS/Nezha; 101 faults.

```
python -m evaluation.external                      # RCAEval bb48c5aa and its comparators' sources into external/
python -m evaluation.datasets --archive <unpacked RCAEval-v2> --suite RE1 --out prepared/RE1 \
    --rcaeval-src external/RCAEval-src --extra-path external/tqdm-src            # likewise RE3

export PYTHONPATH=src
python -m evaluation.study --suite RE1 --cases prepared/RE1 --validate        # likewise RE3
python -m evaluation.study --suite H2 --cases <unpacked torai-data> --validate  # likewise H1
python -m evaluation.study --suite PS --cases <petshop>/dataset
python -m evaluation.study --suite NZ --cases <Nezha checkout>
python -m evaluation.study --suite RE1 --cases prepared/RE1 --bocpd           # likewise RE3, H1, H2
python -m evaluation.study --suite PS --cases <petshop>/dataset --placebo
python -m evaluation.study --suite NZ --cases <Nezha checkout> --no-tolerance
python -m evaluation.controls --suite RE1 --cases prepared/RE1                # likewise RE3
python -m evaluation.controls --suite H2 --cases <unpacked torai-data>          # likewise H1
python -m evaluation.controls --suite PS --cases <petshop>/dataset
python -m evaluation.controls --suite NZ --cases <Nezha checkout>
python -m evaluation.controls --suite RE1 --cases prepared/RE1 --timing       # likewise RE3
python -m evaluation.analysis --check --figures figs
```

`--validate` compares the study's ranks with the original runs' records of the same methods: the ranker (v1, v1.1
and its variants) and the re-implemented N-Sigma, BARO scorer and TORAI severity stage give the same rank of the
labelled service as the recorded runs in every case. Finished cases are skipped, so an interrupted run continues
where it stopped. Everything is deterministic. The recorded study times per case (all variants, field `seconds`)
have medians of 1.3 to 2.3 s on the RCAEval suites (means 3.1 to 4.5 s), 0.7 s on PetShop and 1.2 s on Nezha
(mean 2.5 s); the BARO pipeline's onset detection takes a median of 0.2 to 0.3 s per case (`bocpd_seconds`). The
controls take a median of about 0.5 s per RCAEval case, 15 s per PetShop issue and 4 s per Nezha fault (up to
about 20 s for the largest Train Ticket faults). Run times depend on the machine: the analysis reads the recorded
ones (`timing.jsonl`, CPU time from an incident's in-memory windows to its service ranking).

The original runs (`results/<suite>/<method>.jsonl`; RCD, CIRCA, ε-Diagnosis, TORAI and the random order are only
there) were produced with `evaluation.run` and, for TORAI and BARO on RE2, with RCAEval's own `main.py` through
`evaluation.pinned_runner`:

```
python -m evaluation.run --suite RE1 --cases prepared/RE1 --methods v11,v1,v1_inferred,sum_all,count3_max
python -m evaluation.run --suite RE1 --cases prepared/RE1 --methods nsigma,baro,dummy,torai_stage,rcd,circa,e_diagnosis
python -m evaluation.pinned_runner --suite H2 --cases <unpacked torai-data> --methods torai,baro \
    --rcaeval <RCAEval checkout at bb48c5aa> --python <that environment's python>
```

## Study records

Every line of `results/<suite>/study.jsonl` holds the case (`suite`, `system`, `case`, `service`, `fault`,
`instance`, `cluster`), `ranks` (method -> rank of the labelled service; `null` when the label is absent or the
method failed), `top` (method -> first-ranked service, main methods), `abstain` (no candidate has positive evidence;
every method then counts the case as a miss), `candidates`, `services`, `series`, `usable_series`, and `nodes`
(per candidate node: usable series, series with s >= 1, series with s >= 3, largest s, log-compressed sum, sum).
Method names:

| Name | |
| --- | --- |
| `agg:<rule>` | the ranker's series scores with each aggregation rule (`agg:logsum` = v1.1, `agg:max` = v1) |
| `fact:<scale>-<statistic>-<aggregation>` | the factorial: scale `mad`/`std`/`iqr`, statistic `max`/`p90`, aggregation `max`/`logsum` |
| `cmp:nsigma`, `cmp:baro`, `cmp:torai_stage` | N-Sigma, BARO scorer, TORAI severity stage (re-implemented, validated) |
| `cmp:simplerca`, `cmp:alerts3` | SimpleRCA's metric rules; series with an observation above mean + 3 SD, counted |
| `var:<variant>` | v1.1 with one change: `ref`, `two_sided`, `q50`/`q75`/`q95`/`qmax`, `fam_dedup`, `fam_common` |
| `var:sd_dc_max`, `var:sd_dc_logsum` | the factorial's standard-deviation maximum and log sum with N-Sigma's removal of series that are constant in the anomalous window |
| `win:<n>-<a>:<m>`, `shift:<s>:<m>`, `interp:<m>`, `placebo:<m>` | windows, onset shifts, interpolation, pre-fault windows (RCAEval); `m` is `v11`, `v11sd` (the log sum on a standard-deviation scale), `v1`, `nsigma` or `baro` |
| `nostore:<m>` | datastore nodes removed before ranking (RCAEval) |
| `local:<m>` | Nezha with the local pre-onset normal window |

## Control records

Every line of `results/<suite>/controls.jsonl` holds the case, `metric_max` (rank of the labelled service under the
maximum over metric series only), `equal` (`k_min`, the number of random draws `repeats`, and per mode `pair_top`,
`pair_random`, `min`, `cap5` and per aggregation `max`, `logsum`, `mean`, `ref_logsum` the number of draws in which
the labelled service is first), `null` (fault-free windows: their number, first places per aggregation, and per
node the usable series, the null log sum, the null maximum, the series with a positive score and the
reference-adjusted log sum) and `reflen` (per normal-window length: its rows, the number of blocks, and first places
of `mad_logsum`, `sd_logsum`, `mad_max`, `sd_max` and `sd_dc_max`; for Nezha's pre-onset references also
`sd_dc_max_no_tolerance`, N-Sigma's configuration without the zero-spread tolerance). `timing.jsonl` holds per case
the CPU seconds of `v11`, `v1`, `nsigma`, `baro`, `torai_stage` and `simplerca`. The definitions are in
`evaluation/controls.py`.

## Original runs' records

Every record of `results/<suite>/<method>.jsonl` holds the case (`suite`, `system`, `case`, `service`, `fault`,
`instance`), the `method`, a `status` and, when completed, `service_result`: the rank of the labelled service,
Top-1/3/5 hits, Avg@5, the number of ranked services and whether the labelled service was among them. Ranker
records add the windows, the candidate nodes, the first 10 services and 15 nodes of the ranking with their scores
and series counts, and the usable series per node; comparator records add the raw ranking (`raw_ranks`), the
services derived from it, the windows and timings (`function_seconds` is the method call alone,
`process_seconds` includes start-up and imports). A failed, timed-out or invalid run scores 0 and stays in the
denominator.

Comparator adapters, recorded in each record: RCAEval's version-routed imports are bypassed by setting
`RCAEval.utility.is_py312` to return True; CIRCA's `@rca` wrapper is bypassed so that an exception is recorded as
a failure rather than replaced by the unranked column order (`called_undecorated_inner_function`), and its unused
statsmodels imports are satisfied by an import-only stub (`statsmodels_import_stub`). A later RCAEval fix
(29 September 2026) concerns that wrapper and a duplicated `time` column; no record here contains either.

The original records were computed under Python 3.11. A rerun under Python 3.13 with current numpy, pandas,
scikit-learn and scipy reproduced them: the decoded inputs were byte-identical; the rankers' records were
identical on all 717 cases; the TORAI severity stage on 465 of 465; N-Sigma, BARO and the random order on every
case of RE1, RE3, H1 and H2; RCD and ε-Diagnosis on seven sampled cases. For CIRCA on seven sampled cases, the rank
of the labelled service and every scored measure were identical, but in four of them the raw ranking
(`raw_ranks`) ordered near-tied metrics differently, an effect of the numerical library versions.

## License

Code: MIT (`LICENSE`). Results in `results/`: CC BY 4.0. The benchmark data belong to their publishers and are
not included.
