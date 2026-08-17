# CMD QUEUE

Use one command at a time. Do not let the coding assistant redesign unrelated cells. If a validation target fails, stop at that command and diagnose before continuing.

## 001
Treat this notebook as a reference-reproduction notebook only. Preserve the existing data, feature manifest, row-to-text logic, ICL setup, model settings, and evaluation logic. Replace only operations that cannot scale locally with distributed equivalents, and validate each replacement before continuing.

## 002
Replace only the current distributed row-selection logic so it matches the reference pandas groupby(...).sample(frac=..., random_state=2026) per-group sample-size semantics. Do not use sampleBy; match pandas null-group behavior, use deterministic row selection, and do not change downstream logic.

Validate and stop on mismatch:
- train reference rows = 77,447
- OOT stage-1 rows = 1,236,156
- final OOT rows = 1,236
- final label counts = 1,064 / 172

## 003
Add a validation cell for the accepted distributed samples: total rows, per-label counts, number of groups, unique transaction IDs, duplicate IDs, and null counts in grouping keys. Do not mutate any dataframe.

## 004
Freeze the accepted train and final OOT cohorts by saving their transaction IDs plus grouping keys and label to local parquet files. Add code that reloads these IDs and joins them back to the full distributed data so future runs never resample.

## 005
Validate the existing ordered feature manifest only. Print feature count, unique count, duplicates, missing-from-train features, and extra requested columns. Stop if the required feature count is not 94 or any required feature is missing.

## 006
Rebuild the feature-description lookup from the existing metadata file using only the 94 required features. Preserve feature order. Print missing descriptions and stop if any required feature lacks a usable description.

## 007
Create a clean reference row-to-text function that matches the existing baseline format exactly. Do not optimize or compress it. Add a single-row preview and verify that all 94 features appear once and in the correct order.

## 008
Create or restore the reference example formatter, balanced-shot selector, prompt builder, and classifier helper without changing their behavior. Keep 10 examples from each class for the actual scoring run and preserve the current model settings.

## 009
Add a prompt-inspection cell for one target only. Print the number of demonstrations, target ID, prompt character count, estimated token count if a tokenizer is available, and the first/last sections of the prompt. Do not call the API yet.

## 010
Harden only the request wrapper: add HTTP status checks, bounded retries with backoff for transient failures, safe JSON parsing, and error logging. Do not change model, temperature, top-p, prompt text, shot selection, or output semantics.

## 011
Run a smoke test on 5 final-cohort rows only. Record raw response, parsed score, returned transaction ID, latency, retry count, and error if any. Stop if IDs do not match targets or scores are not numeric in [0,1].

## 012
Run the full frozen final cohort using the reference prompt and 20 balanced random demonstrations. Preserve raw responses and never silently drop failures. Save results incrementally so an interrupted run can resume.

## 013
Parse the full result set into target ID, returned ID, numeric score, parse status, API status, latency, and retry count. Flag ID mismatches and invalid scores. Do not compute metrics until coverage is reported.

## 014
Create the frozen reference benchmark report: total targets, successfully scored targets, coverage %, parse failures, API failures, ID mismatches, AUC, KS, PR-AUC, class counts, mean latency, and estimated prompt tokens. Save the report and scored rows with a version tag.

## 015
Freeze the reference benchmark. From this point forward, do not modify its cohort, feature manifest, baseline text format, or baseline metric files. New work must live in separate experiment cells/files and compare against the same frozen cohort.

# EXPERIMENT TRACK

## 101
Create an experiment runner that accepts: formatter version, prompt version, number of shots per class, shot-selection method, seed, model settings, and token budget. It must always use the frozen cohort and log the full configuration with each result.

## 102
Run a shot-count ablation with the same representation and prompt: zero-shot, 1+1, 5+5, and 10+10 demonstrations. Compare AUC, KS, PR-AUC, token count, latency, and coverage.

## 103
Create a prompt-engineering experiment while keeping the same frozen cohort, representation, and demonstrations. Compare the reference instruction prompt against a shorter task prompt and a stricter structured-output prompt. Change only prompt wording/output instructions.

## 104
Create a second row representation that defines the schema once and serializes each row as compact feature=value pairs. Keep the reference representation unchanged. Compare predictive metrics, prompt tokens, latency, and parse reliability.

## 105
Create a third representation that defines column order once and serializes demonstrations/targets as compact value arrays. Preserve labels outside the target row. Compare against the reference and key/value versions.

## 106
Add token accounting to every experiment: prompt characters, estimated input tokens, output tokens if available, number of demonstrations, number of features, and latency. Produce performance-per-1k-token and latency-per-target summaries.

## 107
Build a token-budget experiment. Under fixed context budgets, compare spending the budget on more features versus more demonstrations. Keep the final cohort fixed and report the best metric achieved at each budget.

## 108
Create a deterministic similarity-based demonstration selector using only the existing required feature values. Standardize/encode safely, avoid target-label leakage, retrieve balanced nearest examples from the frozen training pool, and compare with random balanced selection.

## 109
Create a diversity-aware retrieval variant: start from similar candidates, then choose a balanced set that avoids near-duplicate demonstrations. Compare random, similarity-only, and similarity+diversity selection.

## 110
Create a segment-aware retrieval variant that prefers examples from the same existing segment when enough examples exist, with deterministic fallback to the global pool. Compare against random and global similarity retrieval.

## 111
Create a temporal retrieval variant that prefers historically prior examples closest in time to the target while preventing future-data leakage. Compare against random and similarity retrieval.

## 112
Create an uncertainty/context-ensemble experiment: score each target using several independently selected valid demonstration contexts, then store mean score and score variance. Compare single-context versus ensemble performance and stability.

## 113
Analyze disagreement across experiment scores. Create cohorts for low/low, high/high, reference-high/new-low, and reference-low/new-high. Report label rate, sample size, and score distributions for each cohort.

## 114
Create a compact semantic representation where raw values are accompanied by safe relative descriptors only when derivable from training data without leakage, such as training-set percentile bins or recency buckets. Compare it with raw-value representations.

## 115
Create a feature-budget ablation using a defensible model-free ranking derived only from the training data and labels, computed without using the final cohort. Test 94, 50, 25, and 10 features while keeping the cohort fixed. Log the ranking method and prevent leakage.

## 116
Create prototype-based ICL: cluster or summarize the frozen training demonstration pool without using final-cohort labels, then use representative prototypes plus a small number of real examples. Compare tokens and predictive performance with raw-example ICL.

## 117
Create a graph-context research prototype only from relationships already available in approved data. Build compact target-neighborhood summaries rather than sending full graphs to the LLM. Keep this isolated from the frozen benchmark and prevent temporal leakage.

## 118
Create a graph-aware demonstration selector using graph-neighborhood similarity or shared-entity structure where available. Compare it with numeric similarity and random retrieval under the same shot/token budget.

## 119
Create a dynamic-routing simulation. Define a cheap first-pass confidence signal from currently available non-LLM fields/scores without assuming their meaning; verify provenance first. Simulate sending only ambiguous cases to ICL and report percentage routed, latency reduction, and cohort performance.

## 120
Create a calibration evaluation for every score-producing experiment: reliability bins, Brier score, calibration curve data, and calibration error. Do not recalibrate on the final cohort; if calibration is tested, fit it on a separate allowed split.

## 121
Create operating-point metrics suitable for review: recall at fixed false-positive rates, precision at fixed review rates, top-k capture, and score-distribution tables. Use only metrics supported by the available labels/data.

## 122
Create stability reporting by existing month/segment fields: AUC, KS, PR-AUC, coverage, latency, and score distributions per period/segment, with warnings for small cells.

## 123
Create a production-style experiment summary table with one row per version and columns for predictive metrics, coverage, average input tokens, latency, number of LLM calls, and implementation notes. Do not include claims not directly supported by measured results.

## 124
Create an isolated benchmark harness for any approved tabular foundation model available in the environment. It must use the same frozen final cohort and report the same metrics. Do not install or download unapproved packages/models automatically.

## 125
Create a final architecture simulation that separates: distributed feature retrieval, context selection, compact serialization, LLM scoring, parsing/retries, calibration, monitoring, and optional routing. Keep it implementation-neutral and base every performance claim on measured experiment results.

## 126
Generate a concise research summary from saved experiment artifacts only: reference benchmark, strongest improvement, token reduction, latency impact, reliability, limitations, and next experiments. Never invent missing numbers.
