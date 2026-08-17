Stop optimizing the sampling section for exact reference row counts. We only need a sensible, reproducible stratified cohort from the same train/OOT sources.

Use a fixed seed (2026), create roughly ~77K train rows and ~1.2K OOT rows while preserving segment/frd_tag/yyyymm representation, validate the distributions, then freeze the selected transaction IDs. Different exact rows/counts from the original implementation are acceptable.

After freezing, move on to the actual ICL experiments. Also make demonstration selection deterministic so prompt/model comparisons use the same evaluation cohort and reproducible examples.
