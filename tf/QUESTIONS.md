# TF — my open questions

Prepared 2026-10-05; answer lines updated 2026-10-20. I can answer by question number. Questions 1–6 are public adaptations of the shared review questions; 7–10 are additional CH follow-ups. Most are now answered or overtaken by the completed version 1 runs; what remains for a person is in the [20 October list](DATA_OWNER_QUESTIONS_2026-10-20.md).

I retain approved figures here. I only need short answers or relevant metric summaries; raw transaction records and full internal logs are unnecessary.

1. **Which evaluation cohort did I actually score?** My checked-in configs select file 8000 (290 documented fraud cases), while the standing document describes files 8000–8010 (3,818 fraud cases). Which range produced each recorded evaluation AP? I need the range, row count and positive count from the scoring summary.

   **My answer:** Answered (12 October): the old sweep scored file 8000 only; the completed version 1 runs scored files 8000–8010 (2,466,461 rows, 3,818 fraud cases), with the range, row count and positive count in each run's scoring summary.

2. **Are My preprocessing artifacts identical to Ya’s?** Did I copy Ya’s version 1 byte for byte, or rebuild the medians, scaling statistics and category vocabularies? If rebuilt, I need that difference recorded before interpreting the comparison.

   **My answer:** Pending. The first read-only script compares the two artifact folders file by file if Ya's folder is readable from my machine; otherwise my word settles it. It affects only comparisons with Ya's historical scores: version 2 builds its preprocessing from my version 1 artifacts either way.

3. **How will I apply the reviewed fixes?** Which candidate fixes, if any, have I already applied in the bank environment? Will I apply the remaining fixes there, or have CH/CL prepare a separate reviewed source change? How will I coordinate shared-pipeline changes with Ya? Publishing these documents does not apply a code fix or contact anyone.

   **My answer:** Answered (8 and 11 October): Ya's files are never edited; corrected copies live under my own folder and reach the bank's machine through reviewed pull requests that I merge and install in place. CH and CL write the code; my local coding assistant helps on the bank side only.

4. **Do I still have each run’s outputs?** Are the result tables, training metrics, training/scoring logs, predictions and best checkpoints retained separately? I need enough information to associate each score with its configuration, preprocessing version and evaluation cohort.

   **My answer:** Answered (12 October): every version 1 run's folder (settings, training metrics, logs, checkpoints and scoring records) is kept on the bank's machine; the earlier, pre-version-1 results were deleted and everything was rerun.

5. **What GPU budget can I use?** Approximately how many H200 hours per week are available? That determines how many model settings and repeated training seeds I can afford.

   **My answer:** Answered (19 October): one GPU, no hour limit stated.

6. **Did my hierarchical sweep finish?** The last recorded snapshot says Architecture B started on 2026-10-04 at 23:09, pod-log time. Do I now have completed h5, h10 or h16 results, with their validation and evaluation AP?

   **My answer:** Overtaken: the sweep was rerun as version 1 (15 runs, 12–13 October); its results are in the version 1 record, and the history-5 hierarchical run is the one whose weakness version 2 investigates.

7. **Has my h5 / 100-training-file model been evaluated?** Its recorded validation AP is 0.141585; evaluation scoring was still pending in the snapshot. If finished, what are the evaluation AP and exact cohort?

   **My answer:** Overtaken: the h5 / 100-training-file model was not evaluated; version 2's final step trains the chosen recipe on a hundred files and evaluates it once.

8. **Can I compare with the bank’s existing model on the same transactions?** Do I have its scores on the same transaction IDs and mature fraud labels? This is needed for a matched challenger comparison.

   **My answer:** Pending in part: the bank's benchmark file with the incumbent models' scores exists on my machine; the first read-only script reads its columns. Which columns hold which scores, and what population the incumbent was trained on, is an item in the 20 October list.

9. **What business outcome should I optimize?** What is the actual review capacity or alert budget, and should I prioritize fraud count caught, fraud value caught, fewer false alerts, or another agreed measure? The guides’ top-0.1% example is illustrative.

   **My answer:** Pending (item in the 20 October list).

10. **What timing and history guarantees are confirmed?** Do my file ranges form a genuine later-time holdout with mature labels? Is each account’s relevant history contained in one file, and is timestamp-plus-ID ordering available at scoring time, including equal-time transactions?

    **My answer:** Partly measured: the evaluation files are a later period (May–June 2025) for accounts that appear in none of the ten version 1 training files (measured in the first bank run; the wider check of the two folders is check (b) of CL question 19, not yet run); each account's rows sit in one file; timestamps are whole seconds and the models use only strictly earlier transactions as history. Whether a same-second earlier transaction is visible at decision time, and when labels become final, are data-owner questions 1 and 4 of the 20 October list.

I have already approved publication of the exact dataset counts, fraud rates, experiment scores and internal model details in these documentation copies. That question is resolved.
