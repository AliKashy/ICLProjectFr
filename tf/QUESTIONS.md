# TF — my open questions

Prepared 2026-10-05. I can answer by question number. Questions 1–6 are public adaptations of the shared review questions; 7–10 are additional CH follow-ups. My answers will help CH and CL update their separate guides. Until answered, these items remain unknown.

I retain approved figures here. I only need short answers or relevant metric summaries; raw transaction records and full internal logs are unnecessary.

1. **Which evaluation cohort did I actually score?** My checked-in configs select file 8000 (290 documented fraud cases), while the standing document describes files 8000–8010 (3,818 fraud cases). Which range produced each recorded evaluation AP? I need the range, row count and positive count from the scoring summary.

   **My answer:** Pending.

2. **Are My preprocessing artifacts identical to Ya’s?** Did I copy Ya’s version 1 byte for byte, or rebuild the medians, scaling statistics and category vocabularies? If rebuilt, I need that difference recorded before interpreting the comparison.

   **My answer:** Pending.

3. **How will I apply the reviewed fixes?** Which candidate fixes, if any, have I already applied in the bank environment? Will I apply the remaining fixes there, or have CH/CL prepare a separate reviewed source change? How will I coordinate shared-pipeline changes with Ya? Publishing these documents does not apply a code fix or contact anyone.

   **My answer:** Pending.

4. **Do I still have each run’s outputs?** Are the result tables, training metrics, training/scoring logs, predictions and best checkpoints retained separately? I need enough information to associate each score with its configuration, preprocessing version and evaluation cohort.

   **My answer:** Pending.

5. **What GPU budget can I use?** Approximately how many H200 hours per week are available? That determines how many model settings and repeated training seeds I can afford.

   **My answer:** Pending.

6. **Did my hierarchical sweep finish?** The last recorded snapshot says Architecture B started on 2026-10-04 at 23:09, pod-log time. Do I now have completed h5, h10 or h16 results, with their validation and evaluation AP?

   **My answer:** Pending.

7. **Has my h5 / 100-training-file model been evaluated?** Its recorded validation AP is 0.141585; evaluation scoring was still pending in the snapshot. If finished, what are the evaluation AP and exact cohort?

   **My answer:** Pending.

8. **Can I compare with the bank’s existing model on the same transactions?** Do I have its scores on the same transaction IDs and mature fraud labels? This is needed for a matched challenger comparison.

   **My answer:** Pending.

9. **What business outcome should I optimize?** What is the actual review capacity or alert budget, and should I prioritize fraud count caught, fraud value caught, fewer false alerts, or another agreed measure? The guides’ top-0.1% example is illustrative.

   **My answer:** Pending.

10. **What timing and history guarantees are confirmed?** Do my file ranges form a genuine later-time holdout with mature labels? Is each account’s relevant history contained in one file, and is timestamp-plus-ID ordering available at scoring time, including equal-time transactions?

    **My answer:** Pending.

I have already approved publication of the exact dataset counts, fraud rates, experiment scores and internal model details in these documentation copies. That question is resolved.
