# TF — CL's questions for me (2026-10-19)

CL (the Claude main worker) prepared these on 2026-10-19. I can answer by number, and a short answer is enough. Each question gives the options and CL's recommendation, so "go with CL" is a valid answer. Until I answer, CL follows the recommendation only where it says so, and nothing runs on the pod.

My answers go back to CL in chat or here. CL records them word for word in the private project record, so CH sees them too and I am never asked twice.

Following [the publication notes](PUBLICATION_NOTES.md), this page uses **Ya** for my collaborator and **the bank** for the institution. Private paths, storage locations and people's names are left out.

---

## A. Team set-up

1. **ChatGPT side of the two new pairs.** CL opened the two Claude assist chats: research (RS-CL) and audit (AU-CL). Their charters and two ready-to-paste opening messages for the ChatGPT side are in the private repository under `bridge/assist/`. The options:
   - (a) Open two new ChatGPT chats, one for research and one for audit.
   - (b) Open one new ChatGPT research chat. Turn CH's existing independent auditor chat into the ChatGPT audit member: it keeps its completeness checks, and modelling validity comes first.

   **CL recommends (b):** one chat fewer, and that auditor already knows the project.

   **My answer:** Pending.

2. **The plain-English explainer.** It kept the plain-English project guide (11 chapters) current with a timer every 2 hours. That timer has been off since 17 October, and CL has now retired the session. The options:
   - (a) Leave it retired.
   - (b) Bring it back on demand only, when I ask for the guide to be updated. No timer.

   **CL recommends (b).**

   **My answer:** Pending.

3. **GitHub Copilot's role from now on.** The 14–15 October work came from Copilot on my work PC: Ya's audit, the sliding package and the V2 plan. My 19 October rule says CL and CH write all the code. The options:
   - (a) Copilot only helps on the bank side: copying, installing, running and checking.
   - (b) Copilot may still write code and plans, which CL and CH then review.

   **CL recommends (a).** Every line of code then has one owner, and nothing is written twice.

   **My answer:** Pending.

## B. The sliding experiment and the next runs

4. **Has the history-5 sliding run started on the pod?** Nothing on GitHub shows that it has. To check, from the project folder on the pod (read-only):

   ```bash
   ls -la waqas/scoring/allpos_sliding_h5/ && cat waqas/scoring/allpos_sliding_h5/LATEST_RUN
   ```

   "No such file or directory" means it never ran. If a run exists, I list that run folder and send the listing.

   **My answer:** Pending.

5. **Do my original sliding files match the runbook?** The copies on GitHub were rebuilt from photos, and neither matches the SHA-256 in the runbook. The runbook expects `45cdb1da…` for the script and `9a0fb794…` for the test file. Two checks on my PC:
   - Do the originals hash to those values?
   - In the original script's `inspect` step, is the block that creates the run folder indented under `if args.create_run:`? In the GitHub copy it is not, so even the "read-only" `inspect` creates a run folder.

   **CL recommends** running the originals. If the `inspect` problem is also in the original, CL sends a one-line fix before the first run. It must come before, because the script records its own hash in each run.

   **My answer:** Pending.

6. **Is the lean V2 plan of 16 October (stages L1–L9) now the plan?** It would replace CH's 14 October improvement pathway, the package-006 data-scaling runs and the "sweep first" order. The options:
   - (a) Yes.
   - (b) Yes, with the feature stage from question 7 added.
   - (c) No; tell CL what to keep from the older plans.

   **CL recommends (b).**

   **My answer:** Pending.

7. **When do the V2 features come in?** The lean plan has no stage for new engineered features. Examples are recent-transaction counts, first visit to a merchant, and amount compared with the account's own past. The options:
   - (a) Add a feature stage after the history-20 comparison (L3). It tests a first wave of 5–8 features that exist at decision time: a cheap single-transaction screen first, then the chosen model.
   - (b) Add it after the final encoder (L7).
   - (c) Leave features for V3.

   **CL recommends (a).** Features are what the incumbent model relies on. The research pair is preparing the first-wave list.

   **My answer:** Pending.

8. **The incumbent model's feature importances.** On 13 October I said I would add the incumbent XGBoost model's feature importances and its full feature list. The full list is in my feature workbook. Are the "rank" values in its master sheet those importances? If not, when can I add the importances?

   **My answer:** Pending.

9. **GPU budget for V2.** The lean plan starts with about 5 training runs and adds up to about 12 more if the early results call for them. A history-20 run took many hours in V1. How many GPUs, and roughly how many GPU-hours a week, can I use now?

   **My answer:** Pending.

## C. Ya's higher history-5 score

10. **Does Ya's 10-file, history-5 checkpoint still exist?** His flat history-5 model scored AP 0.0871 on the same evaluation files where V1's scored 0.0728. If his trained model file still exists in his pod folder, CL can re-score it with our evaluation code, using no training and minutes of GPU time. That shows whether the gap comes from his training or from his history rule: his code let transactions in the same second see each other, and ours does not. The options:
    - (a) Yes, it exists (give the folder name).
    - (b) No.
    - (c) I don't know; I will look.

    **My answer:** Pending.

## D. Data facts only the data owner can confirm

These decide whether a good result would also work in production. Questions 11 to 15 can go to the data owner in one message.

11. **Same-second transactions.** When two transactions of one account share a timestamp, does the bank's fraud decision for the second one know about the first? A "no" means models must never use same-second history. V1 already follows that rule; Ya's runs did not.

    **My answer:** Pending.

12. **Fields that might be filled in after the decision.** The external fraud score input and the PIN, CVV, chip-cryptogram and address-check fields: is each one set before the bank's fraud decision on that transaction, or can it be set by the decision or afterwards?

    **My answer:** Pending.

13. **Label maturity.** How long after a transaction is its fraud label final? Were the June 2025 labels final when the data was extracted?

    **My answer:** Pending.

14. **Advice records.** About one row in twenty carries a transaction-advice code. Are those real-time decisions, or notices that arrive later? Can one transaction appear twice?

    **My answer:** Pending.

15. **The timestamp.** What moment does the transaction time record: request arrival, decision, or posting? Is it UTC or local time?

    **My answer:** Pending.

## E. Closing V1

16. **The rest of the V1 record.** The final 15-run report and a few pod folders never reached GitHub:
    - the review bundle;
    - the paired-report output folders;
    - the history-15 and history-20 retrain and manual-scoring folders;
    - the commands used.

    Uploading them is read-only and takes CPU minutes. The options:
    - (a) I upload them, and CL confirms the two hand-scored results.
    - (b) Close V1 as it stands, with those two results marked provisional.

    **CL recommends (a).** The best two V1 results are the hand-scored ones.

    **My answer:** Pending.

## F. Housekeeping

17. **Old open questions.** The private question list still holds about 14 open items from before V1 ran. The options:
    - (a) CL closes them as superseded, keeping only the data-owner questions, which are now questions 11–15 here.
    - (b) Go through them one by one.

    **CL recommends (a).**

    **My answer:** Pending.

18. **The 14 October README and folder names.**
    - The 14 October README rebuild now covers every line (1–849, with a blank line 850) as partial files. Is the assembled `readme.md` still to come, or can CL work from the partial files?
    - The updates now sit in two folders whose names differ only by a space or an underscore after "updates". Which name should both use?

    **My answer:** Pending.
