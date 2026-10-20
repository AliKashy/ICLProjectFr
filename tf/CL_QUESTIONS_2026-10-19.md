# TF — CL's questions for me (2026-10-19)

CL (the main worker on its side) prepared these on 2026-10-19. I can answer by number, and a short answer is enough. Each question gives the options and CL's recommendation, so "go with CL" is a valid answer.

**Update, 20 October.** I asked CL to make everything right from beginning to end and to replace questions with scripts wherever the data can answer: CL writes a read-only script, I run it on the bank's machine and return what it prints. Read-only CPU scripts may run as soon as I upload them; training waits for my word. The questions below keep their history; the ones that remain for a person are collected in [one short list](DATA_OWNER_QUESTIONS_2026-10-20.md).

My answers go back to CL in chat or here. CL records them word for word in the private project record, so CH sees them too and I am never asked twice.

Following [the publication notes](PUBLICATION_NOTES.md), this page uses **Ya** for my collaborator and **the bank** for the institution. Private paths, storage locations and people's names are left out.

---

## A. Team set-up

1. **CH's side of the two new pairs.** CL opened the two assist chats on its side (research and audit); their charters and two ready-to-paste opening messages for CH's side are in my private record. The options:
   - (a) Open two new chats on CH's side, one for research and one for audit.
   - (b) Open one new research chat on CH's side. Turn CH's existing independent auditor chat into the audit member: it keeps its completeness checks, and modelling validity comes first.

   **CL recommends (b):** one chat fewer, and that auditor already knows the project.

   **My answer:** Delegated on 19 October (I asked CL to decide with the researchers, the plan managers and CH). Decided: (b), the existing independent auditor becomes the audit member on CH's side; I open one research chat there when convenient. Nothing waits on it.

2. **The plain-English explainer.** It kept the plain-English project guide (11 chapters) current with a timer every 2 hours. That timer has been off since 17 October, and CL has now retired the session. The options:
   - (a) Leave it retired.
   - (b) Bring it back on demand only, when I ask for the guide to be updated. No timer.

   **CL recommends (b).**

   **My answer:** Delegated on 19 October (I asked CL to decide with the researchers, the plan managers and CH). Decided: (b), on demand only, no timer.

3. **My local coding assistant's role from now on.** The 14–15 October work came from the coding assistant on my work PC: Ya's audit, the sliding package and the V2 plan. My 19 October rule says CL and CH write all the code. The options:
   - (a) It only helps on the bank side: copying, installing, running and checking.
   - (b) It may still write code and plans, which CL and CH then review.

   **CL recommends (a).** Every line of code then has one owner, and nothing is written twice.

   **My answer:** Delegated on 19 October (I asked CL to decide with the researchers, the plan managers and CH). Decided: (a), the local assistant helps on the bank side only (copying, installing, running, checking); CL and CH write the code.

## B. The sliding experiment and the next runs

4. **Has the history-5 sliding run started on the pod?** Nothing on GitHub shows that it has. To check, from the project folder on the pod (read-only):

   ```bash
   ls -la <my code folder>/scoring/allpos_sliding_h5/ && cat <my code folder>/scoring/allpos_sliding_h5/LATEST_RUN
   ```

   "No such file or directory" means it never ran. If a run exists, I list that run folder and send the listing.

   **My answer:** Answered in chat on 19 October: the answer is in the architecture record. Later the same day I reported that the sliding scoring did run: the all-position history-5 model scored 0.093420 on windows ending at each transaction, against 0.092670 when scored in chunks, on the same eleven evaluation files. CL treats the window-ending figure as the one of record for that model once I return the run's own check file (its tolerance must be small), and notes that it supports V2's design of scoring that way end to end.

5. **Do my original sliding files match the runbook?** The copies on GitHub were rebuilt from photos, and neither matches the SHA-256 in the runbook. The runbook expects `45cdb1da…` for the script and `9a0fb794…` for the test file. Two checks on my PC:
   - Do the originals hash to those values?
   - In the original script's `inspect` step, is the block that creates the run folder indented under `if args.create_run:`? In the GitHub copy it is not, so even the "read-only" `inspect` creates a run folder.

   **CL recommends** running the originals. If the `inspect` problem is also in the original, CL sends a one-line fix before the first run. It must come before, because the script records its own hash in each run.

   **My answer:** Answered in chat on 19 October: my local assistant checked the originals; the indentation and the hashes are correct, so the GitHub copies are reconstruction errors. CL withdrew the fix; the originals run unchanged.

6. **Is the lean V2 plan of 16 October (stages L1–L9) now the plan?** It would replace CH's 14 October improvement pathway, the package-006 data-scaling runs and the "sweep first" order. The options:
   - (a) Yes.
   - (b) Yes, with the feature stage from question 7 added.
   - (c) No; tell CL what to keep from the older plans.

   **CL recommends (b).**

   **My answer:** Answered in chat on 19 October: the lean plan is a proposal; I asked whether CL and CH agree. One GPU, so few runs: find the winning architecture and grow from there. Later the same day I asked for a proper test of the chunk model against normal B at the same history (5 and 20) before choosing, and delegated the remaining decisions. The governing plan (version 6) was agreed by CL and CH on 19 October; its decision rules are written down before any result is read. One rule's recorded risk still awaits CH's confirmation.

7. **When do the V2 features come in?** The lean plan has no stage for new engineered features. Examples are recent-transaction counts, first visit to a merchant, and amount compared with the account's own past. The options:
   - (a) Add a feature stage after the history-20 comparison (L3). It tests a first wave of 5–8 features that exist at decision time: a cheap single-transaction screen first, then the chosen model.
   - (b) Add it after the final encoder (L7).
   - (c) Leave features for V3.

   **CL recommends (a).** Features are what the incumbent model relies on. The research pair is preparing the first-wave list.

   **My answer:** Answered in chat on 19 October: whatever is best, so CL's (a).

8. **The incumbent model's feature importances.** On 13 October I said I would add the incumbent XGBoost model's feature importances and its full feature list. The full list is in my feature workbook. Are the "rank" values in its master sheet those importances? If not, when can I add the importances?

   **My answer:** Answered in chat on 19 October: yes, the ranks are the importances; the VAA score was the most important.

9. **GPU budget for V2.** The lean plan starts with about 5 training runs and adds up to about 12 more if the early results call for them. A history-20 run took many hours in V1. How many GPUs, and roughly how many GPU-hours a week, can I use now?

   **My answer:** Answered in chat on 19 October: 1 GPU.

## C. Ya's higher history-5 score

10. **Do Ya's 10-file, history-5 run records still exist?** His flat history-5 model is recorded at AP 0.0871, which I understand was on the same evaluation files where V1's scored 0.0728; his records would confirm this. His single-transaction model, though, was weaker than ours (0.0792 against 0.0925), so the gap is about history. The leading reason: his code let a transaction see other transactions of the same account in the same second as "history", and ours does not. Those rows are about 2% of transactions but 6.7% of training frauds. If they are not visible when a card is authorised, part of his 0.0871 cannot be reached in production. Three records settle it, cheapest first:
    - his training log and per-epoch scores file. They show how many files he trained on, the steps per epoch, which validation files he used and his best epoch. Reading them takes seconds and touches no transactions.
    - his evaluation prediction file. With it, CL compares his scores with V1's on the same transactions, split into those with and without an earlier transaction in the same second. CPU minutes, no GPU.
    - his trained model file. It is needed only if the prediction file is gone: CL re-scores it with no training, minutes of GPU time.

    The options:
    - (a) Yes (say which of the three, and the folder name).
    - (b) No.
    - (c) I don't know; I will look.

    **My answer:** Not yet. On 19 October I asked why V1's improved code scored lower at history 5 than Ya's different code; CL answered in chat (most likely Ya's same-second history rule, epoch choice and run-to-run noise). The first read-only script lists the prediction and run folders on the machine; my word on Ya's records closes it (item in the [20 October list](DATA_OWNER_QUESTIONS_2026-10-20.md)).

## D. Data facts only the data owner can confirm

These decide whether a good result would also work in production. Since 20 October the scripts come first: the read-only data script measures what the data can show (same-second counts, the hour-of-day and clock-change profiles, the code values, the fraud rate by week, the advice rows and their twins, the limit fields, declines and currency columns if any exist). The questions below remain for the data owner, in the plain wording of the [20 October list](DATA_OWNER_QUESTIONS_2026-10-20.md); they can go in one message.

11. **Same-second transactions.** When two transactions of one account share a timestamp, does the bank's fraud decision for the second one know about the first? A "no" means models must never use same-second history. V1 already follows that rule; Ya's runs did not.

    **My answer:** Pending.

12. **Fields that might be filled in after the decision.** The external fraud score input and the PIN, CVV, chip-cryptogram and address-check fields: is each one set before the bank's fraud decision on that transaction, or can it be set by the decision or afterwards?

    **My answer:** Pending.

13. **Label maturity.** How long after a transaction is its fraud label final? Were the June 2025 labels final when the data was extracted?

    **My answer:** Pending.

14. **Advice records.** About one row in twenty carries a transaction-advice code (Ya's early look at two files; the script will measure it). Are those real-time decisions, or notices that arrive later? Can one transaction appear twice?

    **My answer:** Pending.

15. **The timestamp.** What moment does the transaction time record: request arrival, decision, or posting? Is it UTC or local time?

    **My answer:** Pending.

20. **Which transaction types use which daily limit?** Which transaction types or methods count against the daily merchandise limit, and which against the daily cash limit? A "share of the limit used" feature must divide by the right one.

    **My answer:** Pending.

21. **Declined attempts and the fraud label.** Are declined authorisations in the files, and can a declined attempt be labelled fraud? If declined fraud is never labelled, "this would overdraw the account" can look like a sign of a genuine customer for the wrong reason.

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

    **My answer:** Answered in chat on 19 October: the results already uploaded should be enough. CL closes V1 with the two hand-scored runs labelled as scored separately.

## F. Housekeeping

17. **Old open questions.** The private question list still holds about 14 open items from before V1 ran. The options:
    - (a) CL closes them as superseded, keeping only the data-owner questions, which are now questions 11–15 here.
    - (b) Go through them one by one.

    **CL recommends (a).**

    **My answer:** Delegated on 19 October (I asked CL to decide with the researchers, the plan managers and CH). Decided: (a), CL closes the old items as superseded, keeping the data-owner questions (now 11-15, 20, 21).

18. **The 14 October README and folder names.**
    - The 14 October README rebuild now covers every line (1–849, with a blank line 850) as partial files. Is the assembled `readme.md` still to come, or can CL work from the partial files?
    - The updates now sit in two folders whose names differ only by a space or an underscore after "updates". Which name should both use?

    **My answer:** Pending. Meanwhile CL works from the partial README files as the README, and proposes the underscore form for both folders; I confirm or correct.

19. **May CL prepare one set of quick checks for the pod?** These are the cheapest checks that settle the open points. CL would put them in one script in the private repository; I would run it and send back what it prints.
    - **Read-only, CPU, a few minutes in total:**
      - (a) The history-5 B run's validation score at every epoch. Its best epoch was 0, the first pass of the data. This shows whether that run went wrong, which is most of the "0.06 to 0.09" story.
      - (b) Whether the training folder and the evaluation folder share accounts: the account column of one file number found in both folders, and of V1's own training, validation and evaluation files, compared as counts only. If they share accounts, the next data split must avoid that.
      - (c) The batch and validation settings recorded in each of the 15 finished runs, to confirm the record. The dry run also prints the installed pandas, numpy, pyarrow, torch, lightning and xgboost versions.
      - (e) A re-run of the existing comparison script on the saved V1 scores. It adds the healthy runs as references for the all-position model, on the same transactions. This is the most direct test of the "0.06 to 0.09" question. CPU minutes.
      - (g) Whether the balance and limit fields really hold the values from before each transaction, as the field list says. If they were recorded after the bank's decision, they would quietly tell the model whether the bank approved the transaction, a leak that would also break the new balance and limit features. Counts only, CPU minutes.
      - (f) Same-second transactions, for question 10. How many transactions, and frauds, have an earlier transaction of the same account at the same moment? For each such pair: is it an advice notice, a repeat of the same amount, or a separate purchase? And do the balance and limit fields show that the bank had already processed the earlier one? Counts only, CPU minutes.
    - **One short GPU job, no training:** (d) score the last saved model of two history-5 runs on the evaluation files:
      - the history-5 B run. If it scores much higher than its epoch-0 model, the "0.06" was a bad pick of epoch, not a weak design.
      - V1's flat history-5 run (best epoch 2 of 8). This shows whether a later epoch closes part of the gap to Ya's 0.0871 (question 10).

    The options:
    - (a) Yes, (a)–(g).
    - (b) Only the read-only and CPU checks (a), (b), (c), (e), (f) and (g).
    - (c) Not now.

    **CL recommends (a).** No training is involved, and it answers the "0.06 to 0.09" question directly.

    **My answer:** Delegated on 19 October (I asked CL to decide with the researchers, the plan managers and CH). Decided: (a), yes to all checks (a)-(g). The script is tested and reviewed; the exact commands are in the one end-to-end runbook CL wrote on 20 October, and the CPU checks run as soon as the package is on the machine; (d), the GPU check, waits for my word.
