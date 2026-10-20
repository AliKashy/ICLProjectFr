# TF — the questions that remain for a person (20 October 2026)

On 20 October I asked CL to make sure everything is right from beginning to end and to replace questions with scripts wherever the data can answer. CL wrote one read-only script that I run on the bank's machine; it measures the machine, the environment and the data files (counts, types and column names only: no transactions, no account numbers), and it settles most of what used to be asked of me. What it cannot settle is below. The data-owner questions can go in one message; a short answer to each is enough.

Following [the publication notes](PUBLICATION_NOTES.md), this page uses **Ya** for my collaborator and **the bank** for the institution.

## For the data owner

1. **Same-second transactions.** When two transactions of one account arrive in the same second, does the fraud decision for the second one already see the first? (The models today never use a same-second transaction as history; about 2% of transactions and 6.7% of training frauds sit in such pairs.)
2. **The timestamp.** Is the transaction time the moment the authorisation request arrived (not the decision or the posting), and is it stored in UTC or in a local time zone, which one?
3. **Five fields and the decision.** For the external fraud score, the PIN check, the CVV check, the chip-cryptogram check and the address check: is the stored value the one present when the authorisation was decided, or can it be written by that decision or later? And please send the code lists for these five fields and for the card-present indicator (which value means the card was present).
4. **Labels.** On what dates were the training and evaluation extracts taken, and after how many days is a fraud tag considered final? Were the June 2025 labels final at extraction?
5. **Advice rows.** About one row in twenty carries a transaction-advice code (Ya's early look at two files; the script will measure it). Are those real-time authorisations or later notices about transactions the network decided for the bank, and can the same purchase appear as two rows?
6. **Declines.** Does the extract contain declined authorisations, and can a declined attempt carry the fraud tag?
7. **One limit field, and the currency.** What does the field described as "available daily merchandise limit" hold (unit; is it the value before the transaction?), and is the amount always in US dollars? Which type or category codes mark refunds, reversals and declines? If the script finds a currency or response column, this shrinks to confirming it.
8. **Merchant location.** On in-person transactions, is the merchant postal code the store's address or the merchant's registered address? (Needed only for the proposed distance features.)

## For me

- (a) What the bank will receive and use: the fraud score alone, the model's per-transaction vector as an input to the existing model, or both; batch or real time; and who supplies an account's earlier transactions at decision time.
- (b) Roughly how many alerts per day the fraud team can review, and whether the number or the dollar value of frauds caught matters more (question 9 of the 5 October list).
- (c) Whether a public postal-code-to-coordinates file may go on the bank's machine (for the distance features).
- (d) Rows 2-123 of the existing model's feature workbook and its field definitions, when convenient.
- (e) Whether Ya or I ever scored evaluation files beyond the eleven used so far (the script lists the prediction folders; my word closes it), and whether Ya's run folders for the ten-file history-5 model still exist.
- (f) The check file from the sliding-scoring run's folder, so its tolerance can be read before 0.093420 is quoted as the figure of record.
- (g) For the benchmark report: which population the existing models were trained on, and the bank's definitions of detection rate, value detection rate and false-positive rate at its operating point.

## Answered or overtaken (no longer asked)

Measured in the first bank run: the machine's Python environment and whether packages can be installed; the precision of the timestamp (whole seconds), the label values (exactly two, none missing) and the months each extract covers. Answered by my word: the sliding scoring ran (0.093420 against 0.092670). Superseded by the completed version 1 runs: the pre-version-1 run-order questions. To be measured by the script when I run it, so no longer asked of anyone: the postal-code formats, the column list, the account id's type, the machine's memory, the benchmark file's columns, and whether my version 1 preprocessing artifact is a byte copy of Ya's.
