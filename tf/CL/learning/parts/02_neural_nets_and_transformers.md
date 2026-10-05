# Part 2. Neural networks and transformers from zero

Original chapter author: CL. Public adaptation: CH, in my first-person project voice.
The original author incorporated an independent CH review on 2026-10-05; this adaptation is not a new joint endorsement.

Status note: experiment status is the recorded 2026-10-04 snapshot; audit and proposed-fix statements describe the 2026-10-05 review. They are not live job or deployment status.
Source paths and commands are illustrative aliases, not files or runnable commands supplied by this public guide. `my_work/` means my implementation; `ya_work/` means Ya’s foundation; `s3://<bank-storage>/` replaces the actual storage address.

Part 1 said a model is a function with learned weights. This part opens the box for the particular kind of
function used here, one small idea at a time. Nothing beyond arithmetic is needed.

---

## 2.1 A neuron, a layer, a network

A **neuron** takes some input numbers, multiplies each by a weight, adds them up, adds a constant (the
**bias**), and passes the result through a simple bend such as "negative numbers become zero" (the
**activation**). That bend is what lets networks learn curves instead of straight lines. Ours uses a smooth
bend called GELU; the name does not matter.

A **layer** is many neurons reading the same inputs in parallel; its output is a list of numbers. A
**network** is layers stacked so each reads the previous one's output. "Deep learning" just means several
layers.

If asked "what activation?": GELU in the feed-forward blocks, standard for transformers.

---

## 2.2 Embeddings: turning a value into a vector

A network only eats lists of numbers of a fixed length. A transaction type code like `"P"` or a category
like "Tuesday" is not a number, and an amount like 21.71 is one number where the network wants many.
**Embeddings** solve both.

- For a **categorical** feature, keep a table with one row per possible value, each row a list of `d` learned
  numbers (`d` is 48 in the history models). Looking up the row for "Tuesday" gives Tuesday's vector. The
  numbers in the table are weights, learned like any other, so the model decides for itself that, say,
  Saturday and Sunday should end up with similar vectors if that helps predict fraud.
- For a **continuous** feature, first standardize it (subtract the training mean, divide by the training
  standard deviation, so most values land between about -3 and 3), then multiply the single number by a
  learned vector of length `d` and add a learned bias vector. This is the **linear** embedding. The project
  also tried a **periodic** embedding, which first expands the number into sines and cosines at 16 learned
  frequencies and then projects to `d`; it can capture "wiggly" relationships. In the sweep, linear won.

The output of embedding one transaction is a set of `d`-dimensional vectors, one per feature. Each such vector
is a **token**. One transaction here becomes **30 tokens**: 7 continuous (the 6 numeric features plus a
time-gap feature in the history model) and 23 categorical (13 codes, 4 calendar fields, and 6 "was this
numeric value missing" flags).

If asked "how are categoricals handled?": learned embedding tables with reserved indices 0 for unknown and
1 for missing; continuous features are standardized with frozen training statistics and linearly embedded.

---

## 2.3 Attention: every token looks at every other token

The transformer's one big idea. Given a set of tokens, **attention** lets each token build a new version of
itself as a weighted average of all the tokens, with weights it computes on the fly.

Mechanically, each token produces three vectors through three small learned matrices: a **query** ("what am I
looking for"), a **key** ("what do I contain"), and a **value** ("what I will contribute"). For token A, take
the dot product of A's query with every token's key; the better they match, the bigger the number. Pass those
numbers through a **softmax**, which turns them into positive weights that sum to 1. A's new vector is the
sum of every token's value times its weight.

What this buys us: the "amount" token can learn to look hard at the "merchant category" token and the "hour"
token when deciding what to pass on, and the strength of that look depends on the actual values. That is a
feature interaction learned from data, no human needed. Trees have to find such interactions by splitting;
attention computes them directly.

**Multi-head** attention runs several of these lookups in parallel with different matrices (8 heads here),
so one head can track "amount vs limit" while another tracks "hour vs card-present". A **block** is one
attention step followed by a small two-layer network applied to each token (the **feed-forward** part), with
two housekeeping tricks: a **residual connection** (add the input back to the output so information is never
lost) and **layer normalization** (rescale each token's numbers so training stays stable). The models here
stack 4 blocks (2 plus 4 in the hierarchical variant).

Attention cost grows with the **square** of the number of tokens, because every token looks at every other.
30 tokens is trivial; 511 tokens (history of 16) is 290 times more work per layer. That single fact drives
most of Part 4.

If asked "what is attention, in one sentence?": a learned, input-dependent weighted average over all tokens,
so the model decides at run time which features and which past transactions matter for this one.

---

## 2.4 The CLS token and the head

After the blocks, we have 30 updated token vectors and need one score. The trick, borrowed from BERT, is to
add one extra learned token called **CLS** (for "classification") before the blocks. It has no feature of its
own; through attention it absorbs whatever the other tokens offer. After the last block, we throw away all
tokens except CLS and feed its 48 numbers to the **head**: a single linear layer producing 2 numbers (the
**logits**, one for "legit", one for "fraud"). A softmax turns them into two probabilities that sum to 1; the
fraud probability is the score.

My explanation of producing one score from many tokens: a CLS token aggregates through attention; a linear head
on its final vector gives two logits; softmax gives the probability.

---

## 2.5 FT-Transformer: the architecture this project chose

**FT-Transformer** stands for Feature Tokenizer plus Transformer (Gorishniy, Rubachev, Khrulkov and Babenko,
"Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021). It is exactly Sections 2.2 to 2.4: embed
each feature as a token, add CLS, run transformer blocks, classify from CLS. Two properties make it a good
fit for tables:

1. Features are a **set**, not a sequence. Nothing says "amount comes before hour". FT-Transformer uses no
   position information inside a transaction, which is correct: each token already knows which feature it is
   from its own embedding.
2. It is the strongest general deep model for tables in that paper, roughly matching gradient-boosted trees
   across many datasets, while a plain multi-layer network (an **MLP**) trails.

Ya confirmed this on a public dataset before touching bank data (`ya_work/docs/tabular_transformer_models.md`):
on the UCI Bank Marketing set, FT-Transformer reached ROC-AUC 0.935 versus 0.921 for an MLP and 0.82 to 0.85
for a model called SAINT. That benchmark is a term-deposit prediction task, not fraud, and used no tuning, so
treat it as "a reasonable reason to start here", not as proof.

One precise point for a PhD audience: the single-transaction models (Ya’s Phases 1 to 3) use the
`pytorch_tabular` library's FT-Transformer implementation. The history models (Ya’s Phase 4 and all of my
work) are hand-built with PyTorch's standard `nn.TransformerEncoder` using the same tokenizer idea. They are
FT-Transformer in design, not the library's code, and differ in small internals (pre-norm vs post-norm,
GELU vs a gated activation, head width). The documentation calls both "FT-Transformer".

If asked "why FT-Transformer?": per-feature tokenization fits tabular data, it was the best deep tabular
model in the 2021 benchmark, it matched trees there, and it extends naturally to sequences of transactions,
which trees cannot do.

---

## 2.6 From one transaction to a history: tokens, positions, masks

To add history we embed the target transaction and each of its N previous transactions (same account,
preceding it in timestamp order with the transaction id as a tie-break; rows sharing the target's timestamp can
be included) the same way, giving `(N + 1) * 30` tokens, and let attention run over all of them plus CLS.
Three additions make this work:

- **Transaction position embedding.** Attention is order-blind, so without help the model cannot tell "this
  token belongs to the most recent transaction" from "this belongs to the oldest". A learned vector per slot
  (slot 0 = oldest, slot N = target) is added to every token of that slot. There is no position *within* a
  transaction, by the set argument above.
- **Time delta.** A continuous feature per transaction, `log(1 + seconds before the target)`, so the model knows
  whether the previous five transactions happened in one hour or one month. Velocity is a classic fraud signal.
- **Masks.** Accounts with fewer than N previous transactions get **padding**: empty slots on the left, and a
  mask telling attention to ignore them. A second kind of mask, **causal**, would stop a transaction's tokens
  from looking at later transactions; the default is **bidirectional** (everything sees everything), which is
  safe here because all inputs are legitimately known at scoring time and only the target's label is used.

Note on names: the plans talk about "SEP tokens" separating transactions. The code has none; the position
embeddings mark the boundaries. Both audits noted this as a naming discrepancy, not a defect.

If asked "how does the model know the order of transactions?": a learned per-slot position embedding plus an
explicit log time-gap feature.

---

## 2.7 What a GPU training run actually does

Put together, one run of the history model with N = 5 on 10 files:

1. A data loader reads parquet files from S3 one at a time, groups rows by account, sorts by time, builds one
   window per transaction (target plus 5 prior, left-padded), applies the frozen preprocessing, shuffles
   windows in a 65,536-window buffer, and hands over batches of 4,096 windows.
2. Each window is 6 slots of 30 tokens plus CLS: 181 tokens of 48 numbers. The GPU runs the forward pass for
   the 4,096 windows at once, computes the cross-entropy loss, runs the backward pass, and AdamW nudges the
   roughly 120,000 weights. Arithmetic is done in **bf16**, a 16-bit number format, to halve memory and
   speed things up; it is the standard choice for transformers, and its effect on these results has not been
   measured separately.
3. About 2,100 such steps make one epoch. After each epoch the model scores all validation windows (about
   4.4 million) and `valid_ap` is computed once over all of them.
4. Repeat for up to 10 epochs; stop when `valid_ap` has not improved for 5; keep the best checkpoint.
5. Score the evaluation files with the checkpoint and write a parquet of (id, label, score) to S3.

On an NVIDIA H200 this takes on the order of an hour for 10 files and several hours for 100.

If asked "what hardware and precision?": one NVIDIA H200, bf16 mixed precision, batch 4,096 windows,
gradient accumulation used for the long-history flat runs to keep the effective batch constant.

---

## 2.8 Capacity, and what "feature-limited" means

The number of weights is the model's **capacity**. More capacity can fit more complex patterns but needs more
data and overfits more easily. Ya’s Phase 3 finding was that making the single-transaction model bigger,
or training it longer, or giving it 10x more data, moved validation PR-AUC from about 0.11 only to about 0.125
and then flat. When capacity, epochs and data all fail to help, one natural reading is that the model has
extracted most of what the 23 columns contain: that it is **feature-limited**. The plateau motivates adding new
information (history) rather than a bigger model, but it does not prove the inputs are exhausted; optimization
settings, capacity choices and data coverage can also cause a plateau.

My answer on model capacity: yes at the single-transaction stage; it plateaued at about 0.125, which
is why Phase 4 adds history instead of capacity. A capacity sweep for the history model is still on the list.

---

## 2.9 Optimizer, learning rate schedule, regularization

**AdamW** is the standard optimizer for transformers: it adapts the step size per weight and applies **weight
decay** (a gentle pull of weights toward zero, 1e-5 here) to discourage overfitting. **ReduceLROnPlateau**
halves the learning rate when validation loss stops improving for 3 epochs. **Dropout** (randomly zeroing 10%
of activations during training) is another anti-overfitting trick used in the attention and feed-forward
parts. My sweep also tried **RAdam**, a variant; its gap to AdamW was tiny and is unresolved with one
seed. **EMA** (an
exponential moving average of the weights) exists in the code as an option but was never turned on.

If asked "what optimizer and schedule?": AdamW, learning rate 3e-4, weight decay 1e-5, ReduceLROnPlateau on
validation loss, dropout 0.1, early stopping on validation PR-AUC.
