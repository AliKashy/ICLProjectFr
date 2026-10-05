# Part 5. Pretraining the encoder (Phase 5), from zero

Original chapter author: CL. Public adaptation: CH, in my first-person project voice.
The original author incorporated an independent CH review on 2026-10-05; this adaptation is not a new joint endorsement.

Status note: experiment status is the recorded 2026-10-04 snapshot; audit and proposed-fix statements describe the 2026-10-05 review. They are not live job or deployment status.
Source paths and commands are illustrative aliases, not files or runnable commands supplied by this public guide. `my_work/` means my implementation; `ya_work/` means Ya’s foundation; `s3://<bank-storage>/` replaces the actual storage address.

Phase 5 is written, has two main objective defects with proposed repairs (not yet applied to the project code),
and has no result documented in this repository snapshot. I explain what it is for and how I would test it.

---

## 5.1 The idea

Fraud labels are rare (1 in 600) and arrive late. Transactions themselves are plentiful. **Pretraining**
trains the encoder on a first task before the final fraud classifier is fitted (**fine-tuning**). Of the two
recipes here, the masked autoencoder is self-supervised and ignores fraud labels; SupCon is supervised
pretraining and uses the fraud labels to decide which transactions should look alike. (Both scripts still read
the label column through the shared loader, so neither runs on an unlabeled archive as written.) The analogy
for the self-supervised recipe: learn the language first, then study for the exam. In text and vision this is
the standard recipe; on tabular data the evidence is mixed, which is why Phase 5 is framed as an experiment,
not a plan.

The object being pretrained is the **encoder**: the tokenizer plus transformer that turns one transaction into
a vector (`my_work/phase5_pretrain/encoder.py`, `TabularEncoder`). After pretraining, a 2-way classification head
is attached and the whole thing is fine-tuned on fraud labels (`ssl_finetune.py`). Both pretraining recipes
produce the same kind of encoder, so they can be compared head to head against training from scratch.

If asked "why pretrain?": labels are rare and late, transactions are abundant; a pretrained encoder may need
fewer labels to reach the same ranking quality, and it gives an embedding other tools can use. The label-free
argument applies to the masked autoencoder; SupCon needs the labels.

---

## 5.2 Recipe 1: the masked autoencoder (self-supervised)

**Self-supervised** means the training signal comes from the data itself, no labels. Hide some of a
transaction's fields and ask the model to reconstruct them from the rest: hide the amount and the hour, show
the merchant category and the time of day, predict the amount and the hour. To do that well the model must
learn how fields relate (small amounts at grocery merchants, large at electronics), which is the kind of
knowledge a fraud classifier can reuse.

In `ssl_pretrain.py`: 30% of a transaction's tokens are replaced by a learned "mask" token; the encoder runs;
for each hidden continuous field a small head predicts the number (squared-error loss) and for each hidden
categorical field a head predicts the category (cross-entropy). Early stopping is on validation
reconstruction loss. Then `ssl_finetune.py` loads the encoder weights (`load_pretrained`) and trains the
classifier; `freeze_encoder: true` trains only the head while holding the encoder weights fixed, a cheap
approximation of a "linear probe". As written the encoder's dropout stays active during that training; the
standard deterministic linear-probe baseline needs the encoder put in evaluation mode.

The audit found a design flaw: because the mask token is the same for every field and the numeric decoder is
shared, two hidden numeric fields in one transaction get identical predictions (there is nothing left to say
which field is which). Fix: keep a learned per-field identity when masking, or use one decoder per numeric
field; this fix is designed, not yet written. Until then the objective cannot tell simultaneously hidden
numeric fields apart, which blunts what it is meant to teach; other learning can still happen.

If asked "what is the masked autoencoder?": hide 30% of fields, reconstruct them from the rest, no labels;
the encoder learns field relationships; a classifier is then fine-tuned on top.

---

## 5.3 Recipe 2: supervised contrastive learning (SupCon)

**Contrastive** learning teaches an encoder by comparison: pull representations of things that should be alike
together, push the rest apart. The **supervised** version (Khosla et al., 2020) defines "alike" by label: two
fraud transactions should land near each other, far from legitimate ones. For a rare class this is attractive
because every fraud example is pulled toward every other fraud example in the batch, concentrating the rare
class in one region of the representation space.

In `supcon.py`: each transaction is corrupted twice (30% of its fields swapped for another random row's values,
the SCARF recipe), both views are encoded and projected to 128 numbers, and the SupCon loss, with temperature
0.07, is applied over the batch using fraud labels as the grouping. Then the same fine-tuner as Recipe 1
trains the classifier.

The audit found that the loss attached the labels to the wrong rows (the views were laid out sample by sample
while the label mask assumed view by view), so it would have pulled the wrong pairs together. A one-line fix is
proposed; it is tested on synthetic data, matches the reference formulation, and CH checked it
independently. It is not yet applied to the project code.

A design question the Phase 5 guide raises itself: at 0.165% fraud, a batch of 8,192 holds about 13 fraud
transactions. The guide proposes a balanced sampler (keep all fraud, downsample legitimate rows). Note that
after the fix each transaction's two views already give it one positive partner, so the sampler is an
experiment about learning efficiency, not a requirement for the loss to work.

If asked "what is SupCon?": a loss that pulls same-label representations together and pushes others apart,
using two augmented views per transaction; attractive for rare classes; a bug in the label alignment has a
proposed, independently checked fix.

---

## 5.4 What a fair Phase 5 experiment looks like

Three encoders, identical architecture, data, preprocessing, validation files and evaluation cohort:

1. **From scratch**: train the classifier directly.
2. **Masked-autoencoder pretrained** (after the field-identity fix), then fine-tuned.
3. **SupCon pretrained** (after the alignment fix), then fine-tuned.

Report validation PR-AUC with 3 seeds each, plus the linear-probe variant for the two pretrained encoders.
Only then compare. Two mismatches exist in the current configs and must be fixed first: the SSL config
validates on one file (4950) while SupCon validates on five (4950 to 4954); both evaluate on one file (8000).

Also note what Phase 5 is **not** yet: it pretrains a single-transaction encoder of width 64 and 4 blocks,
while the Phase 4 history models use width 48; no Phase 4 script loads a pretrained encoder. The natural
connection is Architecture B, whose stage-1 transaction encoder could be initialized from a pretrained
encoder of matching shape. Matching width alone is not enough: B defaults to 2 transaction blocks against Phase
5's 4, and the history encoder has the extra time-gap channel, so checkpoint keys and shapes would not line
up without a deliberate mapping. That is a future experiment.

If asked "has pretraining helped?": no result is documented in the repository, and the snapshot cannot tell
whether a bank-side run happened; two objective defects were found first and have proposed repairs; the fair
test is pretrained vs from-scratch on identical architecture, data and cohort with seeds.

---

## 5.5 Vocabulary from this part

- **Encoder**: the part of the model that turns a transaction into a vector.
- **Pretraining / fine-tuning**: learn general structure first, then the labeled task.
- **Self-supervised**: training signal from the data itself (reconstruction), no labels.
- **Masked autoencoder**: hide fields, reconstruct them.
- **Contrastive / SupCon**: learn by pulling alike things together; "alike" defined by label in SupCon.
- **Augmentation / view**: a corrupted copy of a transaction (here, 30% of fields swapped with another row).
- **Projection head**: a small network mapping the encoder vector into the space where the contrastive loss is
  computed; discarded after pretraining.
- **Linear probe**: train only a linear head on a frozen encoder; measures how good the representation is on
  its own (deterministic only if the encoder is in evaluation mode).
