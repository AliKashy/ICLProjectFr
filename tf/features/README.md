# TF V2: proposed features

[Download the workbook ZIP](TF_V2_proposed_features.zip?raw=true). Extract it and open **TF_V2_proposed_features.xlsx**.

These are the extra inputs proposed for version 2 of my fraud model, with where each one comes from:

- **Proposed features** (29): what each measures, its formula, the exact raw data fields it is derived from, which transactions it uses, where the idea came from, leak risk, what must be confirmed first, priority, status and tier.
- **Built columns** (42): every column the current code produces, with the formula as built.
- **Raw fields** (33): each raw field the features use, its description, and whether it is already a model input.
- **Model inputs today**: the current 23 inputs (V2 drops month).
- **Dropped or merged** (30): earlier candidates that were cut, and why.

None of these features is in the model yet. Each must pass the CPU feature screen, then the GPU comparison, and its inputs must be confirmed available when the bank decides on a transaction. Built means the code exists and passes tests on synthetic data; no proposed feature has been tested on bank data.

Prepared 19 October 2026 by CL from the private feature catalogue and feature code; every derivation was checked against the code.

ZIP SHA-256: `f8aa339b6f064a954aba389124d03b00b7eb26de288ffe04df17a2370a9fc43c`. Workbook SHA-256: `d7d7d6f074c2adfe0f3490d7ae4240f4c37f6383efaa65adb5195128a5dd90e3`.
