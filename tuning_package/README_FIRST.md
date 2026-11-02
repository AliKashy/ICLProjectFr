# READ THIS FIRST

**Warning.** This package was written by the reviewers (Claude and ChatGPT) from their records. **The reviewers do not know the pod's current state.** The owner has since reorganized the pod by hand. Every path, folder name, file name, trainer location and config location in this package may be out of date.

Copilot, before you run or change anything:
1. **Find the real thing.** For every path, file or command this package names, find the actual one on the laptop and the pod. Do not assume ours is right.
2. **Adapt.** Where ours is wrong, change the command or the path, never the model, training or scoring logic, and list each change in your report.
3. **Stop and ask the owner** if something this package needs cannot be found, or two different candidates exist.
4. **Keep the rules as written.** The runs, learning rates, epoch counts, seeds, populations and decision rules are the plan. Paths are yours to fix; the experiment is not.

## What is in this package

| Item | What it is | Status |
|---|---|---|
| `PLAN.md` | The joint plan of ChatGPT and Claude: answers to the owner's questions, every tuning method and where it goes, which data does what, the stages in the owner's order with their decision rules, and four open questions for the owner | Final draft; ChatGPT is checking it |
| `tuning_run_1/` | Stage 0: the light learning-rate check of the three contenders, ready to run (`README.md`; `COPILOT.md` says how to adapt it to the pod) | Ready |
| `tuning_run_1/REORG_CORRECTIONS.md` | Four corrections to the reorganization plan | Apply now |
| `COPILOT_PROMPT.md` | What to paste into Copilot | Ready |
| `modules/` | Code for every method in the plan, each off by default with CPU tests: loss options, EMA, schedules and optimizer groups, PLR numerical embeddings, the hierarchical tuner, CPU analysis tools | **Part 2, still being built and checked; arrives as a second zip** |

## Order on the pod

1. Stage 0 now: `tuning_run_1`.
2. Then the reorganization reaches the pod, after stage 0's runs and resumes have finished.
3. Then stage 1 (reproduce), and only then stage 2 (tuning). Stage 2 needs part 2's code.
