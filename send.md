One more step, nothing to redo. Read tab_fraud_reorg/EXPECTED_STATE.md. It is what ChatGPT and Claude think the pod looks like, with an ID on every item.

Compare it with everything you found, and write handoff/SEND.md with only the differences, following the "How to answer" section at the end of that file:
1. Changed items: one line per ID that is not as expected (moved to where, not found, different and how, or partly there). Matching items get no line. Then one line counting how many matched exactly.
2. New items: anything not on the list, at folder level, with size, file count and a one-line role.
3. Code differences: path, closest known version, which functions changed; a diff only if 10 lines or fewer.
4. Moves: old path to new path at folder level, same or different bytes.
5. Questions, each with why it matters.

Limits: at most 300 lines, each at most 90 characters, and a line --- page N of T --- every 40 lines. Tables at most 4 columns. No full source code.

Keep all full details in handoff/detail/ under the same IDs, so that when I paste a request like "show M16" you answer with just that item, kept short. When done, tell me how many pages SEND.md has.