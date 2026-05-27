# HH alignment analysis spec

Goal:
Add an offline analysis pipeline for hidden-state alignment \(h_o^\top h_u\) and cosine variants.

We do not want to modify the SFT training behavior.

Core questions:
1. Are negative hidden-state alignments rare?
2. Is hidden-state alignment less sparse than token-gradient alignment?
3. How does alignment vary across layers?

Implementation stages:
1. Add hidden-state extraction debug script.
2. Add token-level hidden record extraction.
3. Add pairwise metric utilities.
4. Add pair sampling utilities.
5. Add full analysis runner that saves CSV/JSON summaries.

Experimental stages:
1. Add .sh file in bash/, following the project setting in other .sh files.
   Read notes/hh_task/hh_exp_settings.md, and create the corresponding .sh file for each experiments.


Safety:
- Do not run training on oat0.
- Do not run expensive jobs.
- Do not download large models/datasets unless explicitly approved.
- Keep defaults tiny.
- Prefer minimal patches.
- Show git diff after each patch.

Position convention:
For causal LM, hidden state at position p is used to predict the next token. The script must make this explicit and print a small sanity table showing input token, label token, and selected hidden position.