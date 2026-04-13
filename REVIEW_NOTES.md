# REVIEW_NOTES — ConnectomeBench
Date: 2026-04-13
Iteration goal: First run — create RALPH.md by thoroughly understanding the project
Outcome: ✅ achieved

Work done:
- Deep-dived ConnectomeBench repo: 5 benchmark tasks, 4 species (mouse/fly/human/zebrafish), 10+ LLM evaluations
- Discovered `compute-cost-analysis` branch with complete ICML 2026 GPU cost artifacts (figures, LaTeX, data)
- Found related ConnectomeVLM project at `/Users/quileesimeon/ConnectomeVLM/` with full paper reproduction pipeline
- Confirmed ALL critical artifacts are intact on `compute-cost-analysis` branch (figures, JSONs, LaTeX)
- Verified Cursor history backup exists at `~/Library/Application Support/Cursor/User/History/-588e45f7/`
- Created RALPH.md with verified numbers table, artifact map, success criteria, branch structure

Blockers: None

Next iteration: Merge `compute-cost-analysis` into the ralph branch. Specifically:
1. `git merge compute-cost-analysis` into `ralph/ConnectomeBench`
2. Verify all 6 JSON data files are present in `reports/edit_distributions/`
3. Verify both figures exist: `figure_main_gpu_cost.png` and `figure_supplement_analysis.png`
4. Verify `reports/gpu_cost_section_compact.tex` is intact (210 lines, all \ref{} valid)
5. Run `scripts/generate_icml_cost_figures.py` to confirm figures regenerate from JSON data (without changing numbers)
6. Do NOT rerun `analysis/analyze_edit_distributions.py` — requires CAVEClient and data is already pre-computed

Completion: 15% (RALPH.md created; artifacts verified; compute-cost-analysis not yet merged to ralph branch)
