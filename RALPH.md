# RALPH.md — ConnectomeBench

## Project Goal
Benchmark for evaluating multimodal LLMs on connectome proofreading tasks (ConnectomeBench), with a GPU cost analysis section for the ICML 2026 submission on AI-based connectome proofreading (ConnectomeVLM).

## Deliverable Type
- **Primary**: ICML 2026 paper section (GPU Computational Cost Estimation)
- **Secondary**: Open-source benchmark release (ConnectomeBench v1.0 on HuggingFace)

## Audience
- Connectomics researchers evaluating AI proofreading systems
- ML/AI community interested in real-world scientific applications
- ICML 2026 reviewers

## Success Criteria
1. `compute-cost-analysis` branch merged to `main` with all figures/LaTeX/data intact
2. `reports/gpu_cost_section_compact.tex` compiles cleanly with correct figure references
3. Both paper figures render correctly: `figure_main_gpu_cost.png` (4-panel) and `figure_supplement_analysis.png` (7-panel)
4. All verified numbers remain unchanged throughout (see Key Numbers below)
5. ConnectomeBench benchmark is runnable from HuggingFace dataset without CAVEClient auth

## Design Philosophy
- **Never change verified numbers**: All GPU cost estimates were computed from actual connectome edit histories and independently validated. These are ground truth.
- **Reproducibility first**: Any figure must regenerate from the JSON source data files
- **Clean separation**: GPU cost section is self-contained within `reports/`

## Constraints
- Python 3.12+, uv package manager
- CAVEClient authentication needed for live data (tokens in `~/.secrets`)
- Figures must match ICML format (grayscale palette, high-res PNG + PDF)
- **DO NOT regenerate JSON data files** — they require CAVEClient server access which is intermittently unavailable

## Key Verified Numbers (CRITICAL — never change)
These come from actual connectome edit histories (mouse n=500 from 2,314; fly n=1,000 from 139,255):

| Metric | Mouse | Fly |
|--------|-------|-----|
| Mean edits/neuron | 411 ± 288 | 17.5 ± 32 |
| Median edits/neuron | 335 | 8 |
| Species ratio | 23.4× | — |
| Merge/Split ratio | 46.3% / 53.7% | 71.9% / 28.1% |
| Heavy-tail threshold (p95) | 971 edits | 58 edits |
| Heavy-tail neurons | 4.8% | 4.7% |
| Heavy-tail edit fraction | 14.1% | 33.3% |
| **Level 1 Naive cost** | $1,057 | $2,714 |
| **Level 2 Naive cost** | $34,262 | $2,714 |
| **Level 2 Realistic cost** | $33,628 | $3,040 |
| **Level 3 Naive cost** | $4,568,266 | $2,714 |
| GPU rate | $2/hour (dual H100) | — |
| Inference time (naive) | 2.0s uniform | — |
| Inference time (realistic) | merge=2.5s, split=1.5s | — |

## Current State: ~65%

### What Exists (on `compute-cost-analysis` branch)
- ✅ `reports/edit_distributions/*.json` — 6 pre-computed data files (mouse/fly × n=100/500/1000)
- ✅ `reports/edit_distributions/figures/figure_main_gpu_cost.png` — 3713×3065 RGBA PNG
- ✅ `reports/edit_distributions/figures/figure_supplement_analysis.png` — supplement figure
- ✅ `reports/gpu_cost_section_compact.tex` — 210-line complete LaTeX section
- ✅ `scripts/generate_icml_cost_figures.py` — 1054-line figure generation script
- ✅ `analysis/analyze_edit_distributions.py` — 654-line analysis pipeline

### Related Artifacts (in `/Users/quileesimeon/ConnectomeVLM/`)
- ✅ `output/benchmark_table.tex` — main benchmark results table (6 models × 4 tasks)
- ✅ `output/figure2_scaling_curves.png` — linear probe scaling curves
- ✅ Complete paper reproduction pipeline with pre-computed data

### What's Missing / Incomplete
- ❌ `compute-cost-analysis` NOT merged to `main`
- ❌ No PDF output of LaTeX section (need to compile)
- ❌ No integration between GPU cost section and main ConnectomeVLM paper
- ❌ `scripts/training_data/*.json` not in repo (hosted externally or gitignored)
- ❌ Analysis for Human (H01) and Zebrafish datasets (only 1 proofread H01 neuron available)

## Human Actions Needed
1. **CAVEClient auth**: Any new data gathering requires `~/.secrets` tokens for MICrONS/FlyWire
2. **Merge PR**: `git merge compute-cost-analysis` into main requires review/approval
3. **Paper submission**: Final LaTeX compilation and ICML submission needs human review
4. **HuggingFace dataset**: Uploading training data files requires HF credentials

## Codex Delegation Guide
- **Figure regeneration**: `codex:rescue` — pass `scripts/generate_icml_cost_figures.py` and the JSON data paths; regenerate both figures without changing any computed values
- **LaTeX polish**: `codex:rescue` — improve section prose, fix formatting, ensure all `\ref{}` are correct
- **New analysis scripts**: `codex:rescue` — implement additional statistical analyses using the existing JSON data
- **Test writing**: `codex:rescue` — write pytest tests for `analysis/analyze_edit_distributions.py`

## Key File Paths
```
reports/
  gpu_cost_section_compact.tex           # The ICML LaTeX section
  edit_distributions/
    *.json                               # Pre-computed data (DO NOT REGENERATE)
    figures/
      figure_main_gpu_cost.png           # Main 4-panel figure
      figure_supplement_analysis.png    # 7-panel supplement

scripts/
  generate_icml_cost_figures.py          # Regenerates figures from JSON data
  
analysis/
  analyze_edit_distributions.py         # Data pipeline (requires CAVEClient)

/Users/quileesimeon/ConnectomeVLM/      # Full paper reproduction package
  output/benchmark_table.tex            # Main benchmark results
  generate_all_figures.sh               # Reproduces all paper figures
```

## Branch Structure
- `main` — stable benchmark code (current ralph branch base)
- `compute-cost-analysis` — GPU cost section + figures + data (NOT YET MERGED)
- `ralph/ConnectomeBench` — this worktree (based on main)

## Next Agent Suggestion
**Iteration 2 Goal**: Merge `compute-cost-analysis` into the ralph branch, verify all artifacts are intact, and confirm figures regenerate cleanly from JSON data. Do not change any numbers.
