# ICML GPU Computational Cost Analysis: Figure Captions

## Figure 1: Sample Size Validation Across Mouse and Fly
**File path:** reports/edit_distributions/figures/figure1_sample_validation.png

**Sample Size Validation Across Mouse and Fly.** Comparison of n=100 vs n=500 samples reveals <7% variation in key metrics across three panels. Panel (A) shows mean edits per neuron: Mouse decreases 421.75→411.14 (-2.5%), Fly increases 17.84→19.02 (+6.6%). Panel (B) displays extrapolated full-dataset totals: Mouse 976K→951K, Fly 2.48M→2.65M. Panel (C) shows heavy-tail edit contribution: Mouse 10.6%→14.1%, Fly 37.6%→36.4%. Mouse demonstrates greater statistical consistency (2.5% mean difference, 3% extrapolation difference), validating linear extrapolation assumptions for full-dataset projections. Fly samples diverge slightly more (~6.6% mean, ~6.5% extrapolation), likely due to larger dataset size and higher variance. Error bars represent 95% bootstrap confidence intervals.

---

## Figure 2: Edit Distribution Patterns and Heavy-Tail Analysis
**File path:** reports/edit_distributions/figures/figure2_distribution_patterns.png

**Edit Distribution Patterns Across Species.** A 2×2 grid revealing striking differences in proofreading workload between species. Panels (A–B) show distribution histograms: Mouse (n=500) exhibits right-skewed distribution centered at μ=411±288 edits/neuron (median=335, range 0–1,714), while Fly (n=1000) shows extreme right-skew with log-scale y-axis and μ=17.5±32 edits/neuron (median=8, range 0–388). This represents a 23.6× intensity difference in per-neuron proofreading effort. Panels (C–D) reveal divergent heavy-tail patterns: Mouse heavy-tail neurons above the 95th percentile threshold (>964 edits, n=24, 4.8% of sample) contribute ~10.6% of total edits, indicating relatively uniform distribution of effort. By contrast, Fly heavy-tail neurons (>58 edits at 95th percentile, n=47, 4.7% of sample) concentrate 33.3% of total edits, indicating systematic undersegmentation affecting a concentrated set of neurons. These patterns reflect species-specific segmentation challenges and have direct implications for GPU resource allocation during AI-assisted proofreading.

---

## Figure 3: Computational Cost Landscape and GPU-Hour Breakdown
**File path:** reports/edit_distributions/figures/figure3_cost_sensitivity.png

**Computational Cost Landscape and GPU-Hour Breakdown.** Panel (A) presents a cost sensitivity heatmap showing projected computational costs across per-operation inference times (1.0–5.0 sec/operation). For Mouse, estimated costs range $600–$2,000; for Fly, $1,600–$5,000. The realistic operating range (2.0–2.5 sec/operation, highlighted in green) yields $600–$768 for Mouse and $1,568–$1,960 for Fly at $2/GPU-hour on dual H100 systems. Panel (B) shows GPU-hour stacked bar chart breakdown by operation type at the mean inference time estimate (t_avg=2.25 sec/operation). Mouse exhibits balanced cost distribution: 42% merge operations (614 GPU-hours), 58% split operations (614 GPU-hours), totaling 1,228 GPU-hours ($1,228). Fly is merge-dominated: 74% merge operations (1,568 GPU-hours), 26% split operations (526 GPU-hours), totaling 1,568 GPU-hours ($3,136). Fly is 2.5–3× more expensive than Mouse despite 23.6× lower per-neuron editing effort, due to its 60× larger dataset size. The divergent operation ratios reflect fundamentally different segmentation challenges: Mouse requires balanced correction across both undersegmentation and oversegmentation; Fly is dominated by undersegmentation requiring merge corrections.

---
