# IODA Strategy Breakdown: GPU Computational Cost Estimation

## 1. INVENT (Conceptual Design)
**What you thought about:**
- Why estimate computational costs? → "FlyWire needed 30 human-years; we need hardware cost projections"
- What problem are we solving? → "Scaling AI proofreading requires understanding workload distribution"
- Key hypothesis → "Edit distributions vary by species and exhibit heavy-tail patterns"

**Narrative Purpose:**
- Orients readers to *why* this analysis matters for the main ConnectomeAgent story
- Frames cost estimation as a natural follow-on question after demonstrating AI capability

---

## 2. OLD (Prior Work / Historical Context)
**What's been done before:**
- Public connectome datasets (MICrONS, FlyWire) exist with edit histories
- Qualitative understanding: "FlyWire took 30 human years" but no systematic cost model
- CAVEclient API enables programmatic access

**Why this matters:**
- Establishes that the data infrastructure exists
- Shows this is the first systematic statistical characterization
- Positions your work as the missing quantitative analysis

---

## 3. DISCOVER (Methodology & Results)
**What you tried and what worked:**

### Methods:
- ✅ Random sampling (n=100, n=500) — simple, robust, validates via cross-sample consistency
- ✅ Edit history retrieval via `get_tabular_change_log()` — straightforward API call
- ✅ Merge vs. split categorization — direct interpretation of `is_merge` column
- ✅ Heavy-tail analysis at 95th percentile — standard statistical practice

### Results:
- **Discovery 1:** 23.6× intensity difference between species (422 vs 19 edits/neuron)
  - Implies Mouse dominates computational cost despite smaller dataset
- **Discovery 2:** Different operation ratios (47/53 vs 74/26 merge/split)
  - Reveals species-specific segmentation challenges
- **Discovery 3:** Heavy-tail concentration varies dramatically (10.6% vs 36% of edits)
  - Shows Fly proofreading is outlier-driven; Mouse is more uniform

### What Worked Best:
- **Figure 1 (Sample Validation):** Demonstrates robustness; readers gain confidence in extrapolation
- **Figure 2 (Distribution Patterns):** Visual story shows data is dramatically different between species
- **Figure 3 (Cost Sensitivity):** Pragmatic: shows what you *can* estimate vs. what needs benchmarking

---

## 4. ARBITRARY (Design Choices & Robustness Testing)
**Did you pick randomly, or does anything work?**

### Linear Extrapolation:
- ❌ Could have used: Bayesian hierarchical model, non-parametric bootstrap, fitted distributions
- ✅ Why linear: <7% variation between n=100 and n=500 validates assumption that per-neuron stats are representative
- **Robustness check:** Showed variation across sample sizes → confidence in method

### 95th Percentile Threshold:
- ❌ Could have used: 90th, 99th percentile, mixture model, user-defined complexity
- ✅ Why 95th: Standard outlier definition; robust across samples; interpretable (top 5%)
- **Arbitrary aspect:** Acknowledged—different thresholds would yield different heavy-tail contributions

### Multiple Sample Sizes:
- ❌ Could have just done n=100 (faster) or n=1000 (more accurate but slower)
- ✅ Why both: n=100 is quick proof-of-concept; n=500 validates scaling; <7% agreement shows sweet spot
- **Robustness:** Validates that extrapolation is not sensitive to sample size choice

### Placeholder Costs (2.0–2.5 sec/op):
- ❌ Could have assumed: 1 sec/op (optimistic), 5 sec/op (conservative), variable by edit type
- ✅ Why this range: Typical vision-language model inference latencies; shows sensitivity analysis
- **Arbitrary:** Explicitly flagged as placeholder; prioritizes benchmarking as next step

---

## TIGHTENING STRATEGY: Comprehensive Report → ICML Paper

### Removed (too detailed for ICML):
- All individual percentile tables (keeps median, mean, heavy-tail only)
- Detailed subsection breakdowns (A-F format) → consolidate into Results prose
- Extensive methodology details → 1 paragraph on data retrieval/sampling
- Per-species breakdowns for both n=100 and n=500 → focus on key finding (Mouse vs Fly contrast)
- Historical edit log interpretation → summarize as "merge/split categorization"

### Kept (essential for ICML):
- **The big surprises:** 23.6× difference, different operation ratios, heavy-tail patterns
- **Validation:** Cross-sample consistency (<7% variation)
- **Cost framework:** Simple formula + example calculation + sensitivity analysis
- **Reproducibility:** Code, commands, datasets, runtimes

### Restructured (for narrative flow):
- Removed "Section 3.1, 3.2, 3.3, 3.4..." → Integrated into single "Key Findings" block
- Removed "Dataset Overview / Edit Distribution Statistics / Edit Type Breakdown..." → Three bullet points in Results
- Moved "Computational Cost Framework" earlier (before "Limitations")
- **IODA narrative** threads through Invent → Old → Discover → Arbitrary structure

### Page Count:
- Original comprehensive report: ~1,100 lines (4.5 pages single-spaced)
- ICML section: ~280 lines (1.5 pages single-spaced, including figures)
- **Compression ratio: 3× tighter, same essential story**

---

## FIGURE PLANNING SUMMARY

| Figure | Purpose | Key Insight | Visualization Type |
|--------|---------|-------------|-------------------|
| **Figure 1** | Validate extrapolation logic | n=100 and n=500 agree within <7% | Scatter plot (4 subpanels) with error bars |
| **Figure 2** | Show species differences | Mouse: balanced, Fly: merge-dominated; different heavy-tail patterns | Histograms + scatter plots (2×2 grid) |
| **Figure 3** | Cost landscape | Cost scales with per-op time; Fly is bottleneck despite lower per-neuron effort | Heatmap + stacked bar chart (2×1 grid) |

### Figure Captions Strategy:
- **Lead with insight:** First sentence states what you learned
- **Quantify:** Specific numbers (422 vs 19, 23.6×, <7%, 36% of edits)
- **Interpretation:** What does this mean for GPU deployment?
- **Technical detail:** What methods generated this visualization?

---

## HOW THIS FOLLOWS ICML STYLE

**Theory/Methods Papers:** Usually 1-2 pages on computations
**This Section:** Balances theory (why sampling works) + empirical results (what we found) + practical implications (cost estimates)

**ICML Readers (non-connectomics audience):**
- Don't know what edit distributions are → **Figure 2 explains with visual analogy** (right-skewed, different shapes)
- Don't know MICrONS vs FlyWire → **Comparison table** with concrete numbers
- Want to understand cost implications → **Figure 3** shows dollars/hours
- Care about reproducibility → **Reproducibility section** with exact commands

---

## NEXT STEPS FOR YOUR PAPER

1. **Generate actual figures** (placeholders ready in markdown)
   - Use existing data from JSON files in `reports/edit_distributions/`
   - Apply matplotlib/seaborn styling consistent with your paper's figure style

2. **Fill in missing costs:**
   - Benchmark Qwen 32B on H100 → measure $t_{\text{merge}}, t_{\text{split}}$ empirically
   - Update placeholder costs (2.0–2.5 sec) with real numbers
   - Refine sensitivity analysis range

3. **Optional: Add one more figure**
   - Breakdown visualization showing which neurons contribute to heavy tail
   - Morphology/location heatmap of high-cost neurons
   - Would strengthen the narrative: "Why are these specific neurons expensive?"

4. **Integrate into full paper:**
   - This section goes in "Overflow" section of ConnectomeAgent.docx
   - Reference this section from main proofreading results: "Cost implications discussed in Methods: GPU Cost Estimation"
   - Link cost estimates to practical deployment scenario in Discussion

