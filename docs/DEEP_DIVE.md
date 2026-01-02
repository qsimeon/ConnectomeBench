# ConnectomeBench: A Complete Deep Dive

This document explains **everything** about ConnectomeBench from first principles. No prior knowledge of connectomics, neuroscience, or the specific technologies is assumed.

---

## Table of Contents

1. [The Big Picture: What Problem Are We Solving?](#1-the-big-picture-what-problem-are-we-solving)
2. [What is a Connectome?](#2-what-is-a-connectome)
3. [The Data Pipeline: From Brain to Benchmark](#3-the-data-pipeline-from-brain-to-benchmark)
4. [Understanding the Technologies](#4-understanding-the-technologies)
5. [The Two Types of Errors: Merge and Split](#5-the-two-types-of-errors-merge-and-split)
6. [Repository Structure Explained](#6-repository-structure-explained)
7. [Authentication: How to Get Access](#7-authentication-how-to-get-access)
8. [The Current State: What Works and What Doesn't](#8-the-current-state-what-works-and-what-doesnt)
9. [Future Work: Scaling RL for Proofreading](#9-future-work-scaling-rl-for-proofreading)

---

## 1. The Big Picture: What Problem Are We Solving?

### The Goal

Imagine you have a complete 3D map of every neuron in a brain. Each neuron is like a wire in an incredibly complex circuit, and understanding how they connect helps us understand how the brain computes. This 3D map is called a **connectome**.

The problem? These maps are created by algorithms that analyze microscopy images, and **the algorithms make mistakes**. Fixing these mistakes currently requires humans to manually inspect millions of neurons—an impossible task at scale.

**ConnectomeBench asks: Can we train AI (specifically Large Language Models with vision capabilities) to help fix these mistakes?**

### Why LLMs?

Traditional computer vision approaches struggle because:
1. The errors are subtle and require understanding of biological structure
2. Context matters—you need to understand what a "correct" neuron looks like
3. The decision often requires reasoning, not just pattern matching

Vision-Language Models (VLMs) like GPT-4o or Claude can potentially:
- Look at 3D renderings of neurons
- Reason about whether the structure looks correct
- Make decisions about whether segments should be merged or split

---

## 2. What is a Connectome?

### The Brain as a Circuit

Your brain contains roughly 86 billion neurons. Each neuron:
- Has a **cell body (soma)** - the "command center"
- Has **dendrites** - input wires that receive signals
- Has an **axon** - the output wire that sends signals
- Forms **synapses** - connection points with other neurons

A **connectome** is a complete wiring diagram showing:
- Every neuron's shape and position
- How each neuron connects to others
- The strength and type of each connection

### How We Get Connectome Data

1. **Slice the brain very thin** - We use an electron microscope to image brain tissue at ~4nm resolution (a human hair is ~80,000nm wide)

2. **Stack the images** - We take thousands of slices and align them into a 3D volume

3. **Segment the neurons** - Algorithms trace each neuron through the volume, assigning each voxel (3D pixel) to a specific neuron ID

4. **Proofread and correct** - Humans fix the inevitable algorithmic errors

### The Datasets in ConnectomeBench

| Species | Dataset Name | Size | What's Special |
|---------|--------------|------|----------------|
| Mouse | MICrONS (minnie65) | 1mm³ of visual cortex | Most mature, well-characterized |
| Fly | FlyWire (FAFB) | Complete fruit fly brain | First complete fly brain connectome |
| Human | H01 | Small piece of temporal cortex | First detailed human connectome sample |
| Zebrafish | Fish1 | Larval zebrafish brain | Whole-brain imaging of a vertebrate |

---

## 3. The Data Pipeline: From Brain to Benchmark

### Step 1: EM Imagery (Electron Microscopy Data)

The raw data is a 3D grayscale image volume. Think of it as a giant 3D photograph of brain tissue where:
- Lighter areas are membranes (boundaries between cells)
- Darker areas are the insides of cells
- The resolution is so high you can see individual proteins

**EM Path Example:**
```
precomputed://gs://h01-release/data/20210601/4nm_raw
```

This URL points to the raw electron microscopy data stored on Google Cloud Storage.

### Step 2: Segmentation (The Algorithm's Best Guess)

AI algorithms process the EM data and assign every voxel to a neuron ID:
- Voxel at position (100, 200, 50) → Neuron ID 864691134965949727
- Voxel at position (100, 201, 50) → Neuron ID 864691134965949727 (same neuron)
- Voxel at position (100, 202, 50) → Neuron ID 720575940621039145 (different neuron)

**Segmentation Path Example:**
```
graphene://https://prod.flywire-daf.com/segmentation/1.0/flywire_public
```

This URL points to the segmentation data, which can be:
- `precomputed://` - Static, pre-computed segmentation
- `graphene://` - Live, editable segmentation (changes as proofreaders fix errors)

### Step 3: 3D Mesh Generation

To visualize a neuron, we convert the segmentation into a 3D mesh:
- The segmentation tells us which voxels belong to neuron X
- We create a surface mesh (like a 3D model) that wraps around those voxels
- This mesh can be rendered from any angle

### Step 4: Benchmark Creation

ConnectomeBench takes:
1. Known errors from the proofreading edit history
2. Generates images of neurons before/after corrections
3. Asks LLMs to identify which images show errors

---

## 4. Understanding the Technologies

### CloudVolume

**What it is:** A Python library for reading and writing large 3D arrays stored in the cloud.

**Why we need it:** Connectome data is HUGE (petabytes). CloudVolume lets us:
- Download only the specific region we need
- Work at different resolutions (MIP levels)
- Handle the `precomputed://` and `graphene://` URL formats

**Example:**
```python
import cloudvolume
cv = cloudvolume.CloudVolume("precomputed://gs://h01-release/data/20210601/4nm_raw")
# Download a small cube of data
data = cv[1000:1100, 2000:2100, 500:550]  # 100x100x50 voxels
```

### CAVEclient

**What it is:** The Python client for CAVE (Connectome Annotation Versioning Engine).

**Why we need it:** CAVE provides:
- **Authentication** - Who is allowed to access which datasets
- **Datastacks** - Named collections of data (EM + segmentation + annotations)
- **Edit history** - Record of every proofreading edit ever made
- **Skeletonization** - Simplified representations of neurons
- **Materialization** - Snapshots of the data at specific times

**Example:**
```python
from caveclient import CAVEclient
client = CAVEclient("minnie65_public")

# Get the segmentation path
seg_path = client.info.segmentation_source()

# Get edit history for a neuron
edits = client.chunkedgraph.get_tabular_changelog(neuron_id)

# Get skeleton of a neuron
skeleton = client.skeleton.get_skeleton(neuron_id)
```

### Neuroglancer

**What it is:** A web-based viewer for large-scale 3D microscopy data.

**Why it matters:** It's the standard tool proofreaders use to:
- View EM data and segmentation simultaneously
- Navigate through the 3D volume
- Make edits to fix errors

When you see a Neuroglancer URL like:
```
https://ngl.brain-wire.org/#!middleauth+https://...
```
This opens a web viewer showing the data.

### Navis

**What it is:** A Python library for neuron analysis and visualization.

**Why we use it:** It provides:
- Mesh loading and manipulation
- Skeleton operations
- 3D plotting (the renderings we show to LLMs)

---

## 5. The Two Types of Errors: Merge and Split

### Merge Errors (Over-segmentation was WRONG)

**What happened:** The algorithm incorrectly merged two different neurons into one.

**What it looks like:** A single "neuron" that has:
- Two cell bodies (impossible biologically)
- Axons that suddenly change direction at 90°
- Dendrites that loop back and cross themselves

**The task:** Given a single neuron rendering, determine if it contains a merge error.

**Example prompt:**
> "Look at this 3D rendering. Does this appear to be a single neuron, or does it look like two neurons incorrectly merged together?"

### Split Errors (Under-segmentation was WRONG)

**What happened:** The algorithm incorrectly split one neuron into multiple pieces.

**What it looks like:** 
- A neuron that suddenly "ends" in the middle of the tissue
- Multiple fragments that should clearly be connected
- A gap where continuity should exist

**The task:** Given two neuron fragments, determine if they should be merged.

**Example prompt:**
> "Look at these two segments (blue and orange). Should they be merged into a single neuron, or are they correctly separate?"

### Why This Is Hard

1. **Real neurons are complex** - They branch extensively and can look unusual
2. **Imaging artifacts** - Sometimes there are gaps due to damaged tissue
3. **Scale** - You need to consider local structure AND global context
4. **Subjectivity** - Even human experts sometimes disagree

---

## 6. Repository Structure Explained

```
ConnectomeBench/
├── src/                          # Core library code
│   ├── connectome_visualizer.py  # Main class for all visualization
│   ├── prompts.py                # LLM prompt construction
│   ├── util.py                   # LLM API wrappers
│   └── analysis_utils.py         # Result analysis helpers
│
├── scripts/                      # Main processing scripts
│   ├── get_data.py               # Gather training data from edit histories
│   ├── split_resolution.py       # Evaluate LLMs on split error tasks
│   ├── merge_resolution.py       # Evaluate LLMs on merge error tasks
│   ├── segmentation_classification.py  # Classify segment types
│   └── CAVEsetup.ipynb           # Authentication setup notebook
│
├── examples/                     # Tutorial scripts
│   ├── segment_classification.py # Simple example of classification
│   ├── split_error_correction.py # Simple example of split task
│   └── merge_error_identification.py  # Simple example of merge task
│
├── analysis/                     # Result analysis scripts
│   ├── analyze_*_results.py      # Analyze benchmark results
│   └── run_analysis.sh           # Run all analyses
│
├── visualization/                # Visualization tools
│   ├── visualize_merge_errors.py
│   └── visualize_merge_errors_html.py
│
├── tests/                        # Unit tests
└── baselines/                    # Non-LLM baseline models (ResNet)
```

### Key Files Explained

#### `src/connectome_visualizer.py`

This is the heart of the codebase. The `ConnectomeVisualizer` class:

```python
class ConnectomeVisualizer:
    """
    Main class for loading and visualizing neurons.
    
    Supports: mouse, fly, human, zebrafish
    """
    
    def __init__(self, species="fly"):
        # 1. Set up data paths for the species
        # 2. Initialize CAVEclient for API access
        # 3. Connect to CloudVolume for data access
        
    def load_neurons(self, neuron_ids):
        # Download 3D meshes for the given neuron IDs
        
    def create_3d_neuron_figure(self, bbox=None):
        # Create a 3D visualization
        
    def save_3d_views(self, base_filename):
        # Save front/side/top views as images
        
    def get_edit_history(self, neuron_id):
        # Get the proofreading history for a neuron
```

The `data_parameters` dictionary defines each species:

```python
data_parameters = {
    "mouse": {
        "em_path": "precomputed://...",      # EM imagery location
        "seg_path": "graphene://...",        # Segmentation location
        "datastack_name": "minnie65_public", # CAVE datastack name
        "em_mip": 2,                         # Resolution level for EM
        "seg_mip": 2                         # Resolution level for seg
    },
    # ... similar for fly, human, zebrafish
}
```

#### `scripts/get_data.py`

This script:
1. Connects to CAVEclient for a species
2. Queries for neurons that have been proofread
3. Gets the edit history (what merges/splits were made)
4. Downloads the relevant EM and mesh data
5. Saves it as JSON for later processing

```bash
python scripts/get_data.py --species mouse --num-neurons 200
```

#### `src/prompts.py`

Contains functions that construct prompts for LLMs:

- `create_merge_identification_prompt()` - "Is this merge correct?"
- `create_split_identification_prompt()` - "Does this need to be split?"
- `create_segment_classification_prompt()` - "What type of structure is this?"

Each prompt includes:
1. Base64-encoded images of the neuron
2. Context about what the LLM is looking at
3. Instructions for how to respond

---

## 7. Authentication: How to Get Access

### Why Authentication?

These datasets represent years of work and significant investment. Access is controlled to:
- Track who is using the data
- Ensure compliance with data sharing agreements
- Prevent abuse

### Species by Access Level

| Species | Access Level | How to Get Access |
|---------|--------------|-------------------|
| Mouse (MICrONS) | Auto-authorized | Just run `caveclient` setup |
| Fly (FlyWire) | Auto-authorized | Just run `caveclient` setup |
| Human (H01) | Gated public | Must accept TOS at proofreading page |
| Zebrafish (Fish1) | Gated public | Must run setup notebook |

### Step-by-Step Authentication

#### For Mouse and Fly (Easy)

```python
from caveclient import CAVEclient

# First time only - follow the prompts
client = CAVEclient()
client.auth.setup_token(make_new=True)
# This opens a browser, you log in, copy the token, paste it back

# After setup, just use:
client = CAVEclient("minnie65_public")  # Mouse
client = CAVEclient("flywire_fafb_public")  # Fly
```

#### For Human (H01)

1. **Visit the proofreading page:**
   https://h01-release.storage.googleapis.com/proofreading.html

2. **Log in with your Google account** when prompted

3. **Run the CAVEsetup notebook** (`scripts/CAVEsetup.ipynb`):
   ```python
   import caveclient
   url = "https://global.brain-wire-test.org/"
   auth = caveclient.auth.AuthClient(server_address=url)
   auth.setup_token(make_new=True)
   # Follow the link, copy token, save it
   auth.save_token(token="YOUR_TOKEN_HERE", overwrite=True)
   ```

4. **Test the connection:**
   ```python
   client = caveclient.CAVEclient(
       datastack_name='h01_c3_flat', 
       server_address="https://global.brain-wire-test.org/"
   )
   print("Success!")
   ```

#### For Zebrafish (Fish1)

Same process as H01, but start at:
https://fish1-release.storage.googleapis.com/tutorials.html

The datastack name is `fish1_full`.

---

## 8. The Current State: What Works and What Doesn't

### What Works ✅

| Feature | Mouse | Fly | Human | Zebrafish |
|---------|-------|-----|-------|-----------|
| CAVEclient connection | ✅ | ✅ | ✅ | ✅ |
| EM data access | ✅ | ✅ | ✅ | ✅ |
| Segmentation access | ✅ | ✅ | ❌ | ❌ |
| Mesh loading | ✅ | ✅ | ⚠️ | ⚠️ |
| Edit history | ✅ | ✅ | ✅ | ✅ |
| get_data.py | ✅ | ✅ | ✅ | ✅ |

### The Segmentation Issue for Human and Zebrafish

When you run `ConnectomeVisualizer` for human or zebrafish, you get:
```
human: EM=True, Seg=False
zebrafish: EM=True, Seg=False
```

**What this means:**
- EM imagery works (we can download the raw microscopy images)
- Segmentation via CloudVolume fails

**Why this happens:**

For **Human (H01):**
- The segmentation path `precomputed://gs://h01-release/data/20210601/c3` requires authenticated access
- CloudVolume doesn't automatically use CAVEclient's token
- The workaround: Use CAVEclient's segmentation API directly

For **Zebrafish (Fish1):**
- The segmentation path `graphene://https://pcgv3local.brain-wire-test.org/...` returns 401 Unauthorized
- This is a server-side authentication issue
- CloudVolume can't access it without proper token handling

**Why this might be OK:**

The core workflow doesn't always need direct CloudVolume segmentation access:
1. CAVEclient can still provide edit history
2. CAVEclient can provide mesh data through its API
3. The code has fallbacks that use CAVEclient's segmentation source

**Current workaround in code:**
```python
# In connectome_visualizer.py, we try to get the segmentation path from CAVEclient
client_seg_path = self.client.info.segmentation_source()
if client_seg_path.startswith("graphene://"):
    self.seg_path = client_seg_path  # Use the graphene path
```

---

## 9. Future Work: Scaling RL for Proofreading

### The Vision

Right now, ConnectomeBench evaluates whether LLMs *can* identify errors. The next step is:

**Train AI to actively proofread using Reinforcement Learning**

Imagine an AI agent that:
1. Navigates through the connectome
2. Identifies likely errors
3. Proposes fixes
4. Gets feedback on correctness
5. Learns to get better over time

### Why RL?

- **Supervised learning** would require labeled examples of every possible error type
- **RL** can learn from trying things and seeing if they work
- The proofreading task has clear rewards: correct fixes improve connectivity

### Current Limitations

1. **No VLM support yet** - The RL framework doesn't support vision models yet
2. **API latency** - Making many LLM calls is slow and expensive
3. **State representation** - How do you represent the "state" of a connectome to an RL agent?

### What GlebRazgar's Changes Add

By adding human and zebrafish support, we can:
1. Test LLMs on more diverse neuron morphologies
2. Validate that approaches generalize across species
3. Prepare for larger-scale proofreading when VLM-based RL becomes available

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Axon** | The output wire of a neuron that sends signals |
| **CAVEclient** | Python library for accessing CAVE services |
| **CloudVolume** | Python library for reading 3D neuroimaging data |
| **Connectome** | Complete wiring diagram of a brain |
| **Datastack** | Named collection of EM + segmentation + annotations |
| **Dendrite** | Input wires of a neuron that receive signals |
| **EM** | Electron Microscopy - imaging technique |
| **Graphene** | Protocol for live, editable segmentation |
| **Mesh** | 3D surface representation of a neuron |
| **MIP level** | Multi-resolution pyramid level (0 = highest resolution) |
| **Precomputed** | Static, pre-processed data format |
| **Proofreading** | Manual correction of segmentation errors |
| **Segmentation** | Assignment of voxels to neuron IDs |
| **Soma** | Cell body of a neuron |
| **Synapse** | Connection point between neurons |
| **Voxel** | 3D pixel in volumetric data |

---

## Appendix B: Quick Reference Commands

### Setup
```bash
# Clone and install
git clone https://github.com/jffbrwn2/connectomebench.git
cd connectomebench
uv pip install -e .

# Authenticate with CAVEclient
python -c "from caveclient import CAVEclient; c = CAVEclient(); c.auth.setup_token(make_new=True)"
```

### Run Examples
```bash
# Segment classification
python examples/segment_classification.py --num-samples 10 --model gpt-4o

# Split error task
python examples/split_error_correction.py --num-samples 10 --model gpt-4o

# Merge error task
python examples/merge_error_identification.py --num-samples 10 --model gpt-4o
```

### Generate Data
```bash
# Mouse data
python scripts/get_data.py --species mouse --num-neurons 100

# Human data (requires authentication first!)
python scripts/get_data.py --species human --num-neurons 50
```

### Test Species Access
```python
from src.connectome_visualizer import ConnectomeVisualizer

for species in ["mouse", "fly", "human", "zebrafish"]:
    v = ConnectomeVisualizer(species=species, verbose=True)
    print(f"{species}: CAVEclient={v.client is not None}")
```

---

## Appendix C: Links and Resources

### Dataset Pages
- **MICrONS (Mouse):** https://www.microns-explorer.org/
- **FlyWire (Fly):** https://flywire.ai/
- **H01 (Human):** https://h01-release.storage.googleapis.com/landing.html
- **Fish1 (Zebrafish):** https://fish1-release.storage.googleapis.com/tutorials.html

### H01 Proofreading Methodology
The [H01 Proofreading Page](https://h01-release.storage.googleapis.com/proofreading.html) provides detailed documentation on:
- How to identify merge and split errors
- The reasoning behind proofreading decisions
- Visual examples of correct and incorrect segmentation

This is the methodology that ConnectomeBench is based on.

### Libraries
- **CAVEclient:** https://github.com/CAVEconnectome/CAVEclient
- **CloudVolume:** https://github.com/seung-lab/cloud-volume
- **Navis:** https://navis.readthedocs.io/

### Papers
- MICrONS: https://www.biorxiv.org/content/10.1101/2021.07.28.454025v2
- FlyWire: https://www.nature.com/articles/s41586-024-07558-y
- H01: https://www.science.org/doi/10.1126/science.adk4858

