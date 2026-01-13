# ConnectomeBench: Can LLMs Proofread the Connectome?

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jffbrwn2/connectomebench.git
cd connectomebench
```

2. Install the package in editable mode (required to access the `src` module):
```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

The `-e` flag installs the package in editable mode, which makes the `src` directory available for import in your scripts (e.g., `from src.prompts import ...`). This is required for the example scripts to work.


## Quick Start: Running the Benchmark

The easiest way to get started is using our example scripts with the pre-built HuggingFace dataset. These scripts demonstrate how to evaluate LLMs on each task.

### Prerequisites

1. Set up your LLM API keys (e.g., OpenAI, Anthropic)
2. Authenticate with HuggingFace:
```bash
huggingface-cli login
```

### Example Scripts

#### 1. Segment Classification
Classify neuron segments as correctly segmented, undersegmented, or oversegmented:

```bash
python examples/segment_classification.py --num-samples 10 --model gpt-4o
```

#### 2. Split Error Correction
Identify which neuron segments contain split errors (incorrectly split segments):

```bash
python examples/split_error_correction.py --num-samples 10 --model gpt-4o --prompt-mode informative
```

#### 3. Merge Error Identification
Identify which candidate neuron should be merged with a base neuron:

```bash
python examples/merge_error_identification.py --num-samples 10 --model gpt-4o --prompt-mode informative
```

### Common Parameters

All example scripts support:
- `--num-samples N`: Evaluate on N samples (default: all)
- `--model MODEL`: LLM model to use (default: gpt-4o)
- `--prompt-mode MODE`: Prompt style - 'informative' or 'minimal' (default: informative)
- `--output-dir DIR`: Where to save results (default: output/tutorial_results)

Results are saved as CSV files with accuracy metrics when ground truth is available.

## Advanced Usage

### Data Processing

The toolkit provides several scripts for processing connectome data from scratch:

- `scripts/get_data.py`: Gather training data from MICrONS, FlyWire or Fish1 edit histories
- `scripts/split_resolution.py`: Process and evaluate split error resolution tasks
- `scripts/merge_resolution.py`: Process and evaluate merge error detection tasks
- `scripts/segmentation_classification.py`: Classify segmentations

### Visualization

The `ConnectomeVisualizer` class provides tools for visualizing neurons and EM data:

```python
from src.connectome_visualizer import ConnectomeVisualizer

# Initialize visualizer for any supported species
visualizer = ConnectomeVisualizer(species="zebrafish")  # or "mouse", "fly", "human", "zebrafish"

# Load and visualize neurons
visualizer.load_neurons([864691134965949727])
visualizer.save_3d_views(base_filename="3d_neuron_mesh")
```

### Dataset Access and Authentication

The toolkit supports multiple connectomics datasets with varying access requirements:

#### Mouse (MICrONS) and Fly (FlyWire)
These datasets are generally publicly accessible and do not require special authentication.

#### Human (H01) and Zebrafish (Fish1)
These datasets require authentication with the CAVE (Connectome Annotation Versioning Engine) system.

**Quick Setup (Automated):**

```bash
# 1. Install dependencies (if not already done)
pip install -e .

# 2. Run the authentication setup script
python scripts/setup_cave_auth.py
```

The script will guide you through the authentication process.

**Manual Setup:**

1. **Request Access**
   - Fill out the access request form: https://forms.gle/tpbndoL1J6xB47KQ9
   - You will receive approval within 24 hours from the Lichtman Lab team

2. **Get Your Token**
   - After approval, visit ONE of these URLs (log in with the Gmail account you used in the form):
     - For NEW users: https://global.brain-wire-test.org/auth/api/v1/create_token
     - For EXISTING users: https://global.brain-wire-test.org/auth/api/v1/user/token
   - Copy the token (it looks like: `4836d842ee67473399ca468f61d773ff`)

3. **Configure Environment**
   - Create a `.env` file in the project root (see `.env.example` for template):
     ```bash
     # Copy the example file
     cp .env.example .env

     # Edit .env and add your token
     CAVECLIENT_TOKEN=your_token_here
     ```

4. **Verify Setup**
   ```python
   from src.connectome_visualizer import ConnectomeVisualizer

   # This will now work without authentication errors
   visualizer = ConnectomeVisualizer(species="zebrafish")
   ```

**Environment Variables:**

The toolkit uses a `.env` file to manage API keys and tokens. See `.env.example` for a complete list of supported variables:

- `CAVECLIENT_TOKEN` - Required for H01 and Fish1 datasets
- `OPENAI_API_KEY` - For GPT models (benchmark evaluation)
- `ANTHROPIC_API_KEY` - For Claude models (benchmark evaluation)
- `GEMINI_API_KEY` - For Gemini models (benchmark evaluation)
- `HF_TOKEN` - For HuggingFace datasets

**Additional Resources:**

- [H01 Proofreading Page](https://h01-release.storage.googleapis.com/proofreading.html) - Detailed documentation on the proofreading methodology
- [Fish1 Tutorials](https://fish1-release.storage.googleapis.com/tutorials.html) - Zebrafish dataset tutorials
- [CAVEClient Documentation](https://github.com/CAVEconnectome/CAVEclient/) - Core data access library

### LLM Integration

The toolkit integrates with multiple LLM providers for automated analysis:

```python
from src.util import LLMProcessor

# Initialize LLM processor
processor = LLMProcessor(model="gpt-4o")

# Process data
results = await processor.process_batch(["Write prompt here"])
```

## Advanced Workflows

The following sections describe how to generate data and run full benchmarks from scratch.

### Segmentation Classification

The `scripts/segmentation_classification.py` script provides a way to classify segmentations into different categories. To get the same results as the paper, run the following command:

```bash
python scripts/segmentation_classification.py
```

To run with custom parameters (e.g., specific models, species, or number of neurons):

```bash
python scripts/segmentation_classification.py --models claude-3-7-sonnet-20250219 gpt-4o --species mouse fly --num-neurons 200 --k 5 --seed 42
```

See the script's help for all available options:

```bash
python scripts/segmentation_classification.py --help
```

## Split error resolution & merge error detection

The toolkit provides separate scripts for testing LLM performance on split and merge error tasks:

### Split Error Resolution

Use `scripts/split_resolution.py` to evaluate split error correction. For mouse split error correction (identification) with Claude 3.7 Sonnet:

```bash
python scripts/split_resolution.py --input-json scripts/training_data/mouse_256nm.json --task split_identification --species mouse --zoom-margin 2048 --models claude-3-7-sonnet-20250219 --prompt-modes informative
```

For mouse split error correction (comparison) with Claude 3.7 Sonnet:

```bash
python scripts/split_resolution.py --input-json scripts/training_data/mouse_256nm.json --task split_comparison --species mouse --zoom-margin 2048 --models claude-3-7-sonnet-20250219 --prompt-modes informative
```

### Merge Error Detection

Use `scripts/merge_resolution.py` to evaluate merge error detection. For mouse merge error detection (identification) with Claude 3.7 Sonnet:

```bash
python scripts/merge_resolution.py --input-json scripts/training_data/merge_error_only_mouse.json --task merge_identification --species mouse --models claude-3-7-sonnet-20250219 --prompt-modes informative
```

For mouse merge error detection (comparison) with Claude 3.7 Sonnet:

```bash
python scripts/merge_resolution.py --input-json scripts/training_data/merge_error_only_mouse.json --task merge_comparison --species mouse --models claude-3-7-sonnet-20250219 --prompt-modes informative
```

## Testing

Install test dependencies and run tests:
```bash
# Using uv
uv pip install -e ".[test]"

# Or using pip
pip install -e ".[test]"

# Run tests
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

When contributing:
1. Write tests for new features
2. Ensure all tests pass: `pytest`
3. Check code coverage: `pytest --cov=src`
4. Follow existing code style and conventions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```
@software{connectomebench2025,
  author = {Jeff Brown, Andrew Kirjner, Tim Farkas},
  title = {ConnectomeBench: Can LLMs proofread the connectome?},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/jffbrwn2/connectomebench}
}
``` 
