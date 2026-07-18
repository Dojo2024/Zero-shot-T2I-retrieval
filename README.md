
---

# Multimodal Fashion & Context Retrieval Engine

This repository contains a modular, production-grade multimodal retrieval engine that evaluates zero-shot image-text alignment, context grounding, and compositionality. It supports side-by-side comparison across five state-of-the-art vision-language model architectures, with dedicated evaluation pipelines for fashion taxonomies (using the Fashionpedia subset) and general-purpose scenes (using MS COCO).

---

## 📊 Supported Architectures

| CLI Option | Architecture | Model Identifier / Scale | Description |
| :--- | :--- | :--- | :--- |
| `clip` | **Traditional CLIP** | `openai/clip-vit-base-patch32` | Standard dual-encoder baseline |
| `fclip` | **Fashion-CLIP** | `patrickjohncyh/fashion-clip` | Domain-specific fine-tuned model for retail fashion |
| `siglip2` | **SigLIP 2** | `google/siglip2-so400m-patch14-224` | Pairwise Sigmoid Loss with location-aware dense decoding |
| `tipsv2` | **TIPSv2** | `google/tipsv2-so400m14` | Google DeepMind SOTA with strict patch-text alignment |
| `pe` | **Perception Encoder** | `PE-Core-G14-448` | Facebook Research model preserving intermediate features |

---

## ⚙️ Prerequisites & Setup

This repository has been tested on Linux and cloud environments utilizing NVIDIA GPUs (e.g., H100, L4). Follow these setup steps to align dependencies and prevent C-extension binary compilation mismatches (such as the standard `NumPy 2.x` / `matplotlib` compatibility issue).

### 1. Set Up the Environment
Create and activate your Python environment:
```bash
conda create -n retrieval python=3.12
conda activate retrieval
```

### 2. Install Dependencies
Install PyTorch and the core scientific libraries. To maintain compatibility with pre-compiled libraries in the workspace, pin the NumPy version to `<2.0`:
```bash
# Install PyTorch (CUDA supported)
pip install torch torchvision torchaudio

# Install baseline libraries
pip install transformers timm einops sentencepiece scikit-learn matplotlib tqdm
pip install ftfy decord

# Downgrade NumPy to prevent compilation mismatches with C-extensions
pip install "numpy<2"
```

### 3. Clone and Install the Perception Encoder Repository
The Facebook Research `pe` model requires local file paths. Clone it and install it in editable mode so it can be resolved globally:
```bash
git clone https://github.com/facebookresearch/perception_models.git
pip install -e ./perception_models/
```

---

## 🚀 Pipeline A: Fashion & Context Retrieval (Fashionpedia)

This pipeline parses a raw directory of fashion images, constructs a high-dimensional vector database, and queries it using natural language.

### Step 1: Verify & Build the Vector Index (`indexer.py`)
Run the unified indexer. It scans the target image directory, checks for empty or corrupted files, processes the healthy images through the specified vision encoder, and saves the L2-normalized embeddings inside a distinct index file on disk.

```bash
# Standard usage template:
python indexer.py --model [clip | fclip | siglip2 | tipsv2 | pe] --batch_size 32

# Examples:
python indexer.py --model tipsv2
python indexer.py --model fclip
python indexer.py --model pe
```
*Indices are saved in the format `/teamspace/studios/this_studio/work/index_{model_name}.pt`.*

### Step 2: Query the Vector Database (`query.py`)
Search your index using standard natural language prompts. It encodes your query text with the matching text encoder, performs a cosine-similarity matrix multiplication, and saves a visual layout grid of the top results under the correct query results directory.

```bash
# Standard usage template:
python query.py --model [clip | fclip | siglip2 | tipsv2 | pe] --query "your query text" --top_k 5

# Examples:
python query.py --model tipsv2 --query "A person in a bright yellow raincoat."
python query.py --model pe --query "Professional business attire inside a modern office."
```
*Visual grid results are saved inside `/teamspace/studios/this_studio/work/query_results_{model_name}/`.*

---

## 🔬 Pipeline B: Quantitative Zero-Shot Evaluation (MS COCO)

These scripts evaluate the zero-shot performance of Facebook's **Perception Encoder** on general-purpose images, analyzing retrieval metrics and layers.

### 1. Evaluate MS COCO Zero-Shot Metrics (`verify_coco_pe.py`)
This script downloads the validation annotations (Karpathy Split) and MS COCO `val2014` images. It processes them on your GPU and outputs the exact Text-to-Image (T2I) and Image-to-Text (I2T) Recall@1, Recall@5, and Recall@10 metrics:

```bash
python verify_coco_pe.py
```

### 2. Sweep Intermediate Layers for Token Impacts (`layer_impact_coco.py`)
The Perception Encoder retains rich visual details before standard contrastive projection collapse [1.1.8]. This diagnostic utility truncates the Vision Transformer blocks at various depths, performs MS COCO evaluation on a 500-image subset, and saves a comparative plot named `layer_impact_recall.png` to analyze where uncollapsed visual representations reside.

```bash
python layer_impact_coco.py
```

### 3. Query the MS COCO Database (`query_coco.py`)
This script processes the full 5,000-image MS COCO validation set to build a cached index. Once cached, you can run instant, low-latency search queries:

```bash
# Build the index (Runs once and caches) and query
python query_coco.py --query "A red train parked on the tracks" --top_k 5
```
*Visual grid plots are stored inside `./query_results/`.*

---

## 📁 Repository Structure

```
Retrieval/
├── dataset.py               # Custom Dataset pipeline, image integrity checks, and collator
├── indexer.py               # Unified index generator (CLIP, FCLIP, SigLIP 2, TIPSv2, PE)
├── query.py                 # Multi-model retrieval engine with automated text calibration
├── verify_coco_pe.py        # Quantitative MS COCO Recall@1 benchmark for PE
├── layer_impact_coco.py     # Diagnostic visual layer truncation sweep and plot generator
├── query_coco.py            # Search utility for MS COCO using the PE G14-448 model
├── perception_models/       # Cloned dependency folder for the Perception Encoder
└── README.md                # General system documentation
```

---

## ⚠️ Troubleshooting & Warnings

*   **Matplotlib/Decord NumPy Crash:** If you encounter `ImportError: numpy.core.multiarray failed to import` when plotting or loading data, make sure you ran `pip install "numpy<2"`.
*   **SigLIP 2 Processor Crash:** Standard Hugging Face `AutoProcessor` can crash with SigLIP 2 due to custom tokenization mappings. Our `indexer.py` and `query.py` bypass this issue by decoupling the loading of `AutoImageProcessor` and `AutoTokenizer` separately.
*   **TIPSv2 Remote Execution:** Since TIPSv2 models run via custom repository modules on Hugging Face, ensure `trust_remote_code=True` remains active during initialization.