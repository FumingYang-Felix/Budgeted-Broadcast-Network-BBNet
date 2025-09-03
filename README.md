# BBNet – Notebook Guides (ASR • CV • Change Detection)

This README walks you through **three end‑to‑end notebooks** that demonstrate Budgeted Broadcast (BB) in different domains:

- **ASR** — `ASR_BBN(SP)_RigL_Dense_Magnitude_TopK.ipynb`
- **Computer Vision (Face ID/Verification)** — `CV_BBNet.ipynb`
- **Change Detection (LEVIR‑CD)** — `LEVIR_CD_BBN(SP)_Dense.ipynb`

> **Where to run?** All three notebooks are Colab‑friendly and can also be run locally with a CUDA‑enabled PyTorch. ASR and CD notebooks **download their datasets automatically inside the notebook**. The CV notebook expects a data pack from this repo’s **Releases** (see below).


---

## Table of contents

1. [Common setup (Colab & local)](#common-setup-colab--local)
2. [ASR notebook](#1-asr-notebook---asr_bbnsprigl_dense_magnitude_topkipynb)
3. [CV notebook (Face ID/Verification)](#2-cv-notebook---cv_bbnetipynb)
4. [Change Detection notebook (LEVIR‑CD)](#3-change-detection-notebook---levir_cd_bbnsp_denseipynb)
5. [Results artifacts & where they are saved](#results-artifacts--where-they-are-saved)
6. [Reproducibility (seeds, budgets, masks)](#reproducibility-seeds-budgets-masks)
7. [Troubleshooting](#troubleshooting)

---

## Common setup (Colab & local)

### Run on Colab
- In Colab, set **GPU**: `Runtime → Change runtime type → T4/A100 GPU`.
- The notebooks take care of their own `pip install`s where needed.
- Optional but recommended: **mount Drive** to persist checkpoints/logs:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

### Run locally
- Install Python **3.10–3.12** and a CUDA‑compatible **PyTorch** build.
- Create a venv and install typical deps (PyTorch, TorchVision, TorchAudio, numpy, pandas, matplotlib, tqdm, scikit‑image, opencv‑python, sentencepiece/transformers where used). Most `pip install` cells are embedded in the notebooks.
- Launch Jupyter/VSCode and open the notebook you want.

> **Tip:** If you want a minimal one‑liner for PyTorch (Linux, CUDA 12.1), see https://pytorch.org/get-started/locally/ for the exact command for your OS/CUDA.


---

## 1) ASR notebook — `ASR_BBN(SP)_RigL_Dense_Magnitude_TopK.ipynb`

**Goal.** Compare Dense vs **BB (SP‑in / SP‑out)** vs Magnitude vs Top‑K vs RigL on a standard LibriSpeech setting. The notebook computes **WER/WERR** and optional **bucketed ΔWER** (Head/Mid/Tail).

### What it does
- Sets up a vanilla encoder–decoder Transformer (no task‑specific tricks).
- **Downloads LibriSpeech** splits used in the paper (inside the notebook).
- Implements BB **SP‑in / SP‑out** mask refresh with activity EMA and Top‑k regrowth.
- Trains with the fixed schedule described in the paper (dense warm‑up → sparse refresh cycles).

### How to run
1. Open the notebook on **Colab** (GPU recommended) or locally.
2. Run the **Setup** cell(s) to install packages.
3. (Optional) Mount Google Drive to persist outputs.
4. Choose a **method** and **budget** in the configuration cell (e.g., `method="SP_in"`, `budget=0.5`).
5. Run all cells. The notebook will:
   - Download the dataset automatically.
   - Train/evaluate and print **WER/WERR** for `dev-clean`, `test-clean`, `test-other`.
   - (If enabled) compute **bucketed ΔWER** using a fixed Head/Mid/Tail construction from `train-clean-100`.

### Outputs
- Checkpoints and logs (W&B optional) are written to a run directory printed by the notebook near the top.
- Final tables with **WER**, **WERR**, and (optionally) **ΔWER per bucket**.

### Notes
- **Budget (effective density)** is the average kept‑ratio over masked slabs. Values like `0.5–0.7` are a good starting point (as in the paper).
- SP‑in is applied to `W1` (fan‑in), SP‑out to `W2` (fan‑out).


---

## 2) CV notebook — `CV_BBNet.ipynb`

**Goal.** Train a ResNet‑101 variant on **Face identification** (classification) and **verification** with the **BB (SP‑in)** rule on the $1\times1$ channel–channel slab. This notebook expects a dataset pack named **`Facial_Recognition.zip`** that you posted under **GitHub Releases**.

### Get the data (Colab)
Run this one‑cell snippet **before** executing the notebook’s training cells (edit the URL to your repo + tag):

```python
from google.colab import drive
drive.mount('/content/drive')

# Clean slate
!rm -rf /content/data && mkdir -p /content/data

# Fast downloader
!apt -y -qq install aria2 >/dev/null

# === Replace with your release URL ===
ASSET_URL = "https://github.com/<your-org>/<your-repo>/releases/download/<tag>/Facial_Recognition.zip"
OUT = "/tmp/" + ASSET_URL.rsplit("/", 1)[-1]

# 16-connection download with resume
!aria2c -q -x 16 -s 16 -k 1M -c -o "$(basename {OUT})" "{ASSET_URL}" -d /tmp
!unzip -qo "{OUT}" -d "/content/data"

!echo "--- Top-level listing ---"
!ls -lah /content/data | head -n 100
```

> If you want the data to persist across restarts, copy it into your Drive:
> ```bash
> cp -r /content/data "/content/drive/MyDrive/BBNet_Facial_Recognition"
> ```

### Point the notebook to the data
- In `CV_BBNet.ipynb`, set the dataset root to `/content/data` (or to your Drive copy). The exact variable name is shown in the first configuration cell (e.g., `DATA_ROOT = "/content/data"`).

### What it does
- Builds/loads **ResNet‑101** with a $1{\times}1$ SP‑in adaptor (variance‑preserving rescale).
- Trains across **budgets** and compares **Dense**, **BB (SP‑in)**, and baselines where provided.
- Reports **Top‑1 classification accuracy** and **verification accuracy**; can plot **Pareto fronts** vs budget.

### Outputs
- Checkpoints, CSV logs, and best‑per‑budget summaries to a run directory printed by the notebook.


---

## 3) Change Detection notebook — `LEVIR_CD_BBN(SP)_Dense.ipynb`

**Goal.** Train a lightweight FC‑Siam‑conc model on **LEVIR‑CD** and compare **Dense** vs **BB (SP‑in)** on **IoU/F1**; export qualitative panels of Dense‑only vs BB‑only TPs.

### What it does
- **Downloads LEVIR‑CD** inside the notebook (no manual download needed).
- Implements SP‑in on convolutional fan‑in with variance‑preserving rescale and Top‑k regrowth.
- Trains both Dense and SP‑in for the same number of epochs; computes IoU/F1 and exports ranked qualitative panels (ΔTP).

### How to run
1. Open the notebook on Colab/local and select GPU if available.
2. Run the setup cells; the dataset is downloaded automatically.
3. Choose `method="Dense"` or `method="SP_in"` and a budget (kept ratio) if applicable.
4. Execute all cells; the notebook will save quantitative results and image panels.

### Outputs
- A results directory with CSV metrics (**IoU**, **F1**) and qualitative PNG panels illustrating where BB adds true positives inside the ground truth regions.


---

## Results artifacts & where they are saved

Each notebook prints its **run/output directory** near the top (e.g., under `/content/outputs/...` or in your Drive if mounted). You can safely rename/move those folders after runs. Typical contents:
- **Checkpoints** (`*.pt` / `*.pth`)
- **CSV logs** (per‑epoch metrics and budgets)
- **Plots** (Pareto, learning curves) / **PNG panels** (for LEVIR‑CD)
- An optional `meta.json` summarizing key hyperparameters

> If you use Weights & Biases, runs will also appear in your W&B project with the same run IDs.


---

## Reproducibility (seeds, budgets, masks)

- **Seeds.** Each notebook exposes a `seed` parameter; set it for deterministic dataloader/model init where supported.
- **Budgets.** The **budget** is the kept‑ratio (effective density) of masked slabs. Common values to try: `0.3, 0.5, 0.7, 1.0`.
- **SP‑in/SP‑out.** SP‑in masks **fan‑in** (columns of `W1` / Cin→Cmid); SP‑out masks **fan‑out** (rows of `W2`). Both use:
  - activity EMA → target degree \(k_j=d_0+\beta^{-1}\log\frac{1-a_j}{a_j}\) with clipping,
  - **Top‑k regrowth** from the **full** row/column on each refresh,
  - variance‑preserving rescale to stabilize training.


---

## Troubleshooting

- **Colab says “No GPU”** — go to `Runtime → Change runtime type` and select a GPU.
- **Torch/TorchAudio import errors** — the setup cell pins compatible versions; re‑run the setup cell and restart the runtime if needed.
- **Out‑of‑memory** — reduce batch size and/or image size (CV/CD), shorten audio segment length (ASR), or try a smaller model.
- **Data not found (CV)** — make sure `Facial_Recognition.zip` was unzipped to `/content/data` (or update `DATA_ROOT` accordingly).
- **Drive permission issues** — if you mounted Drive, verify the target path is inside `/content/drive/MyDrive/...` and that you have write permission.
- **Long training** — you can run fewer epochs first to verify the pipeline, then scale up. (All methods compare fairly at matched budgets/epochs.)

If something still blocks you, please open an issue with the notebook name, the cell that failed, and the last 50 lines of the traceback.


---

**Happy experimenting!** BBNet’s notebooks are designed to be self‑contained: setup cells install what’s needed, ASR/CD fetch their own data, and CV only needs the one **`Facial_Recognition.zip`** release asset.
