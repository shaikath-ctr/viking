# Viking JAX: Open-Source Port & SEANet Replacement

This repository houses a clean, standalone, and 100% open-source compliant JAX/Flax port of the **Viking** audio model.

The primary objective of this port is to decouple Viking from Google's internal `google3` environment so that it can be staged in `third_party` and eventually open-sourced. This JAX implementation serves as a lightweight, high-performance replacement for **SEANet** in streaming audio and neural codec pipelines.

---

## 🎯 Porting Objectives & Current Status

### What We Have Done So Far:
1. **Eliminated `google3` Dependencies:**
   * **Test Suite (`viking_jax_test.py`):** Replaced the absolute google3 import (`from google3.experimental.users.tomoakitomi.afhi import viking_jax`) with a direct local import (`import viking_jax`).
   * **Notebook (`simplified_seanet.ipynb`):** Removed Google-internal Colab development tools (`colabtools.adhoc_import` and `google3.pyglib.resources`) used previously to dynamically import libraries inside CitC clients.

2. **Standardized Open-Source Environment:**
   * Created **`requirements.txt`** containing all public dependencies (`jax`, `flax`, `optax`, `librosa`, etc.).
   * Set up a **`.gitignore`** file to prevent tracking of local virtual environments (`.venv`), temporary python caches, and notebook checkpoints.

3. **Exhaustive Local Verification:**
   * Constructed an isolated python virtual environment (`venv`).
   * Ran the full test suite (`viking_jax_test.py`) to confirm that model initialization, forward execution, backpropagation, and numerical stability (no `NaN`s in gradients or intermediate activations) behave perfectly out-of-the-box.

4. **Pushed & Staged on GitHub:**
   * Hosted on GitHub at [shaikath-ctr/viking](https://github.com/shaikath-ctr/viking) for seamless sharing and integration.

---

## 📁 Repository Structure

*   **`viking_jax.py`**: The core implementation containing:
    *   `VikingCell`: A recurrent cell leveraging parallel LSTM branches (for processing multi-band features) integrated with a central hub dense layer. Includes custom boundary clipping to prevent gradient/cell explosions.
    *   `VikingGenerator`: High-level generator performing short-time Fourier transform (STFT), power magnitude compression, scanned Viking cell computation, and inverse STFT (ISTFT) reconstruction.
*   **`viking_jax_test.py`**: Local testing suite verifying:
    *   Gradients of the custom complex absolute function.
    *   Forward pass shapes, parameter initialization, parameter counts, and overall numerical stability.
*   **`viking_parity_test.py`**: A complete, Google3-independent numerical parity verification framework containing:
    *   `numpy_viking_generator`: A pure NumPy/SciPy equivalent of the entire pipeline (framing, STFT, scanned multi-parallel LSTM, mask calculation, ISTFT, and padding).
    *   `VikingWeightTranspiler`: A conversion utility mapping PyTorch (i-f-g-o order) or TensorFlow (custom order, e.g., i-g-f-o) concatenated weights/biases directly into Flax parameter nested structures.
    *   Automated tests validating mathematical equivalence between the JAX/Flax implementation and NumPy reference to within 1e-5 absolute tolerance.
*   **`simplified_seanet.ipynb`**: Original SEANet baseline training/evaluation notebook. This serves as the reference blueprint for integrating Viking as a direct encoder/decoder/masking replacement.
*   **`requirements.txt`**: Package specifications for setting up dependencies.
*   **`README.md`**: This guide.

---

## 🚀 Getting Started

To install, run tests, or run the notebook locally:

### 1. Environment Setup

```bash
# Create and activate a local virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install standard open-source requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify the Port

Run the local test suites using the isolated environment's Python interpreter:

```bash
# Run basic model and gradient verification checks
python3 viking_jax_test.py

# Run step-by-step NumPy reference parity and transpiler verification tests
python3 viking_parity_test.py
```

The output will display:
*   Gradient correctness checks (especially complex abs gradients around zero boundaries).
*   Correct forward shapes.
*   Initialization shapes for all parallel branches, LSTM cells, and hub interconnections.
*   Verification that output values and gradients remain free of `NaN`s.
*   Passing parity assertions proving mathematical equivalence between standard NumPy references and JAX model structures (to within 1e-5 absolute tolerance).
*   Success messages for transpiling both PyTorch-style and TensorFlow-style weight matrices into Flax.

---

## 🔮 Next Steps & Discussion Points for Tomo
*   **Pipeline Integration:** Discuss swapping the core encoder/decoder blocks in the `simplified_seanet.ipynb` notebook with `viking_jax` components to benchmark SEANet vs. Viking directly.
*   **Pre-trained Weights:** Plan if/how any pre-trained google3 checkpoints should be exported and loaded into this standalone JAX format.
