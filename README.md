# Viking JAX Implementation

This repository contains a simplified, open-source JAX/Flax implementation of the Viking model.

Viking is a neural audio processing model designed as an efficient, lightweight replacement for models like SEANet. This port makes the JAX/Flax version of Viking completely accessible outside of Google's internal `google3` environment, ready for integration into open-source machine learning pipelines.

## Project Structure

*   `viking_jax.py`: The core JAX/Flax model implementation, defining `VikingCell` (recurrent cell with parallel branches and hub interconnect) and `VikingGenerator` (performs STFT, applies the Viking model mask, and performs inverse STFT).
*   `viking_jax_test.py`: A comprehensive test suite to verify the model's forward pass, numerical stability (NaN checks), parameter shapes, and gradient computation.
*   `simplified_seanet.ipynb`: A Jupyter Notebook containing training and evaluation of the baseline SEANet model, demonstrating how Viking can be integrated as a replacement.
*   `requirements.txt`: Python package dependencies.

## Installation

To set up a local environment and install the required dependencies:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Running Tests

To run the model's test suite and verify everything is working correctly:

```bash
# Make sure your virtual environment is active
python3 viking_jax_test.py
```

## License

This project is licensed under the Apache 2.0 License.
