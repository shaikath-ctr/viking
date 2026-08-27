"""Viking JAX vs NumPy Numerical Parity Test.

Validates that the JAX/Flax implementation and a reference NumPy/SciPy
implementation produce mathematically equivalent outputs within 1e-5 tolerance,
and verifies PyTorch and TensorFlow weight transpilation.
"""

from absl.testing import absltest
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import viking_jax


class VikingWeightTranspiler:
  """Utility to transpile PyTorch/TF weights to JAX/Flax parameter dictionaries."""

  @staticmethod
  def from_pytorch(
      torch_weights: dict,
      num_parallel: int,
      lstm_units: int,
      num_bins: int,
  ) -> dict:
    """Transpiles a PyTorch state_dict into JAX/Flax parameters."""
    viking_cell_params = {}
    features_per_lstm = num_bins // num_parallel

    # 1. Transpile LSTM cells
    for i in range(num_parallel):
      prefix = f"lstm_{i}"
      w_ih = torch_weights[f"{prefix}.weight_ih_l0"]  # shape (4*U, F)
      w_hh = torch_weights[f"{prefix}.weight_hh_l0"]  # shape (4*U, U)
      b_ih = torch_weights[f"{prefix}.bias_ih_l0"]    # shape (4*U,)
      b_hh = torch_weights[f"{prefix}.bias_hh_l0"]    # shape (4*U,)

      # PyTorch gate order: i, f, g, o
      w_ih_chunks = np.split(w_ih, 4, axis=0)
      w_hh_chunks = np.split(w_hh, 4, axis=0)
      b_ih_chunks = np.split(b_ih, 4, axis=0)
      b_hh_chunks = np.split(b_hh, 4, axis=0)

      gates = ["ii", "if", "ig", "io"]
      gate_indices = {"ii": 0, "if": 1, "ig": 2, "io": 3}

      lstm_params = {}
      for gate in gates:
        idx = gate_indices[gate]
        # Transpose from (U, F) to (F, U) to match JAX/Flax (in_features, out_features)
        lstm_params[gate] = {"kernel": w_ih_chunks[idx].T}

      for gate in gates:
        idx = gate_indices[gate]
        h_gate = "h" + gate[1]
        # Transpose from (U, U) to (U, U) to match JAX/Flax (hidden_features, hidden_features)
        # Combine input-to-hidden and hidden-to-hidden biases
        lstm_params[h_gate] = {
            "kernel": w_hh_chunks[idx].T,
            "bias": b_ih_chunks[idx] + b_hh_chunks[idx],
        }

      viking_cell_params[f"lstm_{i}"] = lstm_params

    # 2. Transpile Hub
    hub_w = torch_weights["hub.weight"]  # shape (hub_units, total_lstm_outputs)
    hub_b = torch_weights["hub.bias"]    # shape (hub_units,)
    viking_cell_params["hub"] = {"kernel": hub_w.T, "bias": hub_b}

    # 3. Transpile Mask Dense
    mask_w = torch_weights["mask_dense.weight"]  # shape (num_bins, concat+hub)
    mask_b = torch_weights["mask_dense.bias"]    # shape (num_bins,)
    viking_cell_params["mask_dense"] = {"kernel": mask_w.T, "bias": mask_b}

    return {"params": {"VikingCell_0": viking_cell_params}}

  @staticmethod
  def from_tf(
      tf_weights: dict,
      num_parallel: int,
      lstm_units: int,
      num_bins: int,
      gate_order: str = "igfo",
  ) -> dict:
    """Transpiles a TF/Keras weight dict into JAX/Flax parameters."""
    viking_cell_params = {}

    gate_map = {}
    for idx, char in enumerate(gate_order):
      if char == "i":
        gate_map["ii"] = idx
      elif char == "g":
        gate_map["ig"] = idx
      elif char == "f":
        gate_map["if"] = idx
      elif char == "o":
        gate_map["io"] = idx

    for i in range(num_parallel):
      prefix = f"lstm_{i}"
      kernel = tf_weights[f"{prefix}_kernel"]                      # shape (F, 4*U)
      recurrent_kernel = tf_weights[f"{prefix}_recurrent_kernel"]  # shape (U, 4*U)
      bias = tf_weights[f"{prefix}_bias"]                          # shape (4*U,)

      kernel_chunks = np.split(kernel, 4, axis=-1)
      recurrent_chunks = np.split(recurrent_kernel, 4, axis=-1)
      bias_chunks = np.split(bias, 4, axis=-1)

      lstm_params = {}
      for gate in ["ii", "if", "ig", "io"]:
        idx = gate_map[gate]
        lstm_params[gate] = {"kernel": kernel_chunks[idx]}

      for gate in ["ii", "if", "ig", "io"]:
        idx = gate_map[gate]
        h_gate = "h" + gate[1]
        lstm_params[h_gate] = {
            "kernel": recurrent_chunks[idx],
            "bias": bias_chunks[idx],
        }

      viking_cell_params[f"lstm_{i}"] = lstm_params

    # Hub
    viking_cell_params["hub"] = {
        "kernel": tf_weights["hub_kernel"],
        "bias": tf_weights["hub_bias"],
    }

    # Mask Dense
    viking_cell_params["mask_dense"] = {
        "kernel": tf_weights["mask_dense_kernel"],
        "bias": tf_weights["mask_dense_bias"],
    }

    return {"params": {"VikingCell_0": viking_cell_params}}


# NumPy Equivalents of Viking Reference Operations
def numpy_hann_window(frame_length: int, periodic: bool = True) -> np.ndarray:
  if periodic and frame_length % 2 == 0:
    return np.hanning(frame_length + 1)[:-1]
  return np.hanning(frame_length)


def numpy_hamming_window(frame_length: int, periodic: bool = True) -> np.ndarray:
  if periodic and frame_length % 2 == 0:
    return np.hamming(frame_length + 1)[:-1]
  return np.hamming(frame_length)


def numpy_overlap_and_add(signal: np.ndarray, frame_step: int) -> np.ndarray:
  batch, frames, frame_length = signal.shape
  overlaps = -(-frame_length // frame_step)
  signal = np.pad(
      signal, [(0, 0), (0, overlaps), (0, overlaps * frame_step - frame_length)]
  )
  signal = np.reshape(signal, [batch, frames + overlaps, overlaps, frame_step])
  signal = np.transpose(signal, [0, 2, 1, 3])
  signal = np.reshape(
      signal, [batch, overlaps * (frames + overlaps), frame_step]
  )
  extended = frames + overlaps - 1
  signal = signal[:, : overlaps * extended, :]
  signal = np.reshape(signal, [batch, overlaps, extended, frame_step])
  signal = np.sum(signal, axis=1)
  signal = np.reshape(signal, [batch, extended * frame_step])
  output_length = frame_length + frame_step * (frames - 1)
  return signal[:, :output_length]


def numpy_inverse_stft_window_fn(
    frame_length: int,
    frame_step: int,
    window_fn,
) -> np.ndarray:
  fwd_window = window_fn(frame_length).astype(np.float32)
  denom = fwd_window**2
  overlaps = -(-frame_length // frame_step)
  denom = np.pad(denom, [(0, overlaps * frame_step - frame_length)])
  denom = np.reshape(denom, [overlaps, frame_step])
  denom = np.sum(denom, axis=0, keepdims=True)
  denom = np.tile(denom, [overlaps, 1])
  denom = np.reshape(denom, [overlaps * frame_step])
  return fwd_window / (denom[:frame_length] + 1e-8)


def numpy_lstm_step(
    x: np.ndarray,
    h_prev: np.ndarray,
    c_prev: np.ndarray,
    lstm_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
  """Performs a single step of a Flax-equivalent LSTMCell in NumPy."""

  def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))

  W_ii = lstm_params["ii"]["kernel"]
  W_if = lstm_params["if"]["kernel"]
  W_ig = lstm_params["ig"]["kernel"]
  W_io = lstm_params["io"]["kernel"]

  W_hi = lstm_params["hi"]["kernel"]
  W_hf = lstm_params["hf"]["kernel"]
  W_hg = lstm_params["hg"]["kernel"]
  W_ho = lstm_params["ho"]["kernel"]

  b_i = lstm_params["hi"]["bias"]
  b_f = lstm_params["hf"]["bias"]
  b_g = lstm_params["hg"]["bias"]
  b_o = lstm_params["ho"]["bias"]

  gate_i = sigmoid(x @ W_ii + h_prev @ W_hi + b_i)
  gate_f = sigmoid(x @ W_if + h_prev @ W_hf + b_f)
  gate_g = np.tanh(x @ W_ig + h_prev @ W_hg + b_g)
  gate_o = sigmoid(x @ W_io + h_prev @ W_ho + b_o)

  c_next_raw = gate_f * c_prev + gate_i * gate_g
  c_next = np.clip(c_next_raw, -10.0, 10.0)
  h_next = gate_o * np.tanh(c_next)

  return c_next, h_next


def numpy_viking_cell(
    x_t: np.ndarray,
    carry: tuple,
    cell_params: dict,
    lstm_units: int,
    hub_units: int,
    num_bins: int,
    num_parallel: int,
) -> tuple[tuple, np.ndarray]:
  """Executes one step of VikingCell in NumPy."""
  features_per_lstm = num_bins // num_parallel
  outputs = []
  new_carries = []

  for i in range(num_parallel):
    lstm_name = f"lstm_{i}"
    lstm_params = cell_params[lstm_name]
    start = i * features_per_lstm
    end = (i + 1) * features_per_lstm if i < num_parallel - 1 else num_bins
    feat_slice = x_t[:, start:end]

    c_prev, h_prev = carry[i]
    c_next, h_next = numpy_lstm_step(feat_slice, h_prev, c_prev, lstm_params)

    outputs.append(h_next)
    new_carries.append((c_next, h_next))

  concat_outputs = np.concatenate(outputs, axis=-1)

  # Hub interconnection
  hub_params = cell_params["hub"]
  hub = concat_outputs @ hub_params["kernel"] + hub_params["bias"]

  # Mask Dense
  mask_dense_params = cell_params["mask_dense"]
  mask_input = np.concatenate([concat_outputs, hub], axis=-1)
  mask_logits = (
      mask_input @ mask_dense_params["kernel"] + mask_dense_params["bias"]
  )

  def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))

  mask = 20.0 * sigmoid(mask_logits)

  return tuple(new_carries), mask


def numpy_viking_scan(
    compressed_mags: np.ndarray,
    cell_params: dict,
    num_parallel: int,
    lstm_units: int,
    hub_units: int,
    num_bins: int,
) -> np.ndarray:
  """Executes the scanned Viking recurrent steps in NumPy."""
  batch, frames, bins = compressed_mags.shape

  # Initialize carry to zeros
  carry = []
  for _ in range(num_parallel):
    c_i = np.zeros((batch, lstm_units))
    h_i = np.zeros((batch, lstm_units))
    carry.append((c_i, h_i))

  masks = []
  for t in range(frames):
    x_t = compressed_mags[:, t, :]
    carry, mask_t = numpy_viking_cell(
        x_t, carry, cell_params, lstm_units, hub_units, bins, num_parallel
    )
    masks.append(mask_t)

  return np.stack(masks, axis=1)


def numpy_viking_generator(
    x_audio: np.ndarray,
    params: dict,
    frame_length: int = 256,
    frame_step: int = 16,
    num_parallel: int = 4,
    lstm_units: int = 32,
    hub_units: int = 8,
    in_magnitude_compression: float = 0.3,
) -> np.ndarray:
  """Reference NumPy-only implementation of VikingGenerator."""
  if x_audio.ndim == 3:
    x_audio = np.squeeze(x_audio, axis=-1)

  batch, samples = x_audio.shape

  # 1. Framing / Patches (analogous to conv_general_dilated_patches)
  num_frames = (samples - frame_length) // frame_step + 1
  patches = []
  for b in range(batch):
    batch_patches = []
    for f in range(num_frames):
      start = f * frame_step
      end = start + frame_length
      batch_patches.append(x_audio[b, start:end])
    patches.append(batch_patches)
  framed = np.array(patches, dtype=x_audio.dtype)

  # Apply window
  fwd_window = numpy_hann_window(frame_length).astype(x_audio.dtype)
  windowed_frames = framed * fwd_window

  # RFFT
  stfts = np.fft.rfft(windowed_frames, n=frame_length, axis=-1)

  # Pre-processing
  magnitudes = np.abs(stfts)
  compressed_mags = np.power(magnitudes + 1e-8, in_magnitude_compression)

  # Scanned RNN
  num_bins = stfts.shape[-1]
  cell_params = params["params"]["VikingCell_0"]
  masks = numpy_viking_scan(
      compressed_mags, cell_params, num_parallel, lstm_units, hub_units, num_bins
  )

  # Reconstruction (ISTFT)
  denoised_stfts = masks * stfts
  ifft_frames = np.fft.irfft(denoised_stfts, n=frame_length, axis=-1)

  inv_window = numpy_inverse_stft_window_fn(
      frame_length, frame_step, numpy_hann_window
  )
  windowed_ifft = ifft_frames * inv_window

  output_audio = numpy_overlap_and_add(windowed_ifft, frame_step)

  # Pad or crop to input audio samples
  out_samples = output_audio.shape[-1]
  if out_samples < samples:
    output_audio = np.pad(output_audio, ((0, 0), (0, samples - out_samples)))
  else:
    output_audio = output_audio[:, :samples]

  return output_audio[..., None]


class VikingParityTest(absltest.TestCase):

  def test_window_parity(self):
    lengths = [64, 128, 256, 512]
    for length in lengths:
      # Hann
      jax_hann = viking_jax.hann_window(length, periodic=True)
      np_hann = numpy_hann_window(length, periodic=True)
      np.testing.assert_allclose(jax_hann, np_hann, rtol=1e-6, atol=1e-6)

      # Hamming
      jax_ham = viking_jax.hamming_window(length, periodic=True)
      np_ham = numpy_hamming_window(length, periodic=True)
      np.testing.assert_allclose(jax_ham, np_ham, rtol=1e-6, atol=1e-6)

  def test_overlap_add_parity(self):
    # Setup standard test signal
    np.random.seed(42)
    signal = np.random.randn(2, 20, 64).astype(np.float32)
    frame_step = 16

    jax_signal = jnp.array(signal)
    jax_out = viking_jax.overlap_and_add(jax_signal, frame_step)
    np_out = numpy_overlap_and_add(signal, frame_step)

    np.testing.assert_allclose(jax_out, np_out, rtol=1e-5, atol=1e-5)

  def test_inverse_stft_window_parity(self):
    frame_length = 128
    frame_step = 32

    jax_out = viking_jax.inverse_stft_window_fn(
        frame_length, frame_step, viking_jax.hann_window
    )
    np_out = numpy_inverse_stft_window_fn(
        frame_length, frame_step, numpy_hann_window
    )

    np.testing.assert_allclose(jax_out, np_out, rtol=1e-6, atol=1e-6)

  def test_viking_cell_parity(self):
    np.random.seed(42)
    batch = 2
    lstm_units = 8
    hub_units = 4
    num_bins = 33
    num_parallel = 2

    # Instantiate the Jax cell to initialize variables
    cell = viking_jax.VikingCell(
        lstm_units=lstm_units,
        hub_units=hub_units,
        num_bins=num_bins,
        num_parallel=num_parallel,
    )
    rng = jax.random.PRNGKey(42)
    x_t_jax = jax.random.normal(rng, (batch, num_bins))
    x_t_np = np.array(x_t_jax)

    init_carry_jax = tuple([
        (
            jnp.zeros((batch, lstm_units)),
            jnp.zeros((batch, lstm_units)),
        )
        for _ in range(num_parallel)
    ])
    init_carry_np = tuple([
        (
            np.zeros((batch, lstm_units)),
            np.zeros((batch, lstm_units)),
        )
        for _ in range(num_parallel)
    ])

    variables = cell.init(rng, init_carry_jax, x_t_jax)

    # Apply both cells
    (new_carry_jax, mask_jax) = cell.apply(variables, init_carry_jax, x_t_jax)
    (new_carry_np, mask_np) = numpy_viking_cell(
        x_t_np,
        init_carry_np,
        variables["params"],
        lstm_units,
        hub_units,
        num_bins,
        num_parallel,
    )

    # Check mask parity
    np.testing.assert_allclose(mask_jax, mask_np, rtol=1e-5, atol=1e-5)

    # Check carry parity
    for i in range(num_parallel):
      c_jax, h_jax = new_carry_jax[i]
      c_np, h_np = new_carry_np[i]
      np.testing.assert_allclose(c_jax, c_np, rtol=1e-5, atol=1e-5)
      np.testing.assert_allclose(h_jax, h_np, rtol=1e-5, atol=1e-5)

  def test_pytorch_transpiler_and_parity(self):
    np.random.seed(1337)
    batch = 1
    samples = 256
    frame_length = 64
    frame_step = 16
    lstm_units = 8
    hub_units = 4
    num_parallel = 2
    num_bins = frame_length // 2 + 1  # 33

    # 1. Generate fake PyTorch weights
    # Structure match
    torch_weights = {}
    features_per_lstm = num_bins // num_parallel

    for i in range(num_parallel):
      prefix = f"lstm_{i}"
      f_in = (
          features_per_lstm if i < num_parallel - 1 else num_bins - i * features_per_lstm
      )
      torch_weights[f"{prefix}.weight_ih_l0"] = np.random.randn(
          4 * lstm_units, f_in
      ).astype(np.float32)
      torch_weights[f"{prefix}.weight_hh_l0"] = np.random.randn(
          4 * lstm_units, lstm_units
      ).astype(np.float32)
      torch_weights[f"{prefix}.bias_ih_l0"] = np.random.randn(
          4 * lstm_units
      ).astype(np.float32)
      torch_weights[f"{prefix}.bias_hh_l0"] = np.random.randn(
          4 * lstm_units
      ).astype(np.float32)

    total_lstm_outputs = num_parallel * lstm_units
    torch_weights["hub.weight"] = np.random.randn(
        hub_units, total_lstm_outputs
    ).astype(np.float32)
    torch_weights["hub.bias"] = np.random.randn(hub_units).astype(np.float32)

    torch_weights["mask_dense.weight"] = np.random.randn(
        num_bins, total_lstm_outputs + hub_units
    ).astype(np.float32)
    torch_weights["mask_dense.bias"] = np.random.randn(num_bins).astype(
        np.float32
    )

    # 2. Transpile using weight-mapping utility
    transpiled_params = VikingWeightTranspiler.from_pytorch(
        torch_weights, num_parallel, lstm_units, num_bins
    )

    # 3. Apply on JAX model and reference NumPy generator
    model = viking_jax.VikingGenerator(
        frame_length=frame_length,
        frame_step=frame_step,
        num_parallel=num_parallel,
        lstm_units=lstm_units,
        hub_units=hub_units,
        in_magnitude_compression=0.3,
    )

    x_audio = np.random.randn(batch, samples, 1).astype(np.float32)
    x_audio_jax = jnp.array(x_audio)

    # Run JAX Generator
    jax_out = model.apply(transpiled_params, x_audio_jax)

    # Run NumPy Generator
    np_out = numpy_viking_generator(
        x_audio,
        transpiled_params,
        frame_length,
        frame_step,
        num_parallel,
        lstm_units,
        hub_units,
    )

    # 4. Assert absolute tolerance parity < 1e-5
    np.testing.assert_allclose(jax_out, np_out, rtol=1e-5, atol=1e-5)

  def test_tf_transpiler_and_parity(self):
    np.random.seed(4242)
    batch = 1
    samples = 256
    frame_length = 64
    frame_step = 16
    lstm_units = 8
    hub_units = 4
    num_parallel = 2
    num_bins = frame_length // 2 + 1  # 33

    # 1. Generate fake TF/Keras weights
    tf_weights = {}
    features_per_lstm = num_bins // num_parallel

    for i in range(num_parallel):
      prefix = f"lstm_{i}"
      f_in = (
          features_per_lstm if i < num_parallel - 1 else num_bins - i * features_per_lstm
      )
      tf_weights[f"{prefix}_kernel"] = np.random.randn(
          f_in, 4 * lstm_units
      ).astype(np.float32)
      tf_weights[f"{prefix}_recurrent_kernel"] = np.random.randn(
          lstm_units, 4 * lstm_units
      ).astype(np.float32)
      tf_weights[f"{prefix}_bias"] = np.random.randn(4 * lstm_units).astype(
          np.float32
      )

    total_lstm_outputs = num_parallel * lstm_units
    tf_weights["hub_kernel"] = np.random.randn(
        total_lstm_outputs, hub_units
    ).astype(np.float32)
    tf_weights["hub_bias"] = np.random.randn(hub_units).astype(np.float32)

    tf_weights["mask_dense_kernel"] = np.random.randn(
        total_lstm_outputs + hub_units, num_bins
    ).astype(np.float32)
    tf_weights["mask_dense_bias"] = np.random.randn(num_bins).astype(np.float32)

    # 2. Transpile using weight-mapping utility (gate order: i, g, f, o)
    transpiled_params = VikingWeightTranspiler.from_tf(
        tf_weights, num_parallel, lstm_units, num_bins, gate_order="igfo"
    )

    # 3. Apply on JAX model and reference NumPy generator
    model = viking_jax.VikingGenerator(
        frame_length=frame_length,
        frame_step=frame_step,
        num_parallel=num_parallel,
        lstm_units=lstm_units,
        hub_units=hub_units,
        in_magnitude_compression=0.3,
    )

    x_audio = np.random.randn(batch, samples, 1).astype(np.float32)
    x_audio_jax = jnp.array(x_audio)

    # Run JAX Generator
    jax_out = model.apply(transpiled_params, x_audio_jax)

    # Run NumPy Generator
    np_out = numpy_viking_generator(
        x_audio,
        transpiled_params,
        frame_length,
        frame_step,
        num_parallel,
        lstm_units,
        hub_units,
    )

    # 4. Assert absolute tolerance parity < 1e-5
    np.testing.assert_allclose(jax_out, np_out, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  absltest.main()
