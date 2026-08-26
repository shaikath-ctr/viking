"""Viking JAX Class Definition.

Simplified JAX/Flax implementation of the Viking model.
"""

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp


def hann_window(frame_length: int, periodic: bool = True) -> jnp.ndarray:
  """Hanning window function."""
  if periodic and frame_length % 2 == 0:
    return jnp.hanning(frame_length + 1)[:-1]
  return jnp.hanning(frame_length)


def hamming_window(frame_length: int, periodic: bool = True) -> jnp.ndarray:
  """Hamming window function."""
  if periodic and frame_length % 2 == 0:
    return jnp.hamming(frame_length + 1)[:-1]
  return jnp.hamming(frame_length)


def overlap_and_add(signal: jnp.ndarray, frame_step: int) -> jnp.ndarray:
  """Overlap and add implementation in JAX."""
  batch, frames, frame_length = signal.shape
  overlaps = -(-frame_length // frame_step)
  signal = jnp.pad(
      signal, [(0, 0), (0, overlaps), (0, overlaps * frame_step - frame_length)]
  )
  signal = jnp.reshape(signal, [batch, frames + overlaps, overlaps, frame_step])
  signal = jnp.transpose(signal, [0, 2, 1, 3])
  signal = jnp.reshape(
      signal, [batch, overlaps * (frames + overlaps), frame_step]
  )
  extended = frames + overlaps - 1
  signal = signal[:, : overlaps * extended, :]
  signal = jnp.reshape(signal, [batch, overlaps, extended, frame_step])
  signal = jnp.sum(signal, axis=1)
  signal = jnp.reshape(signal, [batch, extended * frame_step])
  output_length = frame_length + frame_step * (frames - 1)
  return signal[:, :output_length]


def inverse_stft_window_fn(
    frame_length: int,
    frame_step: int,
    window_fn: Callable[[int], jnp.ndarray],
) -> jnp.ndarray:
  """Normalizes window for inverse STFT."""
  fwd_window = window_fn(frame_length).astype(jnp.float32)
  denom = fwd_window**2
  overlaps = -(-frame_length // frame_step)
  denom = jnp.pad(denom, [(0, overlaps * frame_step - frame_length)])
  denom = jnp.reshape(denom, [overlaps, frame_step])
  denom = jnp.sum(denom, axis=0, keepdims=True)
  denom = jnp.tile(denom, [overlaps, 1])
  denom = jnp.reshape(denom, [overlaps * frame_step])
  return fwd_window / (denom[:frame_length] + 1e-8)


class VikingCell(nn.Module):
  """Submodule for a single Viking step."""

  lstm_units: int
  hub_units: int
  num_bins: int
  num_parallel: int

  @nn.compact
  def __call__(self, carry, x_t):
    # carry: tuple of (c, h) per LSTM
    # x_t: [batch, num_bins]
    features_per_lstm = self.num_bins // self.num_parallel
    outputs = []
    new_carries = []
    print(f'VikingCell __call__ start. num_parallel: {self.num_parallel}')
    for i in range(self.num_parallel):
      print(f'VikingCell processing parallel branch {i}')
      start = i * features_per_lstm
      end = (
          (i + 1) * features_per_lstm
          if i < self.num_parallel - 1
          else self.num_bins
      )
      feat_slice = x_t[:, start:end]
      (next_c, next_h), res = nn.LSTMCell(self.lstm_units, name=f'lstm_{i}')(
          carry[i], feat_slice
      )
      outputs.append(res)
      # We want to clip the cell state to prevent explosion.
      # Use min/max formulation to potentially avoid gradient issues at boundaries/infinity
      next_c_clipped = jnp.minimum(jnp.maximum(next_c, -10.0), 10.0)
      new_carries.append((next_c_clipped, next_h))

    # Hub interconnection
    concat_outputs = jnp.concatenate(outputs, axis=-1)
    hub = nn.Dense(self.hub_units, name='hub')(concat_outputs)

    # Final dense for mask logits (initialized to start at gain = 1.0)
    mask_logits = nn.Dense(
        self.num_bins,
        bias_init=nn.initializers.constant(-2.9444),
        name='mask_dense',
    )(jnp.concatenate([concat_outputs, hub], axis=-1))
    mask = 20.0 * nn.sigmoid(mask_logits)

    return tuple(new_carries), mask


class VikingGenerator(nn.Module):
  """Viking Model JAX Implementation."""

  frame_length: int = 256
  frame_step: int = 16
  num_parallel: int = 4
  lstm_units: int = 32
  hub_units: int = 8
  in_magnitude_compression: float = 0.3

  @nn.compact
  def __call__(self, x_audio, train: bool = False):
    # x_audio shape: [batch, samples, 1]
    if x_audio.ndim == 3:
      x_audio = jnp.squeeze(x_audio, axis=-1)

    batch, samples = x_audio.shape

    # 1. STFT
    framed = jax.lax.conv_general_dilated_patches(
        x_audio[..., None],
        (self.frame_length,),
        (self.frame_step,),
        padding='VALID',
        dimension_numbers=('NTC', 'OIT', 'NTC'),
    )

    # Applying window
    window = hann_window(self.frame_length).astype(x_audio.dtype)
    windowed_frames = framed * window

    stfts = jnp.fft.rfft(windowed_frames, self.frame_length)
    jax.debug.print('=== VikingGenerator Debug ===')
    jax.debug.print('stfts has NaN: {}', jnp.any(jnp.isnan(stfts)))

    # 2. Pre-processing
    magnitudes = jnp.abs(stfts)
    compressed_mags = jnp.power(
        magnitudes + 1e-8, self.in_magnitude_compression
    )
    jax.debug.print(
        'compressed_mags has NaN: {}', jnp.any(jnp.isnan(compressed_mags))
    )

    # 3. Core RNN Model (Scanned)
    num_bins = compressed_mags.shape[-1]

    init_carry = tuple([
        (
            jnp.zeros((batch, self.lstm_units)),
            jnp.zeros((batch, self.lstm_units)),
        )
        for _ in range(self.num_parallel)
    ])

    viking_cell = VikingCell(
        lstm_units=self.lstm_units,
        hub_units=self.hub_units,
        num_bins=num_bins,
        num_parallel=self.num_parallel,
    )

    def body_fn(cell, carry, x):
      return cell(carry, x)

    # nn.scan(body_fn, ...).
    # scanning happens over arguments starting from the 3rd argument (x).
    # carry is the 2nd argument. cell is the 1st argument (Module).
    viking_scan = nn.scan(
        body_fn,
        variable_broadcast='params',
        split_rngs={'params': False},
        in_axes=1,  # Refers to the 'x' argument
        out_axes=1,
    )

    _, masks = viking_scan(viking_cell, init_carry, compressed_mags)
    jax.debug.print('masks has NaN: {}', jnp.any(jnp.isnan(masks)))

    # 4. Reconstruct
    denoised_stfts = masks * stfts

    # ISTFT
    ifft_frames = jnp.fft.irfft(denoised_stfts, self.frame_length)

    # Applying inverse window
    inv_window = inverse_stft_window_fn(
        self.frame_length, self.frame_step, hann_window
    )
    windowed_ifft = ifft_frames * inv_window

    output_audio = overlap_and_add(windowed_ifft, self.frame_step)
    jax.debug.print(
        'output_audio has NaN: {}', jnp.any(jnp.isnan(output_audio))
    )
    jax.debug.print('=============================')

    out_samples = output_audio.shape[-1]
    if out_samples < samples:
      output_audio = jnp.pad(output_audio, ((0, 0), (0, samples - out_samples)))
    else:
      output_audio = output_audio[:, :samples]

    return output_audio[..., None]
