from absl import app
import jax
import jax.numpy as jnp
import viking_jax

jax.config.update("jax_disable_jit", False)


def test_complex_abs_grad():
  def f(x):
    return jnp.sum(jnp.abs(x))

  x = jnp.array([0.0 + 0.0j, 1.0 + 0.0j])
  grad_f = jax.grad(f)(x)
  print(f"Grad at 0 (complex): {grad_f}")
  print(f"Is NaN: {jnp.any(jnp.isnan(grad_f))}")

  print("=== Real Abs Grad at 0 ===")
  x_real = jnp.array([0.0, 1.0])
  grad_f_real = jax.grad(f)(x_real)
  print(f"Grad at 0 (real): {grad_f_real}")
  print(f"Is NaN: {jnp.any(jnp.isnan(grad_f_real))}")
  print("=============================")


def test_viking():
  print("Starting full VikingGenerator test...")
  model = viking_jax.VikingGenerator(
      frame_length=64, frame_step=16, lstm_units=8, hub_units=4, num_parallel=2
  )
  batch = 1
  samples = 256  # multiple frames

  # Generate random noise
  key = jax.random.PRNGKey(0)
  x = jax.random.normal(key, (batch, samples, 1))
  rng = jax.random.PRNGKey(1)

  print("Initializing VikingGenerator variables...")
  variables = model.init(rng, x, train=False)
  print("Initialization successful.")
  print(
      "Viking Params Shapes:"
      f" {jax.tree_util.tree_map(lambda x: x.shape, variables)}"
  )

  print("Applying VikingGenerator...")
  out = model.apply(variables, x, train=False)
  print("Apply successful.")

  print(f"Output shape: {out.shape}")
  print(f"Output mean: {jnp.mean(out)}")
  print(f"Output max: {jnp.max(out)}")
  print(f"Output min: {jnp.min(out)}")

  print("Full VikingGenerator test passed!")


def main(argv):
  del argv
  jax.config.update("jax_debug_nans", True)
  test_complex_abs_grad()
  test_viking()


if __name__ == "__main__":
  app.run(main)
