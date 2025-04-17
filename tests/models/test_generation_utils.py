import jax
import jax.numpy as jnp
import pytest
from jax import random

from jaxgarden.models.generation_utils import GenerationMixin

# --- Mock Model for GenerationMixin Tests ---


class MockModel(GenerationMixin):
    """A simple mock model inheriting GenerationMixin for testing."""

    def __init__(self, vocab_size, eos_token_id=None, pad_token_id=0):
        # Mock config
        self.config = {
            "vocab_size": vocab_size,
            "eos_token_id": eos_token_id,
            "pad_token_id": pad_token_id,
        }
        self.vocab_size = vocab_size
        self._eos_token_id = eos_token_id  # Store separately if needed

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        """Mock call that returns deterministic logits."""
        batch_size, seq_len = input_ids.shape
        if attention_mask is None:
            # Fallback: If no mask, assume the sequence length is the full length
            current_length = jnp.array(seq_len)
        else:
            # Determine current length from attention mask
            # Handle 1D [max_length] or 2D [batch, max_length] masks
            if attention_mask.ndim == 1:
                current_length = jnp.sum(attention_mask)
            elif attention_mask.ndim == 2:
                # Ensure current_length is scalar if attention_mask is [1, seq_len]
                if attention_mask.shape[0] == 1:
                    current_length = jnp.sum(attention_mask[0])
                else:  # Handle [batch, seq_len]
                    current_length = jnp.sum(attention_mask, axis=-1)
            else:
                raise ValueError(f"Unexpected attention_mask ndim: {attention_mask.ndim}")

        # Ensure length is at least 1 to avoid index -1
        valid_length = jnp.maximum(1, current_length)
        last_token_index = valid_length - 1  # Index of the last valid token

        # Gather the last valid token for each item in the batch
        # Needs to handle scalar or per-batch indices
        if isinstance(last_token_index, int) or (
            hasattr(last_token_index, "shape") and last_token_index.shape == ()
        ):
            # If current_length (and thus index) is scalar across batch
            last_token = input_ids[:, last_token_index]  # Shape [batch_size]
        elif hasattr(last_token_index, "shape") and last_token_index.ndim == 1:
            # If current_length varies per batch item (shape [batch_size])
            # Use gather to select the token at last_token_index for each batch item
            last_token_index_expanded = last_token_index[:, None]  # Shape [batch_size, 1]
            last_token = jnp.take_along_axis(input_ids, last_token_index_expanded, axis=1)[:, 0]
        else:
            raise TypeError(f"""Unexpected type or shape for last_token_index:
            {type(last_token_index)}""")

        # Create deterministic logits: next token is always (last_token + 1) % vocab_size
        next_token_logits = (
            jax.nn.one_hot(
                (last_token + 1) % self.vocab_size, num_classes=self.vocab_size, dtype=jnp.float32
            )
            * 10.0
        )  # Multiply to make it peaky
        # Return shape [batch_size, seq_len, vocab_size]
        # We only care about the last token logits for generation,
        # so we just broadcast it for simplicity in this mock.
        # A real model would compute logits based on the full sequence.
        return jnp.repeat(next_token_logits[:, None, :], seq_len, axis=1)


# --- Tests for GenerationMixin ---


@pytest.fixture
def generation_setup():
    key = random.PRNGKey(42)
    vocab_size = 10
    eos_token_id = 9
    pad_token_id = 0
    model = MockModel(vocab_size, eos_token_id, pad_token_id)
    input_ids = jnp.array([[1, 2], [5, 6]], dtype=jnp.int32)
    return key, model, input_ids, vocab_size, eos_token_id, pad_token_id


def test_generate_greedy(generation_setup):
    """Tests greedy generation (do_sample=False)."""
    key, model, input_ids, _, eos_token_id, pad_token_id = generation_setup
    max_length = 7

    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=False,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    assert output_ids.shape == (input_ids.shape[0], max_length)

    # Expected sequence based on mock model: next = (last + 1) % vocab_size
    # Batch 1: [1, 2] -> 3 -> 4 -> 5 -> 6 -> 7
    # Batch 2: [5, 6] -> 7 -> 8 -> 9 (EOS) -> 0 (PAD) -> 0 (PAD)
    expected_output = jnp.array([[1, 2, 3, 4, 5, 6, 7], [5, 6, 7, 8, 9, 0, 0]], dtype=jnp.int32)
    assert jnp.array_equal(output_ids, expected_output)


def test_generate_sampling(generation_setup):
    """Tests sampling generation (do_sample=True)."""
    key, model, input_ids, vocab_size, eos_token_id, pad_token_id = generation_setup
    max_length = 6

    # Use a very low temperature to make it nearly deterministic for testing
    key, subkey = random.split(key)
    output_ids_low_temp = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=0.01,  # Very low temp
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        rng=subkey,
    )

    # Expected sequence (likely, due to low temp):
    # Batch 1: [1, 2] -> 3 -> 4 -> 5 -> 6
    # Batch 2: [5, 6] -> 7 -> 8 -> 9 (EOS) -> 0 (PAD)
    expected_output_likely = jnp.array([[1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 0]], dtype=jnp.int32)
    assert output_ids_low_temp.shape == (input_ids.shape[0], max_length)
    # Check it's highly likely the same as greedy due to low temp
    assert jnp.array_equal(output_ids_low_temp, expected_output_likely)

    # Test reproducibility with the same key
    key, subkey = random.split(key)
    output_ids1 = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=1.0,
        rng=subkey,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    output_ids2 = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=1.0,
        rng=subkey,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    assert jnp.array_equal(output_ids1, output_ids2)


def test_generate_max_length(generation_setup):
    """Tests that generation stops at max_length."""
    key, model, input_ids, _, eos_token_id, pad_token_id = generation_setup
    max_length = 4  # Shorter than needed to reach EOS for batch 1

    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=False,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    assert output_ids.shape == (input_ids.shape[0], max_length)
    # Expected sequence (truncated):
    # Batch 1: [1, 2] -> 3 -> 4
    # Batch 2: [5, 6] -> 7 -> 8
    expected_output = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
    assert jnp.array_equal(output_ids, expected_output)


def test_generate_eos_handling(generation_setup):
    """Tests EOS token handling and subsequent padding."""
    key, model, input_ids, _, eos_token_id, pad_token_id = generation_setup
    max_length = 8  # Long enough for EOS

    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=False,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    assert output_ids.shape == (input_ids.shape[0], max_length)
    # Expected sequence:
    # Batch 1: [1, 2] -> 3 -> 4 -> 5 -> 6 -> 7 -> 8
    # Batch 2: [5, 6] -> 7 -> 8 -> 9 (EOS) -> 0 (PAD) -> 0 (PAD) -> 0 (PAD)
    expected_output = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 7, 8, 9, 0, 0, 0]], dtype=jnp.int32
    )
    assert jnp.array_equal(output_ids, expected_output)

    # Test case without EOS token (should just fill to max_length)
    model_no_eos = MockModel(model.vocab_size, eos_token_id=None, pad_token_id=pad_token_id)
    output_ids_no_eos = model_no_eos.generate(
        input_ids,
        max_length=max_length,
        do_sample=False,
        eos_token_id=None,
        pad_token_id=pad_token_id,
    )
    expected_output_no_eos = jnp.array(
        [[1, 2, 3, 4, 5, 6, 7, 8], [5, 6, 7, 8, 9, 0, 1, 2]],  # Continues sequence cyclically
        dtype=jnp.int32,
    )
    assert jnp.array_equal(output_ids_no_eos, expected_output_no_eos)


def test_generate_padding_default(generation_setup):
    """Tests generation using default pad_token_id from config."""
    key, model, input_ids, _, eos_token_id, pad_token_id = generation_setup
    max_length = 7

    # Don't pass pad_token_id, should use model.config['pad_token_id'] (which is 0)
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=False,
        eos_token_id=eos_token_id,
        # pad_token_id not provided
    )
    # Expected is same as test_generate_greedy
    expected_output = jnp.array([[1, 2, 3, 4, 5, 6, 7], [5, 6, 7, 8, 9, 0, 0]], dtype=jnp.int32)
    assert jnp.array_equal(output_ids, expected_output)


def test_generate_input_length_equals_max_length(generation_setup):
    """Tests generation when input_ids length equals max_length."""
    key, model, input_ids, _, eos_token_id, pad_token_id = generation_setup
    max_length = input_ids.shape[1]  # 2

    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=False,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    # Should return the input_ids unchanged
    assert output_ids.shape == (input_ids.shape[0], max_length)
    assert jnp.array_equal(output_ids, input_ids)


def test_generate_input_length_greater_than_max_length(generation_setup):
    """Tests generation when input_ids length exceeds max_length."""
    key, model, input_ids, _, eos_token_id, pad_token_id = generation_setup
    max_length = 1

    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=False,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    # Should return input_ids truncated to max_length
    assert output_ids.shape == (input_ids.shape[0], max_length)
    assert jnp.array_equal(output_ids, input_ids[:, :max_length])


def test_generate_sampling_with_filtering(generation_setup):
    """Tests sampling with filtering parameters (top_k, top_p, min_p)."""
    key, model, input_ids, vocab_size, eos_token_id, pad_token_id = generation_setup
    max_length = 6

    # Test with top_k
    key, subkey = random.split(key)
    output_ids_top_k = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=3,
        rng=subkey,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    assert output_ids_top_k.shape == (input_ids.shape[0], max_length)

    # Test with top_p
    key, subkey = random.split(key)
    output_ids_top_p = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_p=0.8,
        rng=subkey,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    assert output_ids_top_p.shape == (input_ids.shape[0], max_length)

    # Test with min_p
    key, subkey = random.split(key)
    output_ids_min_p = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        min_p=0.1,
        rng=subkey,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    assert output_ids_min_p.shape == (input_ids.shape[0], max_length)

    # Test with multiple filtering methods
    key, subkey = random.split(key)
    output_ids_combo = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=3,
        top_p=0.8,
        min_p=0.1,
        rng=subkey,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    assert output_ids_combo.shape == (input_ids.shape[0], max_length)


def test_generate_rng_none_warning(generation_setup, capsys):
    """Tests that a warning is printed when do_sample is True but rng is None."""
    key, model, input_ids, _, eos_token_id, pad_token_id = generation_setup
    max_length = 5

    model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        rng=None,  # No RNG key provided
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    captured = capsys.readouterr()
    assert "Warning: No RNG key provided for sampling, using default key 0." in captured.out


def test_generate_no_sample_rng_none(generation_setup):
    """Tests that do_sample=False (greedy) works without rng."""
    key, model, input_ids, _, eos_token_id, pad_token_id = generation_setup
    max_length = 5
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=False,
        rng=None,  # No RNG key provided
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    assert output_ids.shape == (input_ids.shape[0], max_length)


def test_generate_jax_compile():
    """Tests that jax.jit can be used with generate without error."""
    vocab_size = 10
    eos_token_id = 9
    pad_token_id = 0
    model = MockModel(vocab_size, eos_token_id, pad_token_id)
    input_ids = jnp.array([[1, 2], [5, 6]], dtype=jnp.int32)
    max_length = 7

    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=False,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_jit=True,
    )

    assert output_ids.shape == (input_ids.shape[0], max_length)

    # Expected sequence based on mock model: next = (last + 1) % vocab_size
    # Batch 1: [1, 2] -> 3 -> 4 -> 5 -> 6 -> 7
    # Batch 2: [5, 6] -> 7 -> 8 -> 9 (EOS) -> 0 (PAD) -> 0 (PAD)
    expected_output = jnp.array([[1, 2, 3, 4, 5, 6, 7], [5, 6, 7, 8, 9, 0, 0]], dtype=jnp.int32)
    assert jnp.array_equal(output_ids, expected_output)  # Should match greedy output

    # Test that jax.jit does not affect the output
    output_ids2 = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=False,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_jit=True,
    )
    assert jnp.array_equal(output_ids, output_ids2)
