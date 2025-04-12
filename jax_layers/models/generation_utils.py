import jax
import jax.numpy as jnp
from flax import nnx

# Constants for numerical stability
EPSILON = 1e-9


def temperature_scale(logits: jnp.ndarray, temperature: float) -> jnp.ndarray:
    """Scales logits by temperature.

    Args:
        logits: Logits to scale. Shape: (..., vocab_size)
        temperature: Temperature value. Higher values make the distribution flatter (more random),
                     lower values make it peakier (more deterministic). Must be positive.

    Returns:
        Scaled logits.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    # Prevent division by zero by adding a small epsilon if temperature is zero
    safe_temperature = max(temperature, EPSILON)
    return logits / safe_temperature


def top_k_logits(logits: jnp.ndarray, k: int) -> jnp.ndarray:
    """Masks logits outside the top k values.

    Sets logits not in the top k to negative infinity.

    Args:
        logits: Logits to filter. Shape: (..., vocab_size)
        k: Number of top logits to keep.

    Returns:
        Filtered logits.
    """
    if k <= 0:
        # If k is 0 or negative, mask all logits
        return jnp.full_like(logits, -jnp.inf)

    # Ensure k is not larger than the vocabulary size
    k = min(k, logits.shape[-1])

    # Get top-k values
    top_k_values = jax.lax.top_k(logits, k=k)[0]
    kth_value = top_k_values[..., -1:]

    # Create a mask where logits >= kth_value are True
    mask = logits >= kth_value

    # Set logits below the threshold to -inf
    return jnp.where(mask, logits, -jnp.inf)


def top_p_logits(logits: jnp.ndarray, p: float) -> jnp.ndarray:
    """Filter logits using nucleus (top-p) sampling.

    Args:
        logits: Shape (..., vocab_size)
        p: Probability threshold (0 < p <= 1)

    Returns:
        Filtered logits with -inf for tokens outside the top-p nucleus
    """
    if not 0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")
    if p == 1.0:
        return logits

    # Convert to probabilities
    probs = nnx.softmax(logits, axis=-1)

    # Sort probabilities in descending order
    sorted_probs = jnp.sort(probs, axis=-1)[..., ::-1]

    # Calculate cumulative probabilities and create mask
    cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
    sorted_mask = cumulative_probs <= p

    # Always include at least the top token
    sorted_mask = sorted_mask.at[..., 0].set(True)

    # Find the minimum probability within the nucleus
    threshold = jnp.min(
        jnp.where(sorted_mask, sorted_probs, jnp.ones_like(sorted_probs)), axis=-1, keepdims=True
    )

    # Apply threshold to original probabilities
    # Keep tokens whose probability is >= threshold
    mask = probs >= threshold

    # Apply mask to logits
    return jnp.where(mask, logits, -jnp.inf)


def min_p_logits(logits: jnp.ndarray, p: float) -> jnp.ndarray:
    """Masks logits below a probability threshold derived from the max probability (min_p sampling).
    Filters out tokens with probability less than p * max_probability.

    Args:
        logits: Logits to filter. Shape: (..., vocab_size)
        p: Probability threshold factor (0 < p <= 1).

    Returns:
        Filtered logits.
    """
    if not 0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")

    probs = nnx.softmax(logits, axis=-1)
    max_prob = jnp.max(probs, axis=-1, keepdims=True)
    threshold = max_prob * p

    # Identify indices corresponding to max probability
    max_prob_indices = probs >= (max_prob - EPSILON)

    if p == 1.0:
        # When p=1.0, keep just the max probability tokens
        mask = ~max_prob_indices
    else:
        # Otherwise, keep max prob tokens and tokens above the threshold
        mask_below_threshold = probs < threshold
        mask = jnp.where(max_prob_indices, False, mask_below_threshold)

    # Apply the mask to the original logits
    return jnp.where(mask, -jnp.inf, logits)


def sample_logits(
    logits: jnp.ndarray,
    rng_key: jax.random.PRNGKey,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    do_sample: bool = True,
) -> jnp.ndarray:
    """Samples a token index from logits using specified filtering and temperature.

    Applies filtering methods (top_k, top_p, min_p) and temperature scaling,
    then samples from the resulting distribution or takes the argmax.

    Args:
        logits: Raw logits from the model. Shape: (..., vocab_size)
        rng_key: JAX PRNG key for sampling.
        temperature: Temperature scaling factor.
        top_k: If set, keep only top k logits.
        top_p: If set, keep smallest set of logits whose cumulative probability exceeds p.
        min_p: If set, keep logits with probability >= max_prob * p.
        do_sample: If True, sample using categorical distribution.
                    If False, take argmax (greedy decoding).

    Returns:
        Sampled token indices. Shape: (...)
    """
    if not do_sample:
        # Greedy decoding
        return jnp.argmax(logits, axis=-1)

    # 1. Apply temperature scaling
    if temperature != 1.0 and temperature > 0:
        scaled_logits = temperature_scale(logits, temperature)
    else:
        scaled_logits = logits

    # Store the scaled logits as the potential fallback
    logits_for_fallback = scaled_logits

    # 2. Apply filtering
    filtered_logits = scaled_logits
    # Apply filtering in a specific order (min_p -> top_k -> top_p is one common order)
    # Note: The order can matter. Min_p focuses on dynamic range,
    # while top_k/top_p on absolute ranks/mass.
    if min_p is not None and 0 < min_p < 1.0:
        filtered_logits = min_p_logits(filtered_logits, min_p)
    if top_k is not None and top_k > 0:
        filtered_logits = top_k_logits(filtered_logits, top_k)
    if top_p is not None and 0 < top_p < 1.0:  # top_p=1 means no filtering
        filtered_logits = top_p_logits(filtered_logits, top_p)

    # 3. Sample or take argmax, handling the edge case for sampling

    all_filtered_infinite = jnp.all(filtered_logits == -jnp.inf, axis=-1, keepdims=True)

    # Determine the logits to actually sample from:
    # Use the fallback (scaled, unfiltered) if all filtered are -inf
    final_logits_for_sampling = jnp.where(
        all_filtered_infinite,
        logits_for_fallback,  # Fallback to pre-filter (but post-temp) logits
        filtered_logits,  # Otherwise, use the filtered logits
    )

    # Sample using the chosen logits
    sampled_indices = jax.random.categorical(rng_key, final_logits_for_sampling, axis=-1)

    return sampled_indices


def create_causal_mask(seq_len: int) -> jnp.ndarray:
    """Creates a causal attention mask for a given sequence length.

    Args:
        seq_len: The length of the sequence.

    Returns:
        A causal attention mask of shape [seq_len, seq_len].
    """
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    return mask


class GenerationMixin:
    """Mixin that adds text generation capabilities,
    including sampling with temperature, top-k, top-p,
    and min-probability filtering, for CausalLMs."""

    def generate(
        self,
        input_ids: jnp.ndarray,
        max_length: int = 20,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        do_sample: bool = True,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
        rng: jax.random.PRNGKey = None,
    ) -> jnp.ndarray:
        """Generate tokens autoregressively with various sampling methods.

        Args:
            input_ids: Initial token IDs of shape [batch_size, seq_len].
            max_length: Maximum length for generated sequences.
            temperature: Temperature for sampling.
            top_k: If specified, only sample from the top-k logits.
            top_p: If specified, only sample from the smallest set of logits
                  whose cumulative probability exceeds p.
            min_p: If specified, only consider logits with prob >= min_p * prob(max logit).
            do_sample: If True, use sampling; otherwise use greedy/beam search.
            pad_token_id: Token ID to use for padding.
            eos_token_id: Token ID that signals the end of generation.
            rng: Optional PRNG key for sampling.

        Returns:
            Generated token IDs of shape [batch_size, max_length].
        """

        if do_sample and rng is None:
            rng = jax.random.PRNGKey(0)
            print("Warning: No RNG key provided for sampling, using default key 0.")
        elif not do_sample and rng is None:
            rng = jax.random.PRNGKey(0)

        # Get initial sequence length and batch size
        batch_size, seq_len = input_ids.shape

        # If pad_token_id is not provided, try to get from config
        if pad_token_id is None:
            pad_token_id = self.config.get("pad_token_id", 0)

        # Initialize output with padding to max_length
        output_ids = jnp.full((batch_size, max_length), pad_token_id, dtype=input_ids.dtype)

        # Handle cases where input is already long enough
        if seq_len >= max_length:
            return input_ids[:, :max_length]

        # Copy input_ids to the beginning of output_ids
        output_ids = output_ids.at[:, :seq_len].set(input_ids)

        # Track whether each sequence is finished
        finished_sequences = jnp.zeros((batch_size,), dtype=jnp.bool_)

        if eos_token_id is not None:
            finished_sequences = input_ids[:, -1] == eos_token_id

        def scan_step(carry, _):
            """Performs one step of token generation."""
            # Unpack carry state
            current_output_ids = carry["output_ids"]
            current_length = carry["current_length"]
            step_rng = carry["rng"]
            current_finished = carry["finished"]

            # Split RNG key for this step if sampling
            next_rng = step_rng
            # Assuming do_sample is a static argument for JIT compilation
            if do_sample:
                step_rng, next_rng = jax.random.split(step_rng)

            # --- Prepare inputs for the model ---
            # Create a mask indicating valid positions up to current_length.
            position_indices = jnp.arange(max_length)
            # Mask is 1 for valid positions, 0 for padding/future
            attention_mask = (position_indices < current_length).astype(jnp.int32)

            logits = self(input_ids=current_output_ids, attention_mask=attention_mask)

            # Get logits for the next token prediction (at index current_length - 1)
            # Logits shape: [batch_size, max_length, vocab_size]
            next_token_logits = logits[:, current_length - 1, :]

            next_token = sample_logits(
                logits=next_token_logits,
                rng_key=step_rng,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                do_sample=do_sample,
            )
            next_token = next_token.astype(current_output_ids.dtype)  # Shape: (batch_size,)

            # Update finished state and mask token
            if eos_token_id is not None:
                newly_finished = (next_token == eos_token_id) & (~current_finished)
                next_finished = current_finished | newly_finished

                # If a sequence was already finished, keep padding it.
                # Otherwise, use the generated token (which might be EOS).
                output_token = jnp.where(current_finished, pad_token_id, next_token)
            else:
                # No EOS handling, always use the generated token
                next_finished = current_finished
                output_token = next_token

            # Write the potentially masked token at the current_length position
            updated_output_ids = current_output_ids.at[:, current_length].set(output_token)

            # Prepare carry for the next step
            next_carry = {
                "output_ids": updated_output_ids,
                "current_length": current_length + 1,
                "rng": next_rng,
                "finished": next_finished,
            }

            # Return the updated carry. The second element (per-step output) is None.
            return next_carry, None

        # Initialize carry state for scan
        initial_carry = {
            "output_ids": output_ids,
            "current_length": jnp.array(seq_len),
            "rng": rng,
            "finished": finished_sequences,
        }

        num_steps_to_generate = max_length - seq_len

        # Run the scan loop
        if num_steps_to_generate > 0:
            final_carry, _ = jax.lax.scan(
                scan_step, initial_carry, None, length=num_steps_to_generate
            )
            final_output_ids = final_carry["output_ids"]
        else:
            # If input length was already >= max_length, we returned earlier.
            # This case handles input_length == max_length exactly.
            final_output_ids = output_ids

        return final_output_ids
