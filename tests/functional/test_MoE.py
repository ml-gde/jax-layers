import jax
import jax.numpy as jnp
import pytest

from jaxgarden.functional.mixture_of_experts import Expert, GatingNetwork, MixtureOfExperts


@pytest.fixture
def key():
    """Provides a JAX PRNG key for tests."""
    return jax.random.PRNGKey(0)

@pytest.fixture
def dummy_input_batch():
    """Provides a dummy input batch."""
    # (batch_size, input_dim)
    return jnp.ones((4, 16)) # Batch of 4, input dimension 16

def test_expert_initialization_and_call(key, dummy_input_batch):
    """Tests the Expert module initialization and forward pass."""
    expert_output_dim = 8
    expert_model = Expert(num_outputs=expert_output_dim)

    params = expert_model.init(key, dummy_input_batch)['params']
    output = expert_model.apply({'params': params}, dummy_input_batch)

    assert output.shape == (dummy_input_batch.shape[0], expert_output_dim)
    assert output.dtype == jnp.float32 # Default dtype for Dense

def test_gating_network_initialization_and_call(key, dummy_input_batch):
    """Tests the GatingNetwork module initialization and forward pass."""
    num_experts = 3
    gating_model = GatingNetwork(num_experts=num_experts)

    params = gating_model.init(key, dummy_input_batch)['params']
    gate_weights = gating_model.apply({'params': params}, dummy_input_batch)

    assert gate_weights.shape == (dummy_input_batch.shape[0], num_experts)
    # Check if softmax output sums to 1 for each item in the batch
    assert jnp.allclose(jnp.sum(gate_weights, axis=-1),
                        jnp.ones(dummy_input_batch.shape[0]), atol=1e-6)
    assert jnp.all(gate_weights >= 0) # Probabilities should be non-negative

def test_mixture_of_experts_initialization_and_call(key, dummy_input_batch):
    """Tests the MixtureOfExperts module initialization and forward pass."""
    num_experts = 5
    expert_output_dim = 10
    moe_model = MixtureOfExperts(num_experts=num_experts, expert_output_dim=expert_output_dim)

    params = moe_model.init(key, dummy_input_batch)['params']
    final_output = moe_model.apply({'params': params}, dummy_input_batch)

    assert final_output.shape == (dummy_input_batch.shape[0], expert_output_dim)
    assert final_output.dtype == jnp.float32

def test_mixture_of_experts_output_logic(key, dummy_input_batch):
    """
    Tests the output logic of MoE by checking if expert outputs are combined
    as expected based on gate weights.
    """
    num_experts = 2
    expert_output_dim = 3
    # input_dim = dummy_input_batch.shape[1] # Not strictly needed for this test logic after init

    # Create an MoE model
    moe_model = MixtureOfExperts(num_experts=num_experts, expert_output_dim=expert_output_dim)
    variables = moe_model.init(key, dummy_input_batch)
    params = variables['params']

    # --- Manually compute expected output for a specific case ---

    # Get gate weights by applying a GatingNetwork instance with its specific parameters
    gating_sub_model = GatingNetwork(num_experts=num_experts)
    gate_weights = gating_sub_model.apply({'params': params['gating_network']}, dummy_input_batch)

    # Get individual expert outputs
    expert_sub_model_template = Expert(num_outputs=expert_output_dim)

    expert0_output = expert_sub_model_template.apply({'params':
                                                      params['expert_0']}, dummy_input_batch)
    expert1_output = expert_sub_model_template.apply({'params':
                                                      params['expert_1']}, dummy_input_batch)

    # Expected combination
    # gate_weights is (batch_size, num_experts)
    # expertN_output is (batch_size, expert_output_dim)
    # We use slicing and broadcasting for the weighted sum.
    expected_output_manual = \
        gate_weights[:, 0:1] * expert0_output + \
        gate_weights[:, 1:2] * expert1_output

    moe_output = moe_model.apply({'params': params}, dummy_input_batch)

    assert moe_output.shape == (dummy_input_batch.shape[0], expert_output_dim)
    assert jnp.allclose(moe_output, expected_output_manual, atol=1e-6)

def test_mixture_of_experts_single_expert_case(key, dummy_input_batch):
    """Tests MoE with only one expert."""
    num_experts = 1
    expert_output_dim = 7
    moe_model = MixtureOfExperts(num_experts=num_experts, expert_output_dim=expert_output_dim)

    params = moe_model.init(key, dummy_input_batch)['params']
    final_output = moe_model.apply({'params': params}, dummy_input_batch)

    # The output should be identical to the output of the single expert
    # as gate_weights will be [[1.], [1.], ...]

    # Instantiate an Expert model to apply its specific parameters
    expert_sub_model = Expert(num_outputs=expert_output_dim)
    expert_output = expert_sub_model.apply({'params': params['expert_0']}, dummy_input_batch)

    assert final_output.shape == (dummy_input_batch.shape[0], expert_output_dim)
    assert jnp.allclose(final_output, expert_output, atol=1e-6)

    # Check gate weights for the single expert case
    # Instantiate a GatingNetwork model to apply its specific parameters
    gating_sub_model = GatingNetwork(num_experts=num_experts)
    gate_weights = gating_sub_model.apply({'params': params['gating_network']}, dummy_input_batch)

    assert gate_weights.shape == (dummy_input_batch.shape[0], 1)
    assert jnp.allclose(gate_weights, jnp.ones_like(gate_weights), atol=1e-6)


def test_expert_different_output_dims(key):
    """Tests Expert with varying output dimensions."""
    input_data = jnp.ones((2, 5)) # batch=2, features=5
    for out_dim in [1, 5, 20]:
        expert_model = Expert(num_outputs=out_dim)
        params = expert_model.init(key, input_data)['params']
        output = expert_model.apply({'params': params}, input_data)
        assert output.shape == (input_data.shape[0], out_dim)

def test_gating_network_different_num_experts(key):
    """Tests GatingNetwork with varying number of experts."""
    input_data = jnp.ones((3, 8)) # batch=3, features=8
    for num_exp in [1, 4, 10]:
        gating_model = GatingNetwork(num_experts=num_exp)
        params = gating_model.init(key, input_data)['params']
        gate_weights = gating_model.apply({'params': params}, input_data)
        assert gate_weights.shape == (input_data.shape[0], num_exp)
        assert jnp.allclose(jnp.sum(gate_weights, axis=-1),
                            jnp.ones(input_data.shape[0]), atol=1e-6)

def test_mixture_of_experts_different_params(key):
    """Tests MixtureOfExperts with varying numbers of experts and output dimensions."""
    input_data = jnp.ones((2, 12)) # batch=2, features=12
    configurations = [
        (2, 4),  # num_experts, expert_output_dim
        (4, 8),
        (1, 6),
        (3, 3)
    ]
    for num_exp, exp_out_dim in configurations:
        moe_model = MixtureOfExperts(num_experts=num_exp, expert_output_dim=exp_out_dim)
        params = moe_model.init(key, input_data)['params']
        final_output = moe_model.apply({'params': params}, input_data)
        assert final_output.shape == (input_data.shape[0], exp_out_dim)
