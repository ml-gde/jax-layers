"""
JAX/Flax implementation of a Mixture of Experts (MoE) layer.

This code provides a conceptual implementation of a Mixture of Experts layer,
a neural network architecture where multiple specialized "expert" sub-networks
are combined. A gating network determines which expert (or combination of
experts) processes a given input, allowing the model to learn to route
different parts of the input space to specialized modules. This can lead to
models with higher capacity and better efficiency, especially in sparse
formulations where only a subset of experts are activated per input.

The core concept of Mixture of Experts was introduced in the paper:
"Adaptive Mixtures of Local Experts"
by Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton.
Published in Neural Computation, Volume 3, Issue 1, Pages 79-87, 1991.
Available at: https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf
"""

import flax.linen as nn
import jax.numpy as jnp


# Expert Network
class Expert(nn.Module):
    num_outputs: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        A simple feed-forward expert network.
        Args:
            x: Input tensor.
        Returns:
            Output tensor from the expert.
        """
        x = nn.Dense(features=self.num_outputs * 2, name="expert_dense_1")(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_outputs, name="expert_dense_2")(x)
        return x

# Gating Network
class GatingNetwork(nn.Module):
    num_experts: int # The number of experts to choose from

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        A simple gating network that outputs weights for each expert.
        Args:
            x: Input tensor.
        Returns:
            A tensor of weights for each expert (after softmax).
        """
        # The gating network is often a linear layer followed by a softmax
        gate_logits = nn.Dense(features=self.num_experts, name="gating_dense")(x)
        gate_weights = nn.softmax(gate_logits, axis=-1)
        return gate_weights

# Mixture of Experts Layer
class MixtureOfExperts(nn.Module):
    num_experts: int
    expert_output_dim: int

    def setup(self) -> None:
        """
        Initialize the experts and the gating network.
        This method is called by Flax automatically.
        """
        # List of Expert modules
        self.experts = [Expert(num_outputs=self.expert_output_dim,
                               name=f"expert_{i}") for i in range(self.num_experts)]
        # Create the GatingNetwork module
        self.gating_network = GatingNetwork(num_experts=self.num_experts, name="gating_network")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass for the Mixture of Experts layer.
        Args:
            x: Input tensor.
        Returns:
            The combined output from the experts.
        """
        # 1. Get the gating weights
        # Input shape: (batch_size, input_dim)
        # Gate weights shape: (batch_size, num_experts)
        gate_weights = self.gating_network(x)

        # 2. Get the outputs from all experts
        # We'll store expert outputs in a list
        expert_outputs = []
        for i in range(self.num_experts):
            # Expert output shape: (batch_size, expert_output_dim)
            expert_out = self.experts[i](x)
            expert_outputs.append(expert_out)

        # Stack expert outputs along a new dimension to facilitate weighted sum
        # Stacked expert_outputs shape: (batch_size, num_experts, expert_output_dim)
        stacked_expert_outputs = jnp.stack(expert_outputs, axis=1)

        # We want to weight each expert's output for each item in the batch.
        # Gate weights shape:      (batch_size, num_experts)
        # Needs to be broadcast to: (batch_size, num_experts, expert_output_dim)
        # to multiply with stacked_expert_outputs.
        expanded_gate_weights = jnp.expand_dims(gate_weights, axis=-1)

        # Weighted outputs shape: (batch_size, num_experts, expert_output_dim)
        weighted_expert_outputs = stacked_expert_outputs * expanded_gate_weights

        # Sum the weighted outputs along the num_experts dimension
        # Final output shape: (batch_size, expert_output_dim)
        final_output = jnp.sum(weighted_expert_outputs, axis=1)

        return final_output
