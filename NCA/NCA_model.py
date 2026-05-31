import jax
import jax.numpy as jnp
import equinox as eqx
from NCA.filters import create_filter_bank

# Channel constants are loaded from the current puzzle generator.
# First, try the project structure used by the current codebase.
try:
    from StackGenerator.puzzles_train import (
        NUMBER_CHANNELS,
        GIVEN_MASK_CHANNELS,
        BOARD_MASK_CHANNELS,
        RULE_CHANNELS,
        HIDDEN_CHANNELS,
        TARGET_CHANNELS,
    )
except ImportError:
    # Fallbacks for simpler testing or alternative project layouts.
    try:
        from puzzles import (
            NUMBER_CHANNELS,
            GIVEN_MASK_CHANNELS,
            BOARD_MASK_CHANNELS,
            RULE_CHANNELS,
            HIDDEN_CHANNELS,
            TARGET_CHANNELS,
        )
    except ImportError:
        from PuzzleTemplate.puzzles_old import NUMBER_CHANNELS, HIDDEN_CHANNELS  # type: ignore
        GIVEN_MASK_CHANNELS = 1
        BOARD_MASK_CHANNELS = 1
        RULE_CHANNELS = 60
        TARGET_CHANNELS = 9


# Stack layout
GIVEN_MASK_START = NUMBER_CHANNELS
GIVEN_MASK_END = GIVEN_MASK_START + GIVEN_MASK_CHANNELS
BOARD_MASK_START = GIVEN_MASK_END
BOARD_MASK_END = BOARD_MASK_START + BOARD_MASK_CHANNELS
RULE_CHANNELS_START = BOARD_MASK_END
RULE_CHANNELS_END = RULE_CHANNELS_START + RULE_CHANNELS
HIDDEN_CHANNELS_START = RULE_CHANNELS_END
HIDDEN_CHANNELS_END = HIDDEN_CHANNELS_START + HIDDEN_CHANNELS
INPUT_CHANNELS = HIDDEN_CHANNELS_END
TARGET_CHANNELS_START = INPUT_CHANNELS
TOTAL_STACK_CHANNELS = INPUT_CHANNELS + TARGET_CHANNELS

# Metadata embeddings
RULE_EMBEDDINGS = RULE_CHANNELS
GRID_SIZE_EMBEDDINGS = 3
LEN_RULE_EMBEDDINGS = 64
LEN_GRID_SIZE_EMBEDDINGS = 16
MAX_ACTIVE_RULE_EMBEDDINGS = 1
MAX_ACTIVE_GRID_SIZE_EMBEDDINGS = 1

# MLP input size: perceived channels plus rule and grid-size embeddings.
in_dim = (
    (INPUT_CHANNELS * 4)
    + (MAX_ACTIVE_RULE_EMBEDDINGS * LEN_RULE_EMBEDDINGS)
    + (MAX_ACTIVE_GRID_SIZE_EMBEDDINGS * LEN_GRID_SIZE_EMBEDDINGS)
)
out_dim = NUMBER_CHANNELS + HIDDEN_CHANNELS


class NCA(eqx.Module):
    network: eqx.nn.Sequential
    rule_embeddings: eqx.nn.Embedding
    grid_size_embeddings: eqx.nn.Embedding
    filter_bank: jnp.ndarray
    num_filters: int
    update_channels: jnp.ndarray
    perceive: jnp.ndarray

    def __init__(self, key, grid_hwc):
        filter_names = ["identity", "laplace4", "laplace8", "unibox"]
        self.filter_bank = create_filter_bank(filter_names)
        self.num_filters = self.filter_bank.shape[0]

        # Perception should only use the input part of the stack, not the target channels.
        kernels = jnp.transpose(self.filter_bank, (1, 2, 0))
        self.perceive = jnp.tile(kernels[:, :, None, :], (1, 1, 1, INPUT_CHANNELS))

        key, rule_emb_key, size_emb_key, mlp_key1, mlp_key2, mlp_key3, mlp_key4 = jax.random.split(key, 7)

        self.rule_embeddings = eqx.nn.Embedding(RULE_EMBEDDINGS, LEN_RULE_EMBEDDINGS, key=rule_emb_key)
        self.grid_size_embeddings = eqx.nn.Embedding(GRID_SIZE_EMBEDDINGS, LEN_GRID_SIZE_EMBEDDINGS, key=size_emb_key)

        layer_width = in_dim
        self.network = eqx.nn.Sequential([
            eqx.nn.Linear(in_dim, layer_width, key=mlp_key1),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(layer_width, layer_width, key=mlp_key2),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(layer_width, out_dim, key=mlp_key4),
        ])

        # Only number channels and hidden channels are allowed to change.
        update_channels = jnp.zeros(INPUT_CHANNELS)
        update_channels = update_channels.at[0:NUMBER_CHANNELS].set(1.0)
        update_channels = update_channels.at[HIDDEN_CHANNELS_START:HIDDEN_CHANNELS_END].set(1.0)
        self.update_channels = update_channels[None, None, :]

    def _split_stack(self, grid_hwc: jnp.ndarray):
        """
        Support both a full stack and an input-only stack.

        If a full stack is passed in, the target channels are kept frozen.
        """
        channels = grid_hwc.shape[-1]
        if channels == TOTAL_STACK_CHANNELS:
            return grid_hwc[:, :, :INPUT_CHANNELS], grid_hwc[:, :, INPUT_CHANNELS:]
        if channels == INPUT_CHANNELS:
            return grid_hwc, None
        raise ValueError(
            f"Unexpected number of channels: {channels}. "
            f"Expected {INPUT_CHANNELS} (input only) or {TOTAL_STACK_CHANNELS} (full stack)."
        )

    def perceive_local(self, input_grid_hwc: jnp.ndarray) -> jnp.ndarray:
        batched_grid = input_grid_hwc[None, :, :, :]
        perceptions = jax.lax.conv_general_dilated(
            batched_grid,
            self.perceive,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=INPUT_CHANNELS,
        )
        return perceptions[0]

    def _grid_size_embedding(self, board_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Read the grid size from the board-mask channel.

        4x4 has 16 active cells, 6x6 has 36, and 9x9 has 81.
        """
        active_cells = jnp.sum(board_mask)
        grid_size_embedding_index = jnp.where(
            active_cells > 60, 2,
            jnp.where(active_cells > 20, 1, 0)
        ).astype(jnp.int32)
        return self.grid_size_embeddings(grid_size_embedding_index)

    def __call__(self, key, grid_hwc):
        input_grid_hwc, frozen_target_hwc = self._split_stack(grid_hwc)
        H, W, _ = input_grid_hwc.shape

        base_mask = self.update_channels
        given_mask = input_grid_hwc[:, :, GIVEN_MASK_START:GIVEN_MASK_END]
        board_mask = input_grid_hwc[:, :, BOARD_MASK_START:BOARD_MASK_END]

        fire_rate_mask = jax.random.bernoulli(key, p=1.0, shape=(H, W, 1))

        # Number channels are updated only in active board cells that are not givens.
        logits_mask = (
            base_mask[:, :, :NUMBER_CHANNELS]
            * board_mask
            * (1.0 - given_mask)
            * fire_rate_mask
        )

        # Hidden channels are updated only inside the active board.
        rest_mask = jnp.broadcast_to(
            base_mask[:, :, NUMBER_CHANNELS:],
            (H, W, base_mask.shape[-1] - NUMBER_CHANNELS),
        ) * board_mask

        combined_mask = jnp.concatenate([logits_mask, rest_mask], axis=-1)

        perceptions_hwc = self.perceive_local(input_grid_hwc)

        # Detect which rule channels are active anywhere on the board.
        rule_channels = input_grid_hwc[:, :, RULE_CHANNELS_START:RULE_CHANNELS_END]
        rule_presence_list = jnp.max(rule_channels, axis=(0, 1))
        # Combine the embeddings for the active rules into one rule vector.
        all_rule_embeddings = self.rule_embeddings.weight
        masked_embeddings = all_rule_embeddings * rule_presence_list[:, None]
        combined_rule_embeddings = jnp.sum(masked_embeddings, axis=0)

        grid_size_embedding = self._grid_size_embedding(board_mask)

        # Give every cell access to the same rule and grid-size information.
        combined_embeddings = jnp.concatenate([combined_rule_embeddings, grid_size_embedding])
        broadcasted_embeddings = jnp.broadcast_to(
            combined_embeddings,
            (H, W, LEN_RULE_EMBEDDINGS + LEN_GRID_SIZE_EMBEDDINGS),
        )
        total_input = jnp.concatenate([perceptions_hwc, broadcasted_embeddings], axis=-1)

        # Apply the same MLP independently to every cell.
        update_values = jax.vmap(jax.vmap(self.network))(total_input)

        update_grid = jnp.zeros((H, W, INPUT_CHANNELS))
        update_grid = update_grid.at[:, :, :NUMBER_CHANNELS].set(update_values[:, :, :NUMBER_CHANNELS])
        update_grid = update_grid.at[:, :, HIDDEN_CHANNELS_START:HIDDEN_CHANNELS_END].set(
            update_values[:, :, NUMBER_CHANNELS:]
        )

        # Apply the update mask and limit the update size for stability.
        final_update_grid = jnp.clip(update_grid * combined_mask, min=-2, max=2)
        new_input_grid = input_grid_hwc + final_update_grid

        if frozen_target_hwc is None:
            return new_input_grid
        return jnp.concatenate([new_input_grid, frozen_target_hwc], axis=-1)