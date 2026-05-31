# Script for testing the NCA model

import jax
import jax.numpy as jnp
from PuzzleTemplate.puzzles_old import CreateRandomPuzzle
from NCA_model import NCA
from Loss.loss_fn import Loss

print(jax.devices())

key = jax.random.PRNGKey(100)
key, subkey = jax.random.split(key)

puzzle = CreateRandomPuzzle(subkey)
loss_function = Loss

grid = puzzle.get_channel_stack()

print(puzzle.get_grid_size())

print("Print starting stack:")
print(grid[:, :, 0:9])
print()

key, subkey = jax.random.split(key)
model = NCA(subkey, grid)

output = grid
for i in range(100):
    key, step_key = jax.random.split(key)

    new_output = model(step_key, output)
    output = new_output

loss_object = loss_function(output)
loss_value = loss_object(output)

print("Print output stack:")
print(output[:, :, 0:9])
print("Loss:")
print(float(loss_value))
