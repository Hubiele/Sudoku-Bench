# Neural Cellular Automata for Sudoku-Bench

This repository contains the code used for a master's thesis project on training a Neural Cellular Automaton (NCA) to solve Sudoku-like puzzles from Sudoku-Bench.

The project is research code. It is organized around this pipeline:

1. convert or generate Sudoku-style puzzle data,
2. build `.npy` stack datasets,
3. train an NCA model,
4. evaluate saved checkpoints,
5. record and analyze rollout traces.

The code is not packaged as a general Python library. Most scripts are meant to be run directly.

---

## Project structure

The repository is organized into several folders. The main source files are listed below. Generated data files, logs, plots, checkpoints, and JSON datasets are not listed in full.

```text
SudokuBench/
├── NCA/
│   ├── NCA_model.py
│   ├── TEST_NCA_model.py
│   └── filters.py
│
├── StackGenerator/
│   ├── check_stack_dataset_overlap.py
│   ├── puzzles_test.py
│   ├── puzzles_train.py
│   ├── puzzles_train_train_val_test.py
│   ├── remove_duplicate_from_train.py
│   ├── stack_gui_viewer.py
│   ├── dataset_overlap_check/
│   ├── stack_dataset/
│   └── stack_test/
│
├── Train/
│   ├── ParameterSearch/
│   │   ├── __init__.py
│   │   ├── optimizer_search.py
│   │   └── train_search.py
│   ├── checkpoints.py
│   ├── optimizer.py
│   ├── pool.py
│   ├── pool_low_ram.py
│   ├── train.py
│   ├── train_low_ram.py
│   ├── train_low_ram_variable_steps.py
│   └── train_step.py
│
├── Inference/
│   ├── analyze_checkpoint_traces.py
│   ├── checkpoint_trace_gui_viewer.py
│   ├── inference_analysis.py
│   ├── record_selected_puzzle_checkpoint_traces.py
│   ├── record_selected_puzzle_checkpoint_traces_combined_number_and_hidden_reset.py
│   ├── record_selected_puzzle_checkpoint_traces_hidden_reset.py
│   ├── record_selected_puzzle_checkpoint_traces_number_scaled.py
│   ├── testing_test.py
│   └── testing_test_with_csv.py
│
├── LearningRateSearch/
│   ├── LR_analysis.py
│   ├── 20260513_logs/
│   └── lr_search_analysis/
│
├── PuzzleGenerator/
│   └── grid_4x4/
│       ├── grid_generator/
│       │   ├── dataset_generator.py
│       │   ├── dataset_generator_parallel.py
│       │   └── test_new_rule_architecture.py
│       └── rule_logic/
│           ├── _new_arch_anti_knight_rule.py
│           ├── _new_arch_counting_circles_rule.py
│           ├── _new_arch_differences_count_lines_rule.py
│           ├── _new_arch_even_cells_rule.py
│           ├── _new_arch_killer_cages_rule.py
│           ├── _new_arch_little_killer_rule.py
│           ├── _new_arch_mean_baby_snake_rule.py
│           ├── _new_arch_odd_cells_rule.py
│           ├── _new_arch_partial_kropki_rule.py
│           ├── _new_arch_region_sum_lines_rule.py
│           ├── _new_arch_renban_lines_rule.py
│           ├── _new_arch_sunburn_cells_rule.py
│           ├── _new_arch_thermometers_rule.py
│           ├── _new_arch_zipper_lines_rule.py
│           └── common/
│               ├── common_base_4x4.py
│               ├── common_solver_4x4.py
│               ├── rule_api.py
│               └── rule_registry.py
│
└── Sudoku-Bench_json_conversion/
    ├── convert_challenges.py
    ├── challenge/
    └── converted/
```

### Main folders

- `NCA/` contains the NCA model and perception filters.
- `StackGenerator/` contains scripts for creating, checking, and viewing stack datasets.
- `Train/` contains the training loop, pool logic, optimizer, checkpoint tools, and parameter-search scripts.
- `Inference/` contains scripts for testing checkpoints, recording traces, analyzing traces, and inspecting traces with a GUI.
- `LearningRateSearch/` contains learning-rate search logs and analysis scripts.
- `PuzzleGenerator/` contains the 4x4 puzzle generator and rule logic.
- `Sudoku-Bench_json_conversion/` contains scripts and data folders for converting Sudoku-Bench challenge files into the internal JSON format.

### Script overview

#### `NCA/`

- `NCA_model.py`  
  Defines the Neural Cellular Automaton model, channel layout, embeddings, perception step, and update logic.

- `TEST_NCA_model.py`  
  Test or experimental version of the NCA model. This is mainly useful for checking model-related changes.

- `filters.py`  
  Defines perception filters used by the NCA, such as local convolution filters.

#### `StackGenerator/`

- `check_stack_dataset_overlap.py`  
  Checks whether training stacks overlap with test or Sudoku-Bench stacks. This is used to detect possible data leakage.

- `puzzles_train.py`  
  Builds the training and validation stack dataset from generated puzzle JSON files.

- `puzzles_train_train_val_test.py`  
  Alternative stack-generation script that can split generated puzzle data into train, validation, and test subsets.

- `puzzles_test.py`  
  Builds the Sudoku-Bench test stack dataset from converted challenge JSON files.

- `remove_duplicate_from_train.py`  
  Removes training samples that overlap with the Sudoku-Bench test set, based on the overlap report.

- `stack_gui_viewer.py`  
  Tkinter-based GUI for inspecting stack datasets, channels, masks, targets, and metadata.

#### `Train/`

- `checkpoints.py`  
  Saves and loads model checkpoints, optimizer state, pool state, random key, and metadata.

- `optimizer.py`  
  Defines the main optimizer and learning-rate schedule used for training.

- `pool.py`  
  Implements the normal in-memory pool used during pool-based NCA training.

- `pool_low_ram.py`  
  Implements a low-RAM pool loader that can read stack data from disk or memory maps.

- `train.py`  
  Main training script using the normal in-memory pool.

- `train_low_ram.py`  
  Main low-RAM training script. This is the main training entry point for larger stack datasets.

- `train_low_ram_variable_steps.py`  
  Variant of the low-RAM training script where the number of NCA rollout steps can vary during training.

- `train_step.py`  
  Defines the masked cross-entropy loss, rollout function, training step, and evaluation step.

#### `Train/ParameterSearch/`

- `__init__.py`  
  Marks the parameter-search folder as a Python package.

- `optimizer_search.py`  
  Defines optimizer settings that can be controlled through environment variables during parameter search.

- `train_search.py`  
  Training script used for parameter-search runs.

#### `Inference/`

- `testing_test.py`  
  Runs inference on the Sudoku-Bench test stack dataset and prints summary results to the terminal.

- `testing_test_with_csv.py`  
  Runs inference on the Sudoku-Bench test stack dataset and writes the results to a CSV file.

- `inference_analysis.py`  
  Reads inference CSV files and creates plots and summary tables.

- `record_selected_puzzle_checkpoint_traces.py`  
  Records rollout traces for selected puzzles and checkpoints.

- `record_selected_puzzle_checkpoint_traces_hidden_reset.py`  
  Trace-recording variant that resets hidden channels during the rollout analysis.

- `record_selected_puzzle_checkpoint_traces_number_scaled.py`  
  Trace-recording variant that scales number-channel values during the rollout analysis.

- `record_selected_puzzle_checkpoint_traces_combined_number_and_hidden_reset.py`  
  Trace-recording variant that resets both hidden channels and number channels.

- `analyze_checkpoint_traces.py`  
  Analyzes recorded traces and creates per-puzzle plots, combined plots, CSV summaries, and checkpoint/NCA-step summaries.

- `checkpoint_trace_gui_viewer.py`  
  Tkinter-based GUI for inspecting recorded checkpoint traces interactively.

#### `LearningRateSearch/`

- `LR_analysis.py`  
  Analyzes learning-rate search logs and creates comparison plots and ranking tables.

#### `PuzzleGenerator/grid_4x4/grid_generator/`

- `dataset_generator.py`  
  Generates 4x4 Sudoku-like puzzle JSON files with additional rule constraints.

- `dataset_generator_parallel.py`  
  Parallel version of the 4x4 dataset generator.

- `test_new_rule_architecture.py`  
  Tests whether the rule modules follow the expected rule architecture.

#### `PuzzleGenerator/grid_4x4/rule_logic/`

Each `_new_arch_*_rule.py` file implements one additional 4x4 Sudoku rule. These files define rule candidates, check whether a solution satisfies the rule, and provide descriptions used by the generator.

- `_new_arch_anti_knight_rule.py`  
  Implements the anti-knight rule.

- `_new_arch_counting_circles_rule.py`  
  Implements the counting-circles rule.

- `_new_arch_differences_count_lines_rule.py`  
  Implements the differences-count-lines rule.

- `_new_arch_even_cells_rule.py`  
  Implements the even-cells rule.

- `_new_arch_killer_cages_rule.py`  
  Implements the killer-cages rule.

- `_new_arch_little_killer_rule.py`  
  Implements the little-killer rule.

- `_new_arch_mean_baby_snake_rule.py`  
  Implements the mean-baby-snake rule.

- `_new_arch_odd_cells_rule.py`  
  Implements the odd-cells rule.

- `_new_arch_partial_kropki_rule.py`  
  Implements the partial-Kropki rule.

- `_new_arch_region_sum_lines_rule.py`  
  Implements the region-sum-lines rule.

- `_new_arch_renban_lines_rule.py`  
  Implements the Renban-lines rule.

- `_new_arch_sunburn_cells_rule.py`  
  Implements the sunburn-cells rule.

- `_new_arch_thermometers_rule.py`  
  Implements the thermometer rule.

- `_new_arch_zipper_lines_rule.py`  
  Implements the zipper-lines rule.

#### `PuzzleGenerator/grid_4x4/rule_logic/common/`

- `common_base_4x4.py`  
  Defines common 4x4 Sudoku constants, grid utilities, units, peers, and neighbor functions.

- `common_solver_4x4.py`  
  Contains the 4x4 solver and helper functions for checking uniqueness and finding minimal givens.

- `rule_api.py`  
  Defines the common rule interface used by the rule modules.

- `rule_registry.py`  
  Loads and registers available rule modules.

#### `Sudoku-Bench_json_conversion/`

- `convert_challenges.py`  
  Converts Sudoku-Bench challenge JSON files into the internal JSON format used by the stack-generation scripts.



## Environment

The project was developed with Python 3.12.

The main Python dependencies are:

```text
jax
jaxlib
equinox
optax
numpy
pandas
matplotlib
```

Some helper scripts also use:

```text
tkinter
tarfile
multiprocessing
```

A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install jax jaxlib equinox optax numpy pandas matplotlib
```

For GPU use, install the JAX version that matches the target system. The correct JAX installation command depends on whether the system uses CPU, NVIDIA CUDA, or AMD ROCm.

---

## Important note about hardcoded paths

Several scripts contain hardcoded absolute paths from the original development machine, for example:

```text
/home/daniel/Documents/Skole/Masteroppgave/Kode/new_new_SudokuBench
```

If the repository is placed somewhere else, update the path variables near the top of the relevant scripts.

Common variables to check are:

```python
PROJECT_ROOT
DATASET_INPUT_DIR
INPUT_JSON_DIR
TEST_STACK_DIR
CHECKPOINT_PATH
TRACE_ROOT_DIR
```

A useful way to find hardcoded paths is:

```bash
grep -R "/home/daniel" -n .
```

Then update each path to match the new location.

---

## Dataset files

Large generated `.npy` files may not be included directly in the repository because of their size. However, `stack_dataset.tar.gz` may be included and can be unpacked to restore the generated training and validation dataset.

The training pipeline expects files such as:

```text
StackGenerator/stack_dataset/train_stacks.npy
StackGenerator/stack_dataset/validation_stacks.npy
StackGenerator/stack_test/test_stacks.npy
```

If these files are missing, they must be generated before training or inference.

The training script `Train/train_low_ram.py` expects the training dataset in:

```text
StackGenerator/stack_dataset/
```

or as an archive:

```text
StackGenerator/stack_dataset.tar.gz
```

The low-RAM pool loader can also use the environment variable:

```bash
export STACK_DATASET_DIR=/path/to/stack_dataset
```

This is useful if the dataset is stored outside the repository.

---

## Typical workflow

The full workflow depends on what should be reproduced, but the intended order is approximately:

### 1. Convert Sudoku-Bench challenge files

Use the conversion scripts in:

```text
Sudoku-Bench_json_conversion/
```

The converted files are expected under:

```text
Sudoku-Bench_json_conversion/converted/
```

### 2. Generate training and validation stacks

Use:

```bash
python StackGenerator/puzzles_train.py
```

This creates stack arrays for training and validation.

Before running, check the path variables inside the script. In particular, check:

```python
DATASET_INPUT_DIR
```

The expected output folder is usually:

```text
StackGenerator/stack_dataset/
```

### 3. Generate Sudoku-Bench test stacks

Use:

```bash
python StackGenerator/puzzles_test.py
```

This creates the test stack array used for Sudoku-Bench evaluation.

The expected output folder is usually:

```text
StackGenerator/stack_test/
```

### 4. Check for overlap between training and test data

To check whether any Sudoku-Bench test inputs also occur in the training data, run:

```bash
python StackGenerator/check_stack_dataset_overlap.py
```

This produces overlap reports, including:

```text
overlap_input.csv
overlap_full.csv
overlap_summary.json
```

The most important file for data leakage is usually:

```text
overlap_input.csv
```

This checks overlap in the puzzle input representation.

### 5. Remove overlapping training samples

If overlap is found, run:

```bash
python StackGenerator/remove_duplicate_from_train.py
```

This script creates a new deduplicated training file instead of modifying the original file directly.

After this, run the overlap check again to confirm that no identical benchmark inputs remain in the training data.

### 6. Train the model

The main low-RAM training script is:

```bash
python Train/train_low_ram.py
```

There is also a variable-step version:

```bash
python Train/train_low_ram_variable_steps.py
```

Before running, check the training configuration near the top of the script.

Important values include:

```python
MAX_NCA_STEPS
VALIDATION_NCA_STEPS
SAVE_EVERY
CKPT_DIR
```

Checkpoints are usually written to:

```text
Train/checkpoints/
```

or to a relative `checkpoints/` folder depending on the working directory.

---

## Checkpoints

Checkpoints are saved as folders such as:

```text
step_001000/
step_002000/
...
step_100000/
```

Each checkpoint folder normally contains:

```text
model.eqx
opt_state.eqx
pool_state.eqx
key.eqx
meta.json
```

The training iteration is stored in `meta.json` under:

```json
{
  "step": 100000
}
```

This means the iteration can be read from the checkpoint data itself, not only from the folder name.

### Included checkpoint

The repository may include the checkpoint:

```text
step_097000/
```

This checkpoint is included so that inference can be run without retraining the model from scratch.

In the thesis evaluation, the best single checkpoint and NCA-step combination was checkpoint `097000` with NCA step `30`. To use this checkpoint for inference, set the checkpoint path in the inference script to the `step_097000/` folder and set:

```python
NCA_STEPS = 30
```

For example, check these variables in `Inference/testing_test_with_csv.py`:

```python
CHECKPOINT_PATH
NCA_STEPS
```

The checkpoint is mainly meant for inference and result reproduction. To reproduce the full training process, run the training pipeline instead.

---

## Inference

To evaluate a checkpoint on the Sudoku-Bench test stacks, use:

```bash
python Inference/testing_test_with_csv.py
```

Before running, check these variables in the script:

```python
PROJECT_ROOT
TEST_STACK_DIR
CHECKPOINT_PATH
NCA_STEPS
```

The script writes a CSV summary that can later be plotted or analyzed.

---

## Inference result analysis

To create plots from inference CSV files, use:

```bash
python Inference/inference_analysis.py --root Inference
```

The script searches recursively for result CSV files. It expects CSV files with columns such as:

```text
side
puzzle_id
loss_min
board_wrong_best
predict_wrong_best
puzzle_solved_any_trial
```

It creates plots beside each CSV file and a combined summary folder:

```text
_result_plot_summary/
```

---

## Trace recording and analysis

Trace recording stores the model state over NCA rollout steps and checkpoints.

The main recording script is:

```bash
python Inference/record_selected_puzzle_checkpoint_traces.py
```

Before running, check the path variables near the top of the script, especially:

```python
CHECKPOINT_ROOT
TEST_STACK_DIR
OUT_DIR
SELECTED_PUZZLE_INDICES
NCA_STEPS
```

`SELECTED_PUZZLE_INDICES` uses 1-based indexing. For example:

```python
SELECTED_PUZZLE_INDICES = [1]
```

selects the first puzzle in the test stack.

To analyze saved traces, use:

```bash
python Inference/analyze_checkpoint_traces.py
```

Check:

```python
TRACE_ROOT_DIR
OUT_DIR_NAME
```

The analysis script creates combined plots, per-puzzle plots, and summary CSV files.

---

## GUI tools

Some helper scripts provide simple graphical user interfaces for inspecting generated data and model traces. These tools are optional. They are not needed for training or batch evaluation, but they can be useful for manual inspection and debugging.

### Checkpoint trace viewer

The main GUI tool is:

```bash
python Inference/checkpoint_trace_gui_viewer.py
```

This opens a Tkinter-based viewer for trace folders created by:

```bash
python Inference/record_selected_puzzle_checkpoint_traces.py
```

The viewer can be used to inspect how the model state changes across:

```text
puzzles
checkpoints
NCA steps
channels
derived views
```

The GUI can show raw channel values, predicted digits, prediction confidence, digit entropy, givens, target digits, board masks, and prediction masks.

Before running the GUI, make sure that the trace folder exists and contains:

```text
trace_manifest.json
puzzle_*/
    puzzle_metadata.json
    step_*/
        trace.npy
        metrics.json
```

By default, the viewer looks for:

```text
selected_puzzle_checkpoint_traces/
```

relative to the script location or the current working directory.

If the trace folder is somewhere else, update this variable near the top of the script:

```python
TRACE_ROOT_DIR = Path("/path/to/selected_puzzle_checkpoint_traces")
```

The GUI may also have an "Open trace root" button that can be used to select the trace folder manually.

### GUI dependencies

The GUI uses `tkinter` and Matplotlib's Tk backend. On some Linux systems, `tkinter` may need to be installed separately, for example:

```bash
sudo apt install python3-tk
```

If the GUI does not start, first check that `tkinter` is available:

```bash
python -c "import tkinter; print('tkinter works')"
```

The GUI tools are intended for interactive inspection only. They are not used by the training pipeline itself.

---

## Notes on indices

Some scripts use 1-based puzzle indices for readability, especially for selected Sudoku-Bench puzzles.

Internally, NumPy arrays use 0-based indexing.

For example:

```python
puzzle_idx = 51
arr_idx = puzzle_idx - 1
```

This means `puzzle_0051` corresponds to `stacks[50]`.

---

## Reproducibility notes

To reproduce the final evaluation, the important steps are:

1. generate or restore the training and validation stack dataset,
2. generate the Sudoku-Bench test stack dataset,
3. check and remove direct input overlap between training data and Sudoku-Bench,
4. either train the final model or use the included `step_097000` checkpoint,
5. evaluate checkpoint `097000` with NCA step `30`,
6. record and analyze traces if needed.

Because several scripts contain local absolute paths, reproducibility requires checking and updating path variables before running the scripts.

---

## Acknowledgements

This project is inspired by work on Neural Cellular Automata, especially the idea of learned local update rules in systems such as Growing Neural Cellular Automata. The Sudoku-Bench dataset and challenge format are used as the target benchmark for evaluating Sudoku-like reasoning under tightly coupled constraints.

Some analysis and augmentation ideas in the wider project were also influenced by related work on recursive models and grid-based reasoning. The code in this repository should be treated as project-specific research code.
