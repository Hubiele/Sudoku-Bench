from __future__ import annotations
import json
from pathlib import Path


def string_to_2d_matrix(board_str: str, side: int = 9) -> list[list[int]]:
    """Convert a flat board string, using "." for empty cells, into a 2D matrix."""
    matrix = []
    for i in range(side):
        row = []
        for j in range(side):
            char = board_str[i * side + j]
            if char == '.':
                row.append(0)
            else:
                row.append(int(char))
        matrix.append(row)
    return matrix


def convert_challenge_files():
    # Define the input and output paths.
    input_dir = Path("/home/daniel/Documents/Skole/Masteroppgave/Kode/new_new_SudokuBench/Sudoku-Bench_json_conversion/challenge")

    # The output folder is named "converted".
    output_dir = Path("/home/daniel/Documents/Skole/Masteroppgave/Kode/new_new_SudokuBench/Sudoku-Bench_json_conversion/converted")

    # If needed, update the output path to match the local project structure.

    if not input_dir.exists():
        print(f"[-] Could not find the input folder 'challenge' here: {input_dir}")
        print("Check that the folder structure matches the script configuration.")
        return

    # Create the output folder if it does not already exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files in the challenge folder.
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"[-] Could not find any .json files in {input_dir}")
        return

    print(f"[+] Found {len(json_files)} files to convert...")

    success_count = 0
    for file_path in json_files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                source = json.load(f)

            # Read the grid size from the source file. Default is 9x9.
            side = source.get("rows", 9)

            # Convert the input board and solution to 2D lists.
            givens_2d = string_to_2d_matrix(source["initial_board"], side)
            solution_2d = string_to_2d_matrix(source["solution"], side)

            # Build the internal format used by the stack-generation scripts.
            target_format = {
                "variant_id": source.get("puzzle_id", file_path.stem),
                "active_rule_names": [f"standard_sudoku_{side}x{side}"],
                "side": side,
                "digits": list(range(1, side + 1)),
                "board_offset": [0, 0],
                "solution": solution_2d,
                "givens": givens_2d,
                "metadata": {
                    "variant_mode": "regular",
                    "title": source.get("title", ""),
                    "author": source.get("author", ""),
                    "original_rules": source.get("rules", "")
                }
            }

            # Save the converted file in the output folder.
            output_file_path = output_dir / file_path.name
            with output_file_path.open("w", encoding="utf-8") as f:
                json.dump(target_format, f, indent=2, ensure_ascii=False)

            success_count += 1

        except Exception as e:
            print(f"[-] Could not convert {file_path.name}. Reason: {e}")

    print(f"\n[+] Done. Converted {success_count}/{len(json_files)} files.")
    print(f"[+] The converted files are available in: {output_dir.resolve()}")


if __name__ == "__main__":
    convert_challenge_files()