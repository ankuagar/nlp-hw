import itertools
import os
import subprocess
import sys
import pandas as pd
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock

# Define features to use in generating power set
features = ["Length", "Frequency", "Category", "ContextualMatch", "PreviousGuess"]

# Global Lock for file write operations
file_lock = Lock()

# Ensure summary and logs directories exist only once
os.makedirs("summary", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Path for main log file
main_log_file = "logs/train_test_feature_set_log.txt"

# Generate power set of features
feature_subsets = list(itertools.chain.from_iterable(
    itertools.combinations(features, r) for r in range(len(features) + 1)))

def log_message(message, subset):
    """Thread-safe logging function."""
    with file_lock:
        with open(main_log_file, "a") as log_file:
            log_file.write(f"[Subset {subset}] {message}\n")

def generate_filename_stem(subset):
    if not subset:
        return "no_features"
    elif set(subset) == set(features):
        return "with_all_features"
    else:
        return "with_" + "_".join(subset).lower()

def run_commands(subset):
    filename_stem = generate_filename_stem(subset)
    training_limit = 50
    testing_limit = 25
    training_dataset = "../data/qanta.buzztrain.json.gz"
    test_dataset = "../data/qanta.buzzdev.json.gz"
    evaluation = "buzzer"

    # Unique JSON filename for each subset
    output_json = f"summary/eval_output_{filename_stem}.json"
    error_log = f"summary/error_log_{filename_stem}.txt"
    
    # Commands
    buzzer_command = [
        sys.executable, 'buzzer.py', '--guesser_type=Gpr', '--limit', str(training_limit),
        '--GprGuesser_filename=../models/buzztrain_gpr_cache', '--questions', training_dataset,
        '--buzzer_guessers', 'Gpr', '--LogisticBuzzer_filename=models/' + filename_stem
    ]
    if subset:
        buzzer_command.extend(['--features'] + list(subset))

    eval_command = [
        sys.executable, 'eval.py', '--guesser_type=Gpr', '--TfidfGuesser_filename=models/TfidfGuesser',
        '--limit', str(testing_limit), '--questions', test_dataset, '--buzzer_guessers', 'Gpr',
        '--GprGuesser_filename=../models/buzzdev_gpr_cache', '--LogisticBuzzer_filename=models/' + filename_stem,
        '--evaluate', evaluation, '--output_json', output_json
    ]
    if subset:
        eval_command.extend(['--features'] + list(subset))

    try:
        # Log start of commands
        log_message("Starting buzzer.py command.", subset)

        # Run the buzzer.py command
        subprocess.run(buzzer_command, check=True)

        # Run the eval.py command and log errors if any
        log_message("Starting eval.py command.", subset)
        with open(error_log, "w") as err_f:
            subprocess.run(eval_command, stderr=err_f, check=True)

        # Retry logic to wait for JSON output
        max_retries = 5
        retry_delay = 1  # seconds
        for attempt in range(max_retries):
            if os.path.getsize(output_json) > 0:
                break
            time.sleep(retry_delay)
        else:
            raise ValueError(f"File {output_json} is empty after retries.")

        # Validate JSON output
        with open(output_json, "r") as f:
            eval_results = json.load(f)
            if "error" in eval_results:
                raise ValueError(f"Error in eval.py output for subset {subset}: {eval_results['error']}")

        # Log successful processing
        log_message("Successfully processed subset.", subset)

        return {
            "Features": list(subset),
            "Filename Stem": filename_stem,
            "Training Limit": training_limit,
            "Testing Limit": testing_limit,
            "Training Dataset": training_dataset,
            "Test Dataset": test_dataset,
            "Evaluation": evaluation,
            **eval_results["outcome_percentages"],
            "Questions Right": eval_results["questions_right"],
            "Total": eval_results["total"],
            "Accuracy": eval_results["accuracy"],
            "Buzz Ratio": eval_results["buzz_ratio"],
            "Buzz Position": eval_results["buzz_position"]
        }
    except Exception as e:
        # Detailed error logging
        with open(error_log, "a") as err_file:
            err_file.write(f"Error for subset {subset}: {e}\n")
            err_file.write(f"Buzzer command: {' '.join(buzzer_command)}\n")
            err_file.write(f"Eval command: {' '.join(eval_command)}\n")
            if os.path.exists(output_json) and os.path.getsize(output_json) > 0:
                err_file.write("Output JSON file was partially written or corrupted.\n")
            else:
                err_file.write("Output JSON file was empty or not generated.\n")

        log_message(f"Subset {subset} generated an exception: {e}", subset)
        return None

# Multiprocessing with process isolation and file locks
if __name__ == "__main__":
    results = []
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
        future_to_subset = {executor.submit(run_commands, subset): subset for subset in feature_subsets}
        for future in future_to_subset:
            result = future.result()
            if result is not None:
                results.append(result)

    # Final DataFrame and CSV export
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Sort the DataFrame by descending order of Buzz Ratio
        results_df = results_df.sort_values(by="Buzz Ratio", ascending=False)
        
        # Export the initial CSV
        results_df.to_csv("summary/eval_summary.csv", index=False)

        # Data validation: Check for duplicate rows including and to the right of "waiting %"
        columns_to_check = results_df.columns[results_df.columns.get_loc("waiting %"):]
        duplicates = results_df.duplicated(subset=columns_to_check, keep=False)

        if duplicates.any():
            print("Warning: Duplicate rows found in the CSV output.")
            duplicate_rows = results_df[duplicates]
            duplicate_log_path = "summary/duplicate_rows_log.csv"
            duplicate_rows.to_csv(duplicate_log_path, index=False)
            print(f"Duplicate rows have been saved to {duplicate_log_path}")

        # Remove duplicate rows and save a new CSV without them
        results_df.drop_duplicates(subset=columns_to_check, keep='first', inplace=True)
        results_df.to_csv("summary/eval_summary_no_duplicates.csv", index=False)
    else:
        log_message("No results were generated, possibly due to errors in processing.", "Main")
        print("No results were generated, possibly due to errors in processing.")

