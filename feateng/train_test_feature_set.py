import itertools
import os
import subprocess
import sys
import pandas as pd
import json
import time
from datetime import datetime

LOSS_FUNCTIONS = {
    "MLP": "BuzzLoss",
    "LogisticBuzzer": "Logistic Loss",
    "RNNBuzzer": "CrossEntropyLoss"
}

# Define the features to use in generating the power set
features = ["Length", "Frequency", "Category", "ContextualMatch"]
# features = []

# DataFrame to store results
results_df = pd.DataFrame(columns=[
    "Features", "Buzzer Type", "Filename Stem", "Loss Function", "Training Limit", "Testing Limit",
    "Training Dataset", "Test Dataset", "Evaluation",
    "best %", "timid %", "hit %", "close %", "miss %", "aggressive %", "waiting %",
    "Questions Right", "Total", "Accuracy", "Buzz Ratio", "Buzz Position"
])

# Function to generate the filename stem based on the subset of features
def generate_filename_stem(subset, buzzer_type="LogisticBuzzer"):
    if buzzer_type == "LogisticBuzzer":
        buzzer_str = "logit"
    else:
        buzzer_str = buzzer_type.lower()
    if not subset:
        return buzzer_str + "_no_features"
    elif set(subset) == set(features):
        return buzzer_str + "_with_all_features"
    else:
        return buzzer_str + "_with_" + "_".join(subset).lower()

# Function to validate JSON output
def validate_json_output(json_path):
    try:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Output JSON file not found: {json_path}")
        
        if os.path.getsize(json_path) == 0:
            raise ValueError(f"Output JSON file is empty: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)
            if not data:
                raise ValueError(f"Output JSON file contains invalid or empty data: {json_path}")

        return data

    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        return str(e)  # Return error message for logging purposes

# Generate the power set of features
# feature_subsets = list(itertools.chain.from_iterable(itertools.combinations(features, r) for r in range(len(features)+1)))
feature_subsets = [features]

# Set values for the parameters
training_limit = 50
testing_limit = 25
# training_dataset = "../data/qanta.buzztrain.json.gz"
# test_dataset = "../data/qanta.buzzdev.json.gz"
training_dataset = "../data/qanta.buzztrain.json.gz"
test_dataset = "../data/qanta.buzzdev.json.gz"
evaluation = "buzzer"
# guesser_model_train = "../models/buzztrain_gpr_cache"
# guesser_model_test = "../models/buzzdev_gpr_cache"
guesser_model_train = "../models/buzztrain_gpt4o_cache"
guesser_model_test = "../models/buzzdev_gpt4o_cache"

# buzzer_type = "MLP"
# buzzer_type = "LogisticBuzzer"
buzzer_type = "RNNBuzzer"

# Add loss debugging
# if buzzer_type == "MLP":
#     from mlp_buzzer import MLPBuzzer
#     buzzer = MLPBuzzer()
#     print(f"Loss function: {buzzer.loss_function}")
#     print(f"MLP Prediction Threshold: {buzzer.model.threshold if hasattr(buzzer.model, 'threshold') else 0.5}")

# Loop over each subset of features to construct and execute the commands
for subset in feature_subsets:

    # Determine the filename stem based on the subset
    filename_stem = generate_filename_stem(subset, buzzer_type)

    
    # TRAINING SPECS
    # Construct the `buzzer.py` command using sys.executable
    buzzer_command = [
        sys.executable, 'buzzer.py', '--guesser_type=Gpr', '--limit', str(training_limit),
        '--GprGuesser_filename', guesser_model_train,
        '--questions', training_dataset, '--buzzer_guessers', 'Gpr',
        '--buzzer_type', buzzer_type
    ]
        

    
    #TESTING SPECS
    # Construct the `eval.py` command using sys.executable
    output_json = f"summary/eval_output_{filename_stem}.json"
    eval_command = [
        sys.executable, 'eval.py', '--guesser_type=Gpr',
        '--TfidfGuesser_filename=models/TfidfGuesser', '--limit', str(testing_limit),
        '--questions', test_dataset, '--buzzer_guessers', 'Gpr',
        '--GprGuesser_filename', guesser_model_test,
        # '--LogisticBuzzer_filename=models/' + filename_stem,
        '--evaluate', evaluation,
        '--buzzer_type', buzzer_type,
        '--output_json', output_json  # Include output_json flag to specify unique output
    ]
    if buzzer_type == "MLP":
        buzzer_filename_flag = ['--MLPBuzzer_filename=models/' + filename_stem]
        buzzer_command.extend(buzzer_filename_flag)
        eval_command.extend(buzzer_filename_flag)
    elif buzzer_type == "RNNBuzzer":
        buzzer_filename_flag = ['--RNNBuzzer_filename=models/' + filename_stem, '--rnn_hidden_size=128']
        buzzer_command.extend(buzzer_filename_flag)
        eval_command.extend(buzzer_filename_flag)
    else:
        buzzer_filename_flag = ['--LogisticBuzzer_filename=models/' + filename_stem]
        buzzer_command.extend(buzzer_filename_flag)
        eval_command.extend(buzzer_filename_flag)
        

    # Only add --features if subset is not empty
    if subset:
        feature_flag = ['--features'] + list(subset)
        buzzer_command.extend(feature_flag)
        eval_command.extend(feature_flag)
    
    error_log_file = f"summary/error_log_{filename_stem}.txt"
    
    try:
        # Log start of commands
        print(f"Running with feature subset: {subset} -> {filename_stem}")
        time.sleep(1)
        # Run the buzzer.py command
        subprocess.run(buzzer_command, check=True)
        
        # Add an explicit delay to ensure I/O has sufficient time to complete
        time.sleep(2)
        
        eval_output_log = f"evals/eval_output_{filename_stem}.txt"
        with open(eval_output_log, "w") as out_f, open(error_log_file, "w") as err_f:
            subprocess.run(eval_command, stdout=out_f, stderr=err_f, check=True)


        # Add an explicit delay before checking output
        time.sleep(2)

        # Retry logic for validating the output
        max_retries = 3
        retry_delay = 2  # seconds
        for attempt in range(max_retries):
            validation_result = validate_json_output(output_json)
            if isinstance(validation_result, dict):
                # Successfully validated
                eval_results = validation_result
                break
            else:
                # Log the retry attempt
                with open(error_log_file, "a") as err_f:
                    err_f.write(f"Attempt {attempt + 1}: {validation_result}\n")
                time.sleep(retry_delay)
        else:
            # If all retries fail, raise an error
            raise ValueError(f"Failed to validate JSON output after {max_retries} attempts: {output_json}")

        loss_function = LOSS_FUNCTIONS.get(buzzer_type, "Unknown")

        # Create a DataFrame for the new row
        new_row_df = pd.DataFrame([{
            "Features": list(subset),
            "Buzzer Type": buzzer_type,
            "Filename Stem": filename_stem,
            "Loss Function": loss_function,  # Include the loss function dynamically
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
        }])
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)
        # # Validate that the new row is not a duplicate of existing rows
        # columns_to_check = results_df.columns[results_df.columns.get_loc("waiting %"):]
        # if not results_df[columns_to_check].duplicated().any():
        #     # Use pd.concat to add the new row to results_df
        #     
        # else:
        #     print(f"Warning: Duplicate row detected for subset {subset}. Skipping row addition.")

    except Exception as e:
        # Detailed error logging
        with open(error_log_file, "a") as err_file:
            err_file.write(f"Error for subset {subset}: {e}\n")
            err_file.write(f"Buzzer command: {' '.join(buzzer_command)}\n")
            err_file.write(f"Eval command: {' '.join(eval_command)}\n")
            if os.path.exists(output_json) and os.path.getsize(output_json) > 0:
                err_file.write("Output JSON file was partially written or corrupted.\n")
            else:
                err_file.write("Output JSON file was empty or not generated.\n")

        print(f"Subset {subset} generated an exception: {e}. Check {error_log_file} for details.")
        continue

# Sort the DataFrame by descending order of Buzz Ratio
if not results_df.empty:
    columns_to_check = results_df.columns[results_df.columns.get_loc("waiting %"):]
    results_df = results_df.sort_values(by="Buzz Ratio", ascending=False)

    # Validate and remove duplicate rows
    duplicates = results_df.duplicated(subset=columns_to_check, keep=False)
    if duplicates.any():
        print("Warning: Duplicate rows found in the CSV output.")
        duplicate_rows = results_df[duplicates]
        duplicate_log_path = f"summary/{filename_stem}_duplicate_rows_log.csv"
        duplicate_rows.to_csv(duplicate_log_path, index=False)
        print(f"Duplicate rows have been saved to {duplicate_log_path}")

        # Remove duplicates and save a new CSV without them
        results_df.drop_duplicates(subset=columns_to_check, keep='first', inplace=True)

    # Export the DataFrame as CSV
    eval_summary_file_name = f"summary/{filename_stem}_eval_summary.csv"
    results_df.to_csv(eval_summary_file_name, index=False)
    print("Evaluation data exported to: ", eval_summary_file_name)
else:
    print("No results were generated, possibly due to errors in processing.")
