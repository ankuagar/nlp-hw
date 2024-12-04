import itertools
import os
import subprocess
import sys
import pandas as pd
import json
import time

LOSS_FUNCTIONS = {
    "MLP": "BuzzLoss",
    "LogisticBuzzer": "Logistic Loss",
    "RNNBuzzer": "CrossEntropyLoss"
}

# Define the features to use in generating the power set
features = ["Length", "Frequency", "Category", "ContextualMatch"]

# DataFrame to store the best result for each buzzer type
results_df = pd.DataFrame(columns=[
    "Features", "Buzzer Type", "Filename Stem", "Loss Function", "Training Limit", "Testing Limit",
    "Training Dataset", "Test Dataset", "Evaluation",
    "best %", "timid %", "hit %", "close %", "miss %", "aggressive %", "waiting %",
    "Questions Right", "Total", "Accuracy", "Buzz Ratio", "Buzz Position", "Error Log"
])

# Function to generate the filename stem based on the subset of features
def generate_filename_stem(subset, buzzer_type="LogisticBuzzer"):
    buzzer_str = "logit" if buzzer_type == "LogisticBuzzer" else buzzer_type.lower()
    if not subset:
        return f"{buzzer_str}_no_features"
    elif set(subset) == set(features):
        return f"{buzzer_str}_with_all_features"
    else:
        return f"{buzzer_str}_with_" + "_".join(subset).lower()

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
feature_subsets = list(itertools.chain.from_iterable(itertools.combinations(features, r) for r in range(len(features)+1)))

# Set values for the parameters
training_limit = 50
testing_limit = 25
training_dataset = "../data/qanta.buzztrain.json.gz"
test_dataset = "../data/qanta.buzzdev.json.gz"
evaluation = "buzzer"
guesser_model_train = "../models/buzztrain_gpt4o_cache"
guesser_model_test = "../models/buzzdev_gpt4o_cache"

# List of buzzer models
buzzer_models = ["MLP", "LogisticBuzzer", "RNNBuzzer"]

# Main loop to iterate over buzzer models and feature subsets
for buzzer_type in buzzer_models:
    print(f"Running for buzzer type: {buzzer_type}")

    best_result = None  # To track the best model for this buzzer type

    for subset in feature_subsets:
        filename_stem = generate_filename_stem(subset, buzzer_type)
        buzzer_command = [
            sys.executable, 'buzzer.py', '--guesser_type=Gpr', '--limit', str(training_limit),
            '--GprGuesser_filename', guesser_model_train,
            '--questions', training_dataset, '--buzzer_guessers', 'Gpr',
            '--buzzer_type', buzzer_type
        ]

        if buzzer_type == "MLP":
            buzzer_filename_flag = ['--MLPBuzzer_filename=models/' + filename_stem]
        elif buzzer_type == "RNNBuzzer":
            buzzer_filename_flag = ['--RNNBuzzer_filename=models/' + filename_stem, '--rnn_hidden_size=128']
        else:  # LogisticBuzzer
            buzzer_filename_flag = ['--LogisticBuzzer_filename=models/' + filename_stem]

        buzzer_command.extend(buzzer_filename_flag)

        if subset:
            feature_flag = ['--features'] + list(subset)
            buzzer_command.extend(feature_flag)

        output_json = f"summary/eval_output_{filename_stem}.json"
        eval_command = [
            sys.executable, 'eval.py', '--guesser_type=Gpr',
            '--TfidfGuesser_filename=models/TfidfGuesser', '--limit', str(testing_limit),
            '--questions', test_dataset, '--buzzer_guessers', 'Gpr',
            '--GprGuesser_filename', guesser_model_test,
            '--evaluate', evaluation,
            '--output_json', output_json,
            '--buzzer_type', buzzer_type,
        ]
        eval_command.extend(buzzer_filename_flag)
        if subset:
            eval_command.extend(feature_flag)

        error_log_file = f"summary/error_log_{filename_stem}.txt"

        try:
            # Train the model
            print(f"Running training for subset: {subset}")
            subprocess.run(buzzer_command, check=True)

            # Evaluate the model
            print(f"Running evaluation for subset: {subset}")
            eval_output_log = f"evals/eval_output_{filename_stem}.txt"
            with open(eval_output_log, "w") as out_f, open(error_log_file, "w") as err_f:
                subprocess.run(eval_command, stdout=out_f, stderr=err_f, check=True)

            # Validate the output
            eval_results = validate_json_output(output_json)
            if isinstance(eval_results, str):  # If validation fails, log and skip
                raise ValueError(eval_results)

            loss_function = LOSS_FUNCTIONS.get(buzzer_type, "Unknown")

            new_result = {
                "Features": list(subset),
                "Buzzer Type": buzzer_type,
                "Filename Stem": filename_stem,
                "Loss Function": loss_function,
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
                "Buzz Position": eval_results["buzz_position"],
                "Error Log": None
            }

            # Update the best result for this buzzer type
            if not best_result or new_result["Buzz Ratio"] > best_result["Buzz Ratio"]:
                best_result = new_result

        except Exception as e:
            # Log the error and save NaN results for this configuration
            print(f"Error for subset {subset}: {e}")
            with open(error_log_file, "a") as err_file:
                err_file.write(f"Error: {e}\n")
                err_file.write(f"Buzzer command: {' '.join(buzzer_command)}\n")
                err_file.write(f"Eval command: {' '.join(eval_command)}\n")

            new_result = {
                "Features": list(subset),
                "Buzzer Type": buzzer_type,
                "Filename Stem": filename_stem,
                "Loss Function": LOSS_FUNCTIONS.get(buzzer_type, "Unknown"),
                "Training Limit": training_limit,
                "Testing Limit": testing_limit,
                "Training Dataset": training_dataset,
                "Test Dataset": test_dataset,
                "Evaluation": evaluation,
                "best %": None, "timid %": None, "hit %": None, "close %": None,
                "miss %": None, "aggressive %": None, "waiting %": None,
                "Questions Right": None, "Total": None,
                "Accuracy": None, "Buzz Ratio": None, "Buzz Position": None,
                "Error Log": str(e)
            }

        results_df = pd.concat([results_df, pd.DataFrame([new_result])], ignore_index=True)

    # Append the best result for this buzzer type to the final DataFrame
    if best_result:
        results_df = pd.concat([results_df, pd.DataFrame([best_result])], ignore_index=True)

# Export the final results
results_df.to_csv("summary/best_eval_summary.csv", index=False)
