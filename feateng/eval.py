# Jordan Boyd-Graber
# 2023
#
# Run an evaluation on a QA system and print results
import json
import os
import random
import string
import logging
import torch
from mlp_buzzer import MLPBuzzer
from tqdm import tqdm
from params import load_guesser, load_questions, load_buzzer, \
    add_buzzer_params, add_guesser_params, add_general_params, \
    add_question_params, setup_logging

# Ensure the summary directory exists
os.makedirs("summary", exist_ok=True)

kLABELS = {
    "best": "Guess was correct, Buzz was correct",
    "timid": "Guess was correct, Buzz was not",
    "hit": "Guesser ranked right page first",
    "close": "Guesser had correct answer in top n list",
    "miss": "Guesser did not have correct answer in top n list",
    "aggressive": "Guess was wrong, Buzz was wrong",
    "waiting": "Guess was wrong, Buzz was correct"
}

def normalize_answer(answer):
    """
    Normalize an answer string for easier comparison.
    """
    from unidecode import unidecode

    if answer is None:
        return ''
    reduced = unidecode(answer).replace("_", " ")
    if "(" in reduced:
        reduced = reduced.split("(")[0]
    reduced = "".join(x for x in reduced.lower() if x not in string.punctuation).strip()

    for bad_start in ["the ", "a ", "an "]:
        if reduced.startswith(bad_start):
            reduced = reduced[len(bad_start):]
    return reduced.strip()

def rough_compare(guess, page):
    """
    Determine if a guess matches an answer, allowing for minor variations.
    """
    if page is None:
        return False

    guess = normalize_answer(guess)
    page = normalize_answer(page)

    if guess == '':
        return False

    return guess == page or (page.find(guess) >= 0 and (len(page) - len(guess)) / len(page) > 0.5)

def eval_retrieval(guesser, questions, n_guesses=25, cutoff=-1):
    """
    Evaluate the retrieval accuracy of a guesser.
    """
    from collections import Counter, defaultdict
    outcomes = Counter()
    examples = defaultdict(list)

    question_text = []
    for question in tqdm(questions):
        text = question["text"][:cutoff] if cutoff > 0 else question["text"]
        question_text.append(text)

    all_guesses = guesser.batch_guess(question_text, n_guesses)
    for question, guesses, text in zip(questions, all_guesses, question_text):
        guesses = guesses[:n_guesses] if len(guesses) > n_guesses else guesses
        top_guess = guesses[0]["guess"]
        answer = question["page"]

        example = {"text": text, "guess": top_guess, "answer": answer, "id": question["qanta_id"]}

        if any(rough_compare(x["guess"], answer) for x in guesses):
            outcomes["close"] += 1
            if rough_compare(top_guess, answer):
                outcomes["hit"] += 1
                examples["hit"].append(example)
            else:
                examples["close"].append(example)
        else:
            outcomes["miss"] += 1
            examples["miss"].append(example)

    return outcomes, examples
def pretty_feature_print(features, first_features=["guess", "answer", "id"]):
    """
    Nicely print a buzzer example's features
    """
    import textwrap
    wrapper = textwrap.TextWrapper()

    lines = []

    for ii in first_features:
        lines.append("%20s: %s" % (ii, features[ii]))
    for ii in [x for x in features if x not in first_features]:
        if isinstance(features[ii], str):
            if len(features[ii]) > 70:
                long_line = "%20s: %s" % (ii, "\n                      ".join(wrapper.wrap(features[ii])))
                lines.append(long_line)
            else:
                lines.append("%20s: %s" % (ii, features[ii]))
        elif isinstance(features[ii], float):
            lines.append("%20s: %0.4f" % (ii, features[ii]))
        else:
            lines.append("%20s: %s" % (ii, str(features[ii])))
    lines.append("--------------------")
    return "\n".join(lines)

def eval_buzzer(buzzer, questions, history_length, history_depth):
    """
    Evaluate a buzzer's performance on a dataset.
    """
    from collections import Counter, defaultdict


    buzzer.load()
    buzzer.add_data(questions)
    buzzer.build_features(history_length=history_length, history_depth=history_depth)

    if hasattr(buzzer, "model"):  # Only for MLPBuzzer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        buzzer.model.to(device)

    # Predict buzz decisions
    if buzzer_type == "MLP":    
        predict, feature_matrix, feature_dict, correct, metadata = buzzer.predict(buzzer._features)
    elif buzzer_type == "RNNBuzzer":
        predict, feature_matrix, feature_dict, correct, metadata = buzzer.predict(questions)
    else :
        predict, feature_matrix, feature_dict, correct, metadata = buzzer.predict(questions)

    # Debugging: Log predictions and features
    print(f"Predictions (raw): {predict}")  # Raw predictions (probabilities or binary decisions)
    print(f"Feature Matrix Shape: {feature_matrix.shape}")  # Check feature dimensions
    print(f"Feature Dictionary Sample: {feature_dict[:5]}")  # Log a sample of features
    print(f"Correct Labels: {correct[:5]}")  # Check the ground truth labels

    # Keep track of how much of the question you needed to see before answering correctly
    question_seen = {}
    question_length = defaultdict(int)

    outcomes = Counter()
    examples = defaultdict(list)
    for buzz, guess_correct, features, meta in zip(predict, correct, feature_dict, metadata):
        qid = meta["id"]

        # Add back in metadata now that we have prevented cheating in feature creation
        for ii in meta:
            features[ii] = meta[ii]

        # Keep track of the longest run we saw for each question
        question_length[qid] = max(question_length[qid], len(meta["text"]))


        if guess_correct:
            if buzz:
                outcomes["best"] += 1
                examples["best"].append(features)

                if qid not in question_seen:
                    question_seen[qid] = len(meta["text"])
            else:
                outcomes["timid"] += 1
                examples["timid"].append(features)
        else:
            if buzz:
                outcomes["aggressive"] += 1
                examples["aggressive"].append(features)

                if qid not in question_seen:
                    question_seen[qid] = -len(meta["text"])
            else:
                outcomes["waiting"] += 1
                examples["waiting"].append(features)


    unseen_characters = 0.0
    for question, length in question_length.items():
        if question in question_seen:
            seen = question_seen[question]
            unseen_characters += (1.0 - seen / length) if seen > 0 else (-1.0 - seen / length)

    unseen_characters /= len(question_length)
    # Debugging: Log outcome counts
    print(f"Outcomes: {outcomes}")
    print(f"Examples per Outcome: { {k: len(v) for k, v in examples.items()} }")

    return outcomes, examples, unseen_characters


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    add_general_params(parser)
    add_guesser_params(parser)
    add_question_params(parser)
    add_buzzer_params(parser)
    parser.add_argument('--evaluate', default="buzzer", type=str)
    parser.add_argument('--cutoff', default=-1, type=int)
    parser.add_argument('--output_json', type=str, default="summary/eval_output.json", help="Path to save output JSON file")

    flags = parser.parse_args()
    print("In eval.py received flags", flags)
    setup_logging(flags)

    questions = load_questions(flags)
    guesser = load_guesser(flags, load=flags.load)

    if flags.evaluate == "buzzer":
        buzzer = load_buzzer(flags, load=True)
        buzzer_type = flags.buzzer_type
        outcomes, examples, unseen = eval_buzzer(buzzer, questions,
                                                 history_length=flags.buzzer_history_length,
                                                 history_depth=flags.buzzer_history_depth)
        # Debugging evaluation
        if flags.buzzer_type == "MLP":
            print("MLP Buzzer Evaluation Started")

    elif flags.evaluate == "guesser":
        outcomes, examples = eval_retrieval(guesser, questions, flags.num_guesses, flags.cutoff)
    else:
        raise ValueError("Specify a valid evaluation target: 'buzzer' or 'guesser'.")

    total = sum(outcomes[x] for x in outcomes if x != "hit")
    outcome_percentages = {f"{key} %": value / total for key, value in outcomes.items()}

    for key in outcomes:
        print(f"{key} {outcomes[key] / total:.2f}\n===================")
        sample_examples = examples[key][:10] if len(examples[key]) > 10 else examples[key]
        for example in sample_examples:
            print(pretty_feature_print(example))
        print("=================")

    if flags.evaluate == "buzzer" and flags.buzzer_type == "LogisticBuzzer":
        for weight, feature in zip(buzzer._classifier.coef_[0], buzzer._featurizer.feature_names_):
            print(f"{feature.strip():>40}: {weight:.4f}")
    elif flags.evaluate == "buzzer" and flags.buzzer_type == "RNNBuzzer":
        print("RNNBuzzer does not support linear feature inspection.")

    results = {
        "questions_right": outcomes["best"],
        "total": total,
        "accuracy": (outcomes["best"] + outcomes["waiting"]) / total,
        "buzz_ratio": (outcomes["best"] - outcomes["aggressive"] * 0.5) / total,
        "buzz_position": unseen,
        "outcome_percentages": outcome_percentages
    }

    with open(flags.output_json, "w") as f:
        json.dump(results, f)
