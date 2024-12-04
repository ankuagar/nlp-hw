# Buzzing Classifier Project

## Overview

This project focuses on building and improving a classification system for determining the correctness of responses in a quiz-like environment. The system consists of two main components:

1. **Guesser**: Generates possible answers based on partial input from the question.
2. **Buzzer**: Evaluates the guesses and decides whether to "buzz in" based on response quality.

The primary goals of the project are:
- Enhancing the predictive power of the buzzer by extracting meaningful information from the guesser's responses.
- Engineering better features to inform the buzzer's decisions.
- Expanding the buzzer to use more advanced architectures, such as RNNs and MLPs, alongside traditional logistic regression.

The system's performance is measured by its ability to make reliable buzz decisions while maintaining high prediction accuracy.

---

## Getting Started

### Prerequisites

Ensure you have the necessary packages installed. It is recommended to work within a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

If you encounter missing stopwords in NLTK, download them:
``` bash
python -m nltk.downloader stopwords
```

Create a directory to store the models you will generate:

```bash
mkdir -p models
```

## Project Structure

The files are organized as follows:

1. features.py: Contains feature definitions. You will add new features here.
2. params.py: Manages feature initialization. Update this file to integrate new features.
3. buzzer.py: Implements the Buzzer, including feature extraction and classifier training.
4. eval.py: Evaluates the trained Buzzer using various metrics.
5. logistic_buzzer.py: Implements the LogisticBuzzer, a traditional logistic regression-based classifier.
6. rnn_buzzer.py: Contains the implementation of the RNNBuzzer, which uses a recurrent neural network for classification.
7. mlp_buzzer.py: Contains the implementation of the MLPBuzzer, which uses a multilayer perceptron for classification.
8. compare_buzzers.py: Automates training and evaluation for different buzzer models and feature combinations.
   
## Steps to Train the Classifier

### 1. Train Without New Features

To establish a baseline, train the Buzzer using the existing features only. This helps in understanding the model's current performance:

```bash
python3 buzzer.py --guesser_type=Gpr --limit=50 \
  --GprGuesser_filename=models/buzztrain_gpr_cache \
  --questions=data/qanta.buzztrain.json.gz --buzzer_guessers Gpr \
  --LogisticBuzzer_filename=models/no_feature --features ""
```

#### 2. Train With New Features 

Once you’ve added a new feature, train the Buzzer to include it:
```bash
python3 buzzer.py --guesser_type=Gpr --limit=50 \
  --GprGuesser_filename=models/buzztrain_gpr_cache \
  --questions=data/qanta.buzztrain.json.gz --buzzer_guessers Gpr \
  --LogisticBuzzer_filename=models/with_feature --features NewFeatureName
```

Replace NewFeatureName with the name(s) of the feature(s).

** Integration **:
The features are correctly implemented in features.py.
They are instantiated in params.py and included in the feature set.

## Train and Evaluate RNNBuzzer
#### 1.Train RNNBuzzer
To train the RNNBuzzer without new features:
```bash
feateng % python3 buzzer.py --guesser_type=Gpr --limit=50 \
  --GprGuesser_filename=../models/buzztrain_gpr_cache \
  --questions=../data/qanta.buzztrain.json.gz --buzzer_guessers Gpr \
  --buzzer_type=RNNBuzzer --RNNBuzzer_filename=models/rnn_no_feature \
  --features "" --rnn_hidden_size=128

```
To train the RNNBuzzer with new features:
```bash
python3 buzzer.py --guesser_type=Gpr --limit=50 \\
  --GprGuesser_filename=models/buzztrain_gpr_cache \\
  --questions=data/qanta.buzztrain.json.gz --buzzer_guessers Gpr \\
  --buzzer_type=RNNBuzzer --RNNBuzzer_filename=models/rnn_with_feature \\
  --features NewFeatureName --rnn_hidden_size=128
```
#### 2.Evaluate RNNBuzzer
```bash
feateng % .venv/bin/python3 eval.py --guesser_type=Gpr \
--TfidfGuesser_filename=../models/TfidfGuesser --limit=25 \
--questions=../data/qanta.buzzdev.json.gz --buzzer_guessers Gpr \
--GprGuesser_filename=../models/buzzdev_gpr_cache \
--buzzer_type=RNNBuzzer --RNNBuzzer_filename=models/rnn_no_feature \
--features "" --rnn_hidden_size=128
```

## Train and Evaluate MLPBuzzer
#### 1.Train MLBuzzer
To train the MLPBuzzer without new features:
```bash
python3 buzzer.py --guesser_type=Gpr --limit=50 \\
  --GprGuesser_filename=models/buzztrain_gpr_cache \\
  --questions=data/qanta.buzztrain.json.gz --buzzer_guessers Gpr \\
  --buzzer_type=MLP --MLPBuzzer_filename=models/mlp_no_feature \\
  --features "" --mlp_hidden_size=128 --mlp_num_layers=2

```
To train the MLPBuzzer with new features:
```bash
python3 buzzer.py --guesser_type=Gpr --limit=50 \\
  --GprGuesser_filename=models/buzztrain_gpr_cache \\
  --questions=data/qanta.buzztrain.json.gz --buzzer_guessers Gpr \\
  --buzzer_type=MLP --MLPBuzzer_filename=models/mlp_with_feature \\
  --features NewFeatureName --mlp_hidden_size=128 --mlp_num_layers=2
```
#### 2.Evaluate MLPBuzzer
```bash
python3 eval.py --guesser_type=Gpr \\
  --TfidfGuesser_filename=models/TfidfGuesser --limit=25 \\
  --questions=data/qanta.buzzdev.json.gz --buzzer_guessers Gpr \\
  --buzzer_type=MLP --MLPBuzzer_filename=models/mlp_with_feature \\
  --features NewFeatureName --mlp_hidden_size=128 --mlp_num_layers=2
```


## Compare Models
To compare multiple models and features, use compare_buzzers.py. This script automates training and evaluation across feature subsets and buzzer types. It produces a summary CSV (summary/best_eval_summary.csv) with the best model configuration for each buzzer type.

Run the script:
```bash
python3 compare_buzzers.py
```

## Outputs
1. Trained Models: Saved in the models/ directory, named based on buzzer type and features.
2. Evaluation Results: JSON and log files stored in the summary/ directory.
3. Summary CSV: The summary/best_eval_summary.csv file contains metrics for the best configuration of each buzzer type.

