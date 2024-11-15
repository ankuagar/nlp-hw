# Buzzing Classifier Project

## Overview

This project focuses on building and improving a classification system for determining the correctness of responses in a quiz-like environment. The system consists of two main components:

1. **Guesser**: Generates possible answers based on partial input from the question.
2. **Buzzer**: Evaluates the guesses and decides whether to "buzz in" based on response quality.

The primary goal is to enhance the predictive power of the Buzzer by:
- Extracting meaningful information from the Guesser's responses.
- Engineering better features to inform the classifier's decisions.
- Expanding and refining the dataset.

While accuracy is important, our key performance metric is the system's ability to make reliable buzz decisions, as elaborated further in the project.

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

## Steps to Train the Classifier

### 1. Train Without New Features

To establish a baseline, train the Buzzer using the existing features only. This helps in understanding the model's current performance:

```bash
python3 buzzer.py --guesser_type=Gpr --limit=50 \
  --GprGuesser_filename=models/buzztrain_gpr_cache \
  --questions=data/qanta.buzztrain.json.gz --buzzer_guessers Gpr \
  --LogisticBuzzer_filename=models/no_feature --features ""

### Steps to Train the Classifier

#### 1. Train Without New Features

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

## Compare Models
After training, we compare the performance of models with and without the new feature using the evaluation metrics provided by eval.py. For more details on evaluation, see the Evals Folder.

