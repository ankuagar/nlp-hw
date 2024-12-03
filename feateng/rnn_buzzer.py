# # rnn_buzzer.py
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from buzzer import Buzzer

class RNNModel(nn.Module):
    """
    A simple RNN model for classification.
    """
    def __init__(self, input_size, hidden_size, output_size=2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, hidden = self.rnn(x)
        output = self.fc(hidden.squeeze(0))
        return output


class RNNBuzzer(Buzzer):
    """
    RNN-based classifier to predict whether a buzz is correct or not.
    """

    def __init__(self, filename, run_length, num_guesses=1, hidden_size=128):
        super().__init__(filename, run_length, num_guesses)
        self.hidden_size = hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        # Featurize and vectorize data
        X = Buzzer.train(self)
        self.model = RNNModel(input_size=X.shape[1], hidden_size=self.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Convert data to tensors
        X_tensor = torch.tensor(X.todense(), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(self._correct, dtype=torch.long).to(self.device)

        # Reshape for RNN
        X_tensor = X_tensor.unsqueeze(1)  # Add sequence dimension

        # Training loop
        epochs = 10
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = self.criterion(output, y_tensor)
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def single_predict(self, run):
        """
        Predict from a single example.
        """
        guess, features = self.featurize(None, run)
        X = self._featurizer.transform([features])
        X_tensor = torch.tensor(X.todense(), dtype=torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X_tensor)
            predicted_label = torch.argmax(prediction, dim=1).item()
        return predicted_label, guess, features

    def predict(self, questions, online=False):
        """
        Predict from a set of questions.
        """
        assert self.model, "Model not trained"
        X = self._featurizer.transform(self._features)
        X_tensor = torch.tensor(X.todense(), dtype=torch.float32).to(self.device).unsqueeze(1)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predicted_labels = torch.argmax(predictions, dim=1).cpu().tolist()
        return predicted_labels, X, self._features, self._correct, self._metadata

    def save(self):
        """
        Save the RNN model and featurizer.
        """
        Buzzer.save(self)
        torch.save(self.model.state_dict(), f"{self.filename}.rnn_model.pth")

    def load(self):
        """
        Load the RNN model and featurizer.
        """
        Buzzer.load(self)
        input_size = len(self._featurizer.feature_names_)  # Get the number of features
        self.model = RNNModel(input_size=input_size, hidden_size=self.hidden_size).to(self.device)
        self.model.load_state_dict(torch.load(f"{self.filename}.rnn_model.pth"))

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from buzzer import Buzzer
# class RNNBuzzer(Buzzer):
#     """
#     A subclass of Buzzer that uses an RNN instead of logistic regression.
#     """
#     def __init__(self, filename, run_length, num_guesses=1, input_dim=50, hidden_dim=100, output_dim=2):
#         super().__init__(filename, run_length, num_guesses)
#         self.rnn_model = RNNModel(input_dim, hidden_dim, output_dim)
#         self.optimizer = optim.Adam(self.rnn_model.parameters(), lr=0.001)
#         self.criterion = nn.CrossEntropyLoss()
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.rnn_model.to(self.device)

#     def train_rnn(self, features, labels, epochs=10, batch_size=32):
#         """
#         Train the RNN model on the provided features and labels.
#         """
#         dataset = TensorDataset(torch.tensor(features, dtype=torch.float32),
#                                 torch.tensor(labels, dtype=torch.long))
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#         self.rnn_model.train()
#         for epoch in range(epochs):
#             total_loss = 0
#             for batch_features, batch_labels in dataloader:
#                 batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
#                 self.optimizer.zero_grad()
#                 outputs = self.rnn_model(batch_features)
#                 loss = self.criterion(outputs, batch_labels)
#                 loss.backward()
#                 self.optimizer.step()
#                 total_loss += loss.item()
#             print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

#     def predict_rnn(self, features):
#         """
#         Predict using the trained RNN model.
#         """
#         self.rnn_model.eval()
#         with torch.no_grad():
#             features = torch.tensor(features, dtype=torch.float32).to(self.device)
#             outputs = self.rnn_model(features)
#             predictions = torch.argmax(outputs, dim=1)
#         return predictions.cpu().numpy()

#     def featurize_and_train(self, questions, history_length=0, history_depth=0, epochs=10, batch_size=32):
#         """
#         Featurize data, train RNN, and store metadata.
#         """
#         features = self.build_features(history_length, history_depth)
#         labels = torch.tensor(self._correct, dtype=torch.long)
#         self.train_rnn(features, labels, epochs, batch_size)

#     def featurize_and_predict(self, questions, history_length=0, history_depth=0):
#         """
#         Featurize questions and make predictions using the RNN model.
#         """
#         features = self.build_features(history_length, history_depth)
#         return self.predict_rnn(features)

#     def add_guesser(self, guesser_name, guesser, primary_guesser=False):
#         """
#         Add a guesser identified by guesser_name to the set of guessers.
#         """
#         super().add_guesser(guesser_name, guesser, primary_guesser)

#     def finalize(self):
#         """
#         Set the guessers (prevents further additions of features and guessers).
#         """
#         super().finalize()

#     def save(self):
#         """
#         Save the trained RNN model and featurizer to disk.
#         """
#         super().save()
#         torch.save(self.rnn_model.state_dict(), f"{self.filename}_rnn_model.pth")
#         print(f"Model saved to {self.filename}_rnn_model.pth")

#     def load(self):
#         """
#         Load the trained RNN model and featurizer from disk.
#         """
#         super().load()
#         self.rnn_model.load_state_dict(torch.load(f"{self.filename}_rnn_model.pth"))
#         self.rnn_model.to(self.device)
#         print(f"Model loaded from {self.filename}_rnn_model.pth")

# if __name__ == "__main__":
#     import argparse
#     from buzzer import load_questions, add_general_params, add_question_params, add_guesser_params, setup_logging, load_guesser

#     parser = argparse.ArgumentParser()
#     add_general_params(parser)
#     add_question_params(parser)
#     add_guesser_params(parser)

#     parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs for RNN.")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size for RNN training.")
#     parser.add_argument("--hidden_dim", type=int, default=100, help="Hidden dimension of the RNN.")
#     parser.add_argument("--input_dim", type=int, default=50, help="Input dimension of the RNN.")
#     parser.add_argument("--output_dim", type=int, default=2, help="Output dimension of the RNN.")
#     parser.add_argument("--RNNBuzzer_filename", type=str, required=True, help="Filename to save/load the RNN model.")

#     flags = parser.parse_args()
#     setup_logging(flags)

#     # Load guesser
#     guesser = load_guesser(flags)

#     # Load questions
#     questions = load_questions(flags)

#     # Initialize RNNBuzzer
#     rnn_buzzer = RNNBuzzer(
#         filename=flags.RNNBuzzer_filename,
#         run_length=flags.run_length,
#         input_dim=flags.input_dim,
#         hidden_dim=flags.hidden_dim,
#         output_dim=flags.output_dim,
#     )

#     # Add guesser to the buzzer
#     rnn_buzzer.add_guesser("GprGuesser", guesser, primary_guesser=True)

#     # Add data to the buzzer
#     rnn_buzzer.add_data(questions)

#     # Train and predict
#     rnn_buzzer.featurize_and_train(
#         questions,
#         history_length=flags.buzzer_history_length,
#         history_depth=flags.buzzer_history_depth,
#         epochs=flags.epochs,
#         batch_size=flags.batch_size,
#     )
#     predictions = rnn_buzzer.featurize_and_predict(questions)

#     print("Predictions:", predictions)


# # # Author: Jordan Boyd-Graber
# # # 2013

# # import argparse
# # import logging
# # import pickle
# # import numpy as np
# # import chainer.functions as F
# # import chainer.links as L
# # from chainer import reporter

# # from sklearn.feature_extraction import DictVectorizer
# # from tqdm import tqdm

# # from collections import Counter
# # from collections import defaultdict

# # from guesser import add_guesser_params
# # from features import LengthFeature
# # from features import ContextualMatchFeature
# # from features import FrequencyFeature
# # from features import PreviousGuessFeature
# # from features import CategoryFeature
# # from params import add_buzzer_params, add_question_params, load_guesser, load_buzzer, load_questions, add_general_params, setup_logging

# # def runs(text, run_length):
# #     """
# #     Given a quiz bowl questions, generate runs---subsegments that simulate
# #     reading the question out loud.

# #     These are then fed into the rest of the system.

# #     """
# #     words = text.split()
# #     assert len(words) > 0
# #     current_word = 0
# #     last_run = 0

# #     for idx in range(run_length, len(text), run_length):
# #         current_run = text.find(" ", idx)
# #         if current_run > last_run and current_run < idx + run_length:
# #             yield text[:current_run]
# #             last_run = current_run

# #     yield text

# # def sentence_runs(sentences, run_length):
# #     """
# #     Generate runs, but do it per sentence (always stopping at sentence boundaries).
# #     """
    
# #     previous = ""
# #     for sentence in sentences:
# #         for run in runs(sentence, run_length):
# #             yield previous + run
# #         previous += sentence
# #         previous += "  "
    
# # class Buzzer:
# #     """
# #     Base class for any system that can decide if a guess is correct or not.
# #     """
    
# #     def __init__(self, filename, run_length, num_guesses=1):
# #         self.filename = filename
# #         self.num_guesses = num_guesses
# #         self.run_length=run_length
        
# #         self._runs = []
# #         self._questions = []
# #         self._answers = []
# #         self._training = []
# #         self._correct = []
# #         self._features = []
# #         self._metadata = []
# #         self._feature_generators = []
# #         self._guessers = {}

# #         logging.info("Buzzer using run length %i" % self.run_length)
        
# #         self._finalized = False
# #         self._primary_guesser = None
# #         self._classifier = None
# #         self._featurizer = None

# #     def add_guesser(self, guesser_name, guesser, primary_guesser=False):
# #         """
# #         Add a guesser identified by guesser_name to the set of guessers.

# #         If it is designated as the primary_guesser, then its guess will be
# #         chosen in the case of a tie.

# #         """

# #         assert not self._finalized, "Trying to add guesser after finalized"
# #         assert guesser_name != "consensus"
# #         assert guesser_name is not None
# #         assert guesser_name not in self._guessers
# #         self._guessers[guesser_name] = guesser
# #         if primary_guesser:
# #             self._primary_guesser = guesser_name

# #     def add_feature(self, feature_extractor):
# #         """
# #         Add a feature that the buzzer will use to decide to trust a guess.
# #         """

# #         assert not self._finalized, "Trying to add feature after finalized"
# #         assert feature_extractor.name not in [x.name for x in self._feature_generators]
# #         assert feature_extractor.name not in self._guessers
# #         self._feature_generators.append(feature_extractor)
# #         logging.info("Adding feature %s" % feature_extractor.name)
        
# #     def featurize(self, question, run_text, guess_history, guesses=None):
# #         """
# #         Turn a question's run into features.

# #         guesses -- A dictionary of all the guesses.  If None, will regenerate the guesses.
# #         """
        
# #         features = {}
# #         guess = None

# #         # If we didn't cache the guesses, compute them now
# #         if guesses is None:
# #             guesses = {}            
# #             for gg in self._guessers:
# #                 guesses[gg] = self._guessers[gg](run_text)

# #         for gg in self._guessers:
# #             assert gg in guesses, "Missing guess result from %s" % gg
# #             result = list(guesses[gg])[0]
# #             if gg == self._primary_guesser:
# #                 guess = result["guess"]

# #             # This feature could be useful, but makes the formatting messy
# #             # features["%s_guess" % gg] = result["guess"]
# #             features["%s_confidence" % gg] = result["confidence"]

# #         for ff in self._feature_generators:
# #             for feat, val in ff(question, run_text, guess, guess_history):
# #                 features["%s_%s" % (ff.name, feat)] = val

# #         assert guess is not None
# #         return guess, features

# #     def finalize(self):
# #         """
# #         Set the guessers (will prevent future addition of features and guessers)
# #         """
        
# #         self._finalized = True
# #         if self._primary_guesser is None:
# #             self._primary_guesser = "consensus"
        
# #     def add_data(self, questions, answer_field="page"):
# #         """
# #         Add data and store them so you can later create features for them
# #         """
        
# #         self.finalize()
        
# #         num_questions = 0
# #         logging.info("Generating runs of length %i" % self.run_length)        
# #         for qq in tqdm(questions):
# #             answer = qq[answer_field]
# #             text = qq["text"]
# #             # Delete these fields so you can't inadvertently cheat while
# #             # creating features.  However, we need the answer for the labels.
# #             del qq[answer_field]
# #             if "answer" in qq:
# #                 del qq["answer"]
# #             if "page" in qq:
# #                 del qq["page"]
# #             del qq["first_sentence"]
# #             del qq["text"]

# #             for rr in runs(text, self.run_length):
# #                 self._answers.append(answer)
# #                 self._runs.append(rr)
# #                 self._questions.append(qq)

# #     def build_features(self, history_length=0, history_depth=0):
# #         """
# #         After all of the data has been added, build features from the guesses and questions.
# #         """
# #         from eval import rough_compare

# #         all_guesses = {}
# #         logging.info("Building guesses from %s" % str(self._guessers.keys()))
# #         for guesser in self._guessers:
# #             all_guesses[guesser] = self._guessers[guesser].batch_guess(self._runs, self.num_guesses)
# #             logging.info("%10i guesses from %s" % (len(all_guesses[guesser]), guesser))
# #             assert len(all_guesses[guesser]) == len(self._runs), "Guesser %s wrong size" % guesser
            
# #         assert len(self._questions) == len(self._answers)
# #         assert len(self._questions) == len(self._runs)        
            
# #         num_runs = len(self._runs)

# #         logging.info("Generating all features")
# #         for question_index in tqdm(range(num_runs)):
# #             question_guesses = dict((x, all_guesses[x][question_index]) for x in self._guessers)
# #             guess_history = defaultdict(dict)
# #             for guesser in question_guesses:
# #                 # print("Building history with depth %i and length %i" % (history_depth, history_length))
# #                 guess_history[guesser] = dict((time, guess[:history_depth]) for time, guess in enumerate(all_guesses[guesser]) if time < question_index and time > question_index - history_length)

# #             # print(guess_history)
# #             question = self._questions[question_index]
# #             run = self._runs[question_index]
# #             answer = self._answers[question_index]
# #             guess, features = self.featurize(question, run, guess_history, question_guesses)
            
# #             self._features.append(features)
# #             self._metadata.append({"guess": guess, "answer": answer, "id": question["qanta_id"], "text": run})

# #             correct = rough_compare(guess, answer)
# #             logging.debug(str((correct, guess, answer)))
                
# #             self._correct.append(correct)

                
# #             assert len(self._correct) == len(self._features)
# #             assert len(self._correct) == len(self._metadata)
        
# #         assert len(self._answers) == len(self._correct), \
# #             "Answers (%i) does not match correct (%i)" % (len(self._answers), len(self._features))
# #         assert len(self._answers) == len(self._features)        

# #         if "GprGuesser" in self._guessers:
# #             self._guessers["GprGuesser"].save()
            
# #         return self._features
    
# #     def single_predict(self, run):
# #         """
# #         Make a prediction from a single example ... this us useful when the code
# #         is run in real-time.

# #         """
        
# #         guess, features = self.featurize(None, run)

# #         X = self._featurizer.transform([features])

# #         return self._classifier.predict(X), guess, features
    
           
# #     def predict(self, questions, online=False):
# #         """
# #         Predict from a large set of questions whether you should buzz or not.
# #         """
        
# #         assert self._classifier, "Classifier not trained"
# #         assert self._featurizer, "Featurizer not defined"
# #         assert len(self._features) == len(self._questions), "Features not built.  Did you run build_features?"
# #         X = self._featurizer.transform(self._features)

# #         return self._classifier.predict(X), X, self._features, self._correct, self._metadata

# #     def write_json(self, output_filename):
# #         import json
        
# #         vocab = set()
# #         with open(output_filename, 'w') as outfile:
# #             for features, correct, meta in zip(self._features, self._correct, self._metadata):
# #                 assert "label" not in features
# #                 new_features = {}

# #                 new_features['guess:%s' % meta['guess']] = 1                
# #                 for key in features:
# #                     if isinstance(features[key], str):
# #                         new_features["%s:%s" % (key, features[key])] = 1
# #                     else:
# #                         new_features[key] = features[key]
# #                 for key in new_features:
# #                     vocab.add(key)

# #                 new_features['label'] = correct
                    
# #                 outfile.write("%s\n" % json.dumps(new_features))
# #         vocab = list(vocab)
# #         vocab.sort()
# #         return ['BIAS_CONSTANT'] + vocab
    
# #     def load(self):
# #         """
# #         Load the buzzer state from disk
# #         """
        
# #         with open("%s.featurizer.pkl" % self.filename, 'rb') as infile:
# #             self._featurizer = pickle.load(infile)        
    
# #     def save(self):
# #         """
# #         Save the buzzer state to disck
# #         """
        
# #         for gg in self._guessers:
# #             self._guessers[gg].save()
# #         with open("%s.featurizer.pkl" % self.filename, 'wb') as outfile:
# #             pickle.dump(self._featurizer, outfile)  
    
# #     def train(self):
# #         """
# #         Learn classifier parameters from the data loaded into the buzzer.
# #         """

# #         assert len(self._features) == len(self._correct)        
# #         self._featurizer = DictVectorizer(sparse=True)
# #         X = self._featurizer.fit_transform(self._features)
# #         return X

# # class RNNBuzzer(Buzzer):
# #     """
# #     A subclass of Buzzer using an RNN-based approach for decision-making.
# #     """
# #     def __init__(self, filename, run_length, num_guesses=1, n_input=300, n_hidden=128, n_output=2, n_layers=1, dropout=0.1):
# #         super().__init__(filename, run_length, num_guesses)
# #         self.n_input = n_input
# #         self.n_hidden = n_hidden
# #         self.n_output = n_output
# #         self.n_layers = n_layers
# #         self.dropout = dropout

# #         self.rnn = L.NStepLSTM(self.n_layers, self.n_input, self.n_hidden, self.dropout)
# #         self.fc = L.Linear(self.n_hidden, self.n_output)

# #         self.model_name = "RNNBuzzer"
# #         self.model_dir = "output/buzzer/RNNBuzzer"

# #         logging.info("Initialized RNNBuzzer with RNN-based architecture.")

# #     def forward_rnn(self, xs):
# #         xs = [self._featurize_sequence(x) for x in xs]
# #         _, _, hs = self.rnn(None, None, xs)
# #         hs = [F.dropout(h, self.dropout) for h in hs]
# #         outputs = [self.fc(h) for h in hs]
# #         return outputs

# #     def predict_rnn(self, xs, softmax=False, argmax=False):
# #         outputs = self.forward_rnn(xs)
# #         if softmax:
# #             predictions = [F.softmax(o, axis=1).data for o in outputs]
# #         elif argmax:
# #             predictions = [np.argmax(o.data, axis=1) for o in outputs]
# #         else:
# #             predictions = outputs
# #         return predictions

# #     def _featurize_sequence(self, sequence):
# #         return np.random.rand(len(sequence.split()), self.n_input).astype(np.float32)

# #     def __call__(self, xs, ys):
# #         predictions = self.forward_rnn(xs)
# #         concat_predictions = F.concat(predictions, axis=0)
# #         concat_predictions = F.softmax(concat_predictions, axis=1)
# #         concat_truths = F.concat(ys, axis=0)
# #         loss = F.softmax_cross_entropy(concat_predictions, concat_truths)
# #         accuracy = F.accuracy(concat_predictions, concat_truths)
# #         reporter.report({"loss": loss.data}, self)
# #         reporter.report({"accuracy": accuracy.data}, self)
# #         return loss

# #     def train_rnn(self, xs, ys, optimizer):
# #         optimizer.update(self.__call__, xs, ys)

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--RNNBuzzer_filename", type=str, help="File to save the trained RNNBuzzer model")
# #     parser.add_argument("--buzzer_model", type=str, help="Directory to save RNNBuzzer model")

# #     add_general_params(parser)
# #     add_guesser_params(parser)
# #     add_buzzer_params(parser)
# #     add_question_params(parser)
   

# #     flags = parser.parse_args()
# #     setup_logging(flags)

# #     guesser = load_guesser(flags)
# #     buzzer = RNNBuzzer(flags.buzzer_model, flags.run_length, flags.num_guesses)
# #     buzzer.add_guesser("Gpr", guesser, primary_guesser=True)
# #     questions = load_questions(flags)

# #     buzzer.add_data(questions)
# #     buzzer.build_features(flags.buzzer_history_length, flags.buzzer_history_depth)

# #     buzzer.train_rnn()
# #     buzzer.save()

# #     if flags.limit == -1:
# #         print("Ran on %i questions" % len(questions))
# #     else:
# #         print("Ran on %i questions of %i" % (flags.limit, len(questions)))