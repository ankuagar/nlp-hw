# # rnn_buzzer.py
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.feature_extraction import DictVectorizer
from buzzer import Buzzer
import logging 
class RNNModel(nn.Module):
    """
    RNN model with text embeddings and additional features.
    """
    def __init__(self, vocab_size, embedding_dim, feature_dim, hidden_size=128, output_size=2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.feature_fc = nn.Linear(feature_dim, hidden_size)  # Fully connected layer for additional features
        self.combined_fc = nn.Linear(hidden_size * 2, output_size)  # Combine text and features

    def forward(self, text_input, features):
        """
        Forward pass with text embeddings and additional features.
        """
        # Text Embedding and RNN
        embedded = self.embedding(text_input)  # [batch_size, seq_len, embedding_dim]
        _, rnn_hidden = self.rnn(embedded)    # [1, batch_size, hidden_size]

        # Process Features
        feature_out = self.feature_fc(features)  # [batch_size, hidden_size]

        # Combine RNN output and features
        combined = torch.cat((rnn_hidden.squeeze(0), feature_out), dim=1)  # [batch_size, hidden_size * 2]
        output = self.combined_fc(combined)  # [batch_size, output_size]
        return output
class RNNBuzzer(Buzzer):
    """
    RNN-based buzzer integrating text embeddings and additional features.
    """

    def __init__(self, filename, run_length, embedding_dim=50, hidden_size=128, learning_rate=1e-3):
        super().__init__(filename, run_length)
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._featurizer = DictVectorizer(sparse=True)  # Vectorizer for additional features
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

    def _initialize_model(self, vocab_size, feature_dim):
        """
        Initialize the RNN model with text and feature dimensions.
        """
        self.model = RNNModel(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            feature_dim=feature_dim,
            hidden_size=self.hidden_size
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def prepare_embeddings(self):
        """
        Generate embeddings for text in self._runs.
        """
        vocab = {}
        sequences = []
        for run in self._runs:
            tokens = run.split()  # Simple tokenization
            indices = [vocab.setdefault(word, len(vocab)) for word in tokens]
            sequences.append(indices)

        # Pad sequences to the same length
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = [seq + [0] * (max_len - len(seq)) for seq in sequences]

        self.vocab_size = len(vocab)
        return torch.tensor(padded_sequences, dtype=torch.long)

    def prepare_features(self):
        """
        Vectorize self._features using DictVectorizer.
        """
        return self._featurizer.fit_transform(self._features).toarray()

    def train(self):
        """
        Train the RNN buzzer model with text embeddings and additional features.
        """
        # Ensure self._runs, self._features, and self._correct are populated
        if not self._runs or not self._features or not self._correct:
            raise ValueError("No data available. Ensure add_data and build_features are called before training.")

        # Prepare embeddings and feature matrix
        embedded_data = self.prepare_embeddings()
        feature_matrix = self.prepare_features()
        labels_tensor = torch.tensor(self._correct, dtype=torch.long).to(self.device)

        # Initialize model
        self._initialize_model(self.vocab_size, feature_matrix.shape[1])

        # Training loop
        epochs = 10
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()

            predictions = self.model(
                embedded_data.to(self.device),
                torch.tensor(feature_matrix, dtype=torch.float32).to(self.device),
            )
            loss = self.criterion(predictions, labels_tensor)
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def predict(self, questions=None):
        """
        Predict with the RNN model and return outputs in the same format as Buzzer.predict.
        """
        # Ensure self._runs and self._features are populated
        if not self._runs or not self._features or not self._correct or not self._metadata:
            raise ValueError("No data available. Ensure add_data and build_features are called before prediction.")

        # Prepare embeddings and feature matrix
        embedded_data = self.prepare_embeddings()
        feature_matrix = self.prepare_features()

        # Predict using the RNN model
        self.model.eval()
        with torch.no_grad():
            predictions_tensor = self.model(
                embedded_data.to(self.device),
                torch.tensor(feature_matrix, dtype=torch.float32).to(self.device),
            )
            predicted_labels = torch.argmax(predictions_tensor, dim=1).cpu().tolist()

        # Ensure outputs match Buzzer.predict format
        predictions = predicted_labels
        return predictions, feature_matrix, self._features, self._correct, self._metadata

        
    def save(self):
        """
        Save the RNN model, featurizer, and additional metadata.
        """
        Buzzer.save(self)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "vocab_size": self.vocab_size,
            "featurizer": self._featurizer
        }, f"{self.filename}.rnn_model.pth")

    def load(self):
        """
        Load the RNN model, featurizer, and additional metadata.
        """
        Buzzer.load(self)
        checkpoint = torch.load(f"{self.filename}.rnn_model.pth")
        
        # Restore vocab_size and initialize model
        self.vocab_size = checkpoint["vocab_size"]
        feature_dim = len(self._featurizer.feature_names_)
        self._initialize_model(self.vocab_size, feature_dim)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])