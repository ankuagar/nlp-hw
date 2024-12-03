import pickle
import torch
import torch.nn as nn
from buzzer import Buzzer  # Base class for buzzers

class MLPBuzzer(Buzzer):
    def __init__(self, filename, run_length, num_guesses, hidden_dims, learning_rate=1e-3, device=None):
        """
        Initializes the MLP-based buzzer.
        """
        super().__init__(filename, run_length, num_guesses)
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.loss_function = nn.BCELoss()  # Binary Cross-Entropy Loss

    def _initialize_model(self, input_dim):
        """
        Dynamically initializes the MLP model with custom weight initialization.
        """
        layers = []
        prev_dim = input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))  # Final layer for binary output
        layers.append(nn.Sigmoid())  # Ensure output is in range [0, 1]

        self.model = nn.Sequential(*layers).to(self.device)

        # Apply custom weight and bias initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.uniform_(m.bias, -0.01, 0.01)

        self.model.apply(init_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        """
        Train the MLP model using features and labels.
        """
        X = Buzzer.train(self)  # Get features
        self._initialize_model(input_dim=X.shape[1])

        # Prepare tensors
        features = torch.tensor(X.toarray(), dtype=torch.float32).to(self.device)
        labels = torch.tensor(self._correct, dtype=torch.float32).unsqueeze(1).to(self.device)

        for epoch in range(10):  # Train for 10 epochs
            self.model.train()
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.loss_function(predictions, labels)

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")  # Log loss
            loss.backward()
            self.optimizer.step()

    def predict(self, features=None):
        """
        Predict buzz decisions for a batch of input features.
        """
        # Use self._features if features are not provided
        if features is None:
            features = self._features

        # Vectorize features if they are in dictionary format
        if isinstance(features, list) and isinstance(features[0], dict):
            features = self._featurizer.transform(features)

        # Convert sparse matrix to dense if needed
        if hasattr(features, "toarray"):  # Check if features is a sparse matrix
            features = features.toarray()

        # Convert to PyTorch tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions_tensor = self.model(features_tensor)

        predictions = (predictions_tensor > 0.3).float().cpu().numpy()  # Apply threshold for binary output
        return predictions, features, self._features, self._correct, self._metadata

    def save(self):
        """
        Save the MLP model and parent state.
        """
        Buzzer.save(self)
        with open(f"{self.filename}.model.pkl", "wb") as f:
            pickle.dump({"model_state": self.model.state_dict(), "featurizer": self._featurizer}, f)

    def load(self):
        """
        Load the MLP model and parent state.
        """
        with open(f"{self.filename}.model.pkl", "rb") as f:
            checkpoint = pickle.load(f)
            model_state = checkpoint["model_state"]
            self._featurizer = checkpoint["featurizer"]

        assert self._featurizer is not None, "Featurizer must be loaded before loading the model."

        # Rebuild the model
        input_dim = len(self._featurizer.feature_names_)
        self._initialize_model(input_dim)
        self.model.load_state_dict(model_state)

# This loss function is not used in the final implementation ( GPT cond 0 and always predict 1 )
# class BuzzLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, confidences, accuracies):
#         """
#         Custom loss function for MLP buzzer.
#         """
#         batch_size, T = confidences.size()
#         buzz_probs = torch.zeros_like(confidences)
#         system_scores = torch.zeros(batch_size, device=confidences.device)

#         for t in range(T):
#             if t == 0:
#                 buzz_probs[:, t] = confidences[:, t]
#             else:
#                 cumulative_no_buzz = torch.prod(1 - confidences[:, :t], dim=1)
#                 buzz_probs[:, t] = confidences[:, t] * cumulative_no_buzz

#             system_scores += buzz_probs[:, t] * accuracies[:, t]

#         return -torch.mean(system_scores)
