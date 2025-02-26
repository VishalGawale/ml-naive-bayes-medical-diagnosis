import pandas as pd
import math

class NaiveBayes:
    def __init__(self):
        self.class_probabilities = {}
        self.feature_probabilities = {}

    def fit(self, X, y):
        # X is the feature matrix and y is the target vector
        
        # Step 1: Calculate the probability of each class (P(class))
        n_samples, n_features = X.shape
        # Count the occurrences of each class in the target vector and divide by the total number of samples
        self.class_probabilities = {label: (count / n_samples) for label, count in y.value_counts().items()}
        
        # Step 2: Calculate the conditional probabilities of each feature given a class (P(feature | class))
        for feature in X.columns:
            # Initialize a dictionary for the current feature
            self.feature_probabilities[feature] = {}
            for value in X[feature].unique():
                # Calculate the probability of each feature value given each class
                self.feature_probabilities[feature][value] = {
                    label: (X[(X[feature] == value) & (y == label)].shape[0] / (y == label).sum())
                    for label in y.unique()
                }

    def predict_probability(self, X):
        # Predict the probability of each class for the given samples
        # X is the feature matrix for which we want to predict the class probabilities
        
        # Step 3: For each sample, calculate the probability of each class given the features (P(class | features)) using Bayes' theorem
        probabilities = []
        for _, row in X.iterrows():
            # Initialize a dictionary to store the probabilities for the current sample
            class_prob = {}
            for label in self.class_probabilities:
                # Start with the prior probability of the class
                class_prob[label] = self.class_probabilities[label]  # P(class)
                for feature in X.columns:
                    # Multiply by the conditional probability of the feature given the class
                    value = row[feature]
                    if value in self.feature_probabilities[feature]:
                        class_prob[label] *= self.feature_probabilities[feature][value].get(label, 1e-5)
                    else:
                        # Apply smoothing for unseen feature values
                        class_prob[label] *= 1e-5
            probabilities.append(class_prob)
        return probabilities

    def predict(self, X):
        # Predict the class with the highest probability for each sample
        predictions = []
        for prob_dict in self.predict_probability(X):
            # Choose the class with the highest probability
            predictions.append(max(prob_dict, key=prob_dict.get))
        return predictions

    def evaluate_on_data(self, X, y):
        # Evaluate the model by calculating the accuracy on the given data
        predictions = self.predict(X)
        # Count the number of correct predictions
        correct = sum(pred == true for pred, true in zip(predictions, y))
        # Return the accuracy as the ratio of correct predictions to the total number of samples
        return correct / len(y)