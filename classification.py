import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)
from keras.models import Model
from keras.layers import Input, Dense

# Placeholder for ASMO optimizer
class ASMO:
    def __init__(self, population_size, dimensions, bounds):
        self.population_size = population_size
        self.dimensions = dimensions
        self.bounds = bounds
    
    def optimize(self, objective_func, iterations, **kwargs):
        best_score = -1
        best_params = None
        for _ in range(iterations):
            params = np.random.rand(self.dimensions) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
            score = objective_func(params, **kwargs)
            if score > best_score:
                best_score = score
                best_params = params
        return best_params

# Objective Function for ASMO Optimization
def objective_func(params, X_train, y_train):
    weight, gaussian_func, tree_count = params
    gaussian_func = max(gaussian_func, 1e-10)
    tree_count = max(int(tree_count), 1)

    nb_classifier = GaussianNB(var_smoothing=gaussian_func)
    rf_classifier = RandomForestClassifier(n_estimators=tree_count)

    nb_classifier.fit(X_train, y_train)
    rf_classifier.fit(X_train, y_train)

    nb_pred = nb_classifier.predict(X_train)
    rf_pred = rf_classifier.predict(X_train)

    ensemble_pred = (weight * nb_pred + (1 - weight) * rf_pred) > 0.5
    return accuracy_score(y_train, ensemble_pred)

# ELC-Casnet-CNN Class
class ELC_Casnet_CNN:
    def __init__(self, input_dim):
        self.autoencoder, self.encoder = self.build_autoencoder(input_dim=input_dim)
        self.nb_classifier = GaussianNB()
        self.rf_classifier = RandomForestClassifier(n_estimators=50)
        self.asmo_optimizer = ASMO(population_size=10, dimensions=3, bounds=(0, 20))

    def build_autoencoder(self, input_dim):
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(128, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = Model(input_layer, decoded)
        encoder = Model(input_layer, encoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')
        
        return autoencoder, encoder

    def extract_features(self, X):
        return self.encoder.predict(X)

    def train(self, X_train, y_train, objective_func):
        self.autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, verbose=0)
        X_train_encoded = self.extract_features(X_train)

        opt_params = self.asmo_optimizer.optimize(objective_func, iterations=50, X_train=X_train_encoded, y_train=y_train)
        weight, gaussian_func, tree_count = opt_params
        
        self.nb_classifier.set_params(var_smoothing=max(gaussian_func, 1e-10))
        self.rf_classifier.set_params(n_estimators=int(tree_count))
        
        self.nb_classifier.fit(X_train_encoded, y_train)
        self.rf_classifier.fit(X_train_encoded, y_train)

    def predict(self, X_test):
        X_test_encoded = self.extract_features(X_test)
        nb_pred = self.nb_classifier.predict(X_test_encoded)
        rf_pred = self.rf_classifier.predict(X_test_encoded)
        
        ensemble_pred = (0.5 * nb_pred + 0.5 * rf_pred) > 0.5
        return ensemble_pred

# Evaluate model performance
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fmeasure = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F-measure: {fmeasure:.2f}')
    print(f'NPV: {npv:.2f}')
    print(f'FPR: {fpr:.2f}')
    print(f'MCC: {mcc:.2f}')
    print(f'FNR: {fnr:.2f}')