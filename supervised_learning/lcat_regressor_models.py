from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from supervised_learning.lcat_base_regressor import BaseRegressor
import matplotlib.pyplot as plt
from scipy.stats import randint




class RandomForestModel(BaseRegressor):

    hyperparameter_space = {
        'n_estimators': randint(150, 450), #randint(10, 150)
        'max_features': ['sqrt', 'log2'],
        'max_depth': randint(5, 20),#randint(1, 15),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 11),
        'bootstrap': [True, False]
    }
    def __init__(self, n_estimators=100, random_state=None):
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        super().__init__(model)

    def get_feature_importance(self, columns):
        """Get feature importance for the trained RandomForest model."""
        feature_imp = self.model.feature_importances_
        sorted_indices = feature_imp.argsort()
        return [(columns[i], feature_imp[i]) for i in sorted_indices]


# XGBoost Model
class XGBoostModel(BaseRegressor):

    hyperparameter_space = {
        'learning_rate': [0.01, 0.05, 0.1, 0.5],
        'n_estimators': randint(10, 150),
        'max_depth': randint(1, 8),
        'subsample': [0.5, 0.7, 0.9, 1],
        'colsample_bytree': [0.5, 0.7, 0.9, 1],
        'gamma': [0, 0.1, 0.2, 0.5],
        'lambda': [0.1, 1, 10],
        'alpha': [0.1, 1, 10]
    }
    def __init__(self, n_estimators=100, random_state=None):
        model = XGBRegressor(n_estimators=n_estimators, random_state=random_state)
        super().__init__(model)

    def save_model(self, filepath):
        self.model.save_model(filepath)

    def load_model(self, filepath):
        self.model.load_model(filepath)

    def get_feature_importance(self, columns):
        """Get feature importance for the trained XGBoost model."""
        feature_imp = self.model.feature_importances_
        sorted_indices = feature_imp.argsort()
        return [(columns[i], feature_imp[i]) for i in sorted_indices]

class MLPModel(BaseRegressor):

    hyperparameter_space = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'adam'],
        'alpha': [0.0001, 0.001, 0.01, 0.1]
    }

    def __init__(self, hidden_layer_sizes=(100,), random_state=None):
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
        super().__init__(model)
        self.training_loss_ = []

    def train_model(self):
        """Train the model and capture training loss."""
        super().train_model()

        # Assuming the loss curve is available after training
        # (This is true for MLPRegressor from sklearn)
        self.training_loss_ = self.model.loss_curve_

    def model_summary(self):
        """Provide a summary of the neural network architecture."""
        summary = {
            'Number of Layers': len(self.model.hidden_layer_sizes),
            'Neurons in each Layer': self.model.hidden_layer_sizes,
            'Activation Function': self.model.activation,
            'Solver': self.model.solver
        }
        return summary

    def plot_training_loss(self):
        """Plot the training loss over epochs."""
        plt.plot(self.training_loss_)
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


