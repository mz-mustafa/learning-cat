import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error


class BaseRegressor:
    def __init__(self, model):
        self.model = model
        self.metrics = Metrics()  # Composition: Metrics class instance
        self.plots = Plots()  # Composition: Enhanced Plots class instance
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.scaler = StandardScaler()

    def split_data(self, X, y, test_size=0.2, random_state=None):
        """Split the data into training and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

    def scale_features(self):
        """Scale features using StandardScaler."""
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):
        """Train the model using the training data."""
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """Predict values using the trained model for both training and test datasets."""
        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)

    def perform_random_search(self, X, y, n_iter=500, cv=5):
        """Perform random search using the hyperparameter space of the model."""

        # Check if hyperparameter_space is defined for the model
        if not hasattr(self, 'hyperparameter_space'):
            raise NotImplementedError("The model class must define a 'hyperparameter_space' attribute.")

        # Set up random search with cross-validation
        random_search = RandomizedSearchCV(self.model, param_distributions=self.hyperparameter_space,
                                           n_iter=n_iter, cv=cv, verbose=1, scoring='r2', n_jobs=-1)

        # Fit random search
        random_search.fit(X, y)

        # Update the model with the best estimator found during the random search
        self.model = random_search.best_estimator_

        return random_search.best_params_, random_search.best_score_

    def save_model(self, filepath):
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        self.model = joblib.load(filepath)

    # Metrics related methods
    def calculate_metrics(self, y_true, y_pred):
        return self.metrics.calculate(y_true, y_pred)

    def print_metrics(self):
        return self.metrics.print(self.y_train, self.y_pred_train, self.y_test, self.y_pred_test)

    # Plotting related methods
    def plot_feature_importance(self, columns):
        return self.plots.feature_importance(self.model, columns)

    def plot_predicted_vs_actual(self):
        return self.plots.plot_predicted_actual(self.y_train, self.y_pred_train, self.y_test, self.y_pred_test)


class Plots:
    def feature_importance(self, model, columns):
        """Plot feature importances for a given model."""
        try:
            feature_imp = model.feature_importances_
            sorted_indices = feature_imp.argsort()
            features = [columns[i] for i in sorted_indices]
            importances = [feature_imp[i] for i in sorted_indices]

            plt.figure(figsize=(10, len(features) // 2))
            plt.barh(features, importances, align='center')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importances')
            plt.show()
        except AttributeError:
            print("The model does not have a 'feature_importances_' attribute.")

    def plot_predicted_actual(self, y_train, y_pred_train, y_test, y_pred_test):
        """Plot actual vs. predicted values for train and test datasets."""
        plt.figure(figsize=(10, 5))

        # Plot for training data
        plt.subplot(1, 2, 1)
        plt.scatter(y_train, y_pred_train, alpha=0.2)
        plt.plot(range(int(min(y_train)), int(max(y_train) + 1)),
                 range(int(min(y_train)), int(max(y_train) + 1)), 'r')
        plt.title('Train data')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')

        # Plot for test data
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred_test, alpha=0.2)
        plt.plot(range(int(min(y_test)), int(max(y_test) + 1)),
                 range(int(min(y_test)), int(max(y_test) + 1)), 'r')
        plt.title('Test data')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')

        plt.tight_layout()
        plt.show()


class Metrics:
    def calculate(self, y_true, y_pred):
        """Calculate regression metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    def print(self, y_true_train, y_pred_train, y_true_test, y_pred_test):
        """Print regression metrics for train and test datasets."""
        mae_train, rmse_train, r2_train = self.calculate(y_true_train, y_pred_train)
        mae_test, rmse_test, r2_test = self.calculate(y_true_test, y_pred_test)

        print(f'Training MAE: {mae_train}, Test MAE: {mae_test}')
        print(f'Training RMSE: {rmse_train}, Test RMSE: {rmse_test}')
        print('Training R-squared:', r2_train)
        print('Test R-squared:', r2_test)

