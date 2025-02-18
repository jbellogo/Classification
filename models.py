

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression



def random_forest(X, y):
    # Set up a grid of hyperparameters to search over
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    # Initialize the Random Forest model
    rf = RandomForestClassifier(random_state=42)

    # Set up GridSearchCV to search over the hyperparameter grid
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')

    # Fit GridSearchCV to the training data
    grid_search.fit(X, y)

    # Get the best parameters and the best score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")




def hist_grad_boost(X, y):
    # Set up parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'max_iter': [100, 200],
        'min_samples_leaf': [20, 50],
        'l2_regularization': [0, 1.0, 2.0]
    }

    # Initialize the model
    hgb = HistGradientBoostingClassifier(random_state=42)

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        hgb, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1  # Use all available cores
    )

    # Fit GridSearchCV to the training data
    grid_search.fit(X, y)

    # Get the best parameters and score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")

    hgb = HistGradientBoostingClassifier(**grid_search.best_params_)
    return hgb