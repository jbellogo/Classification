from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier
# import xgboost as xgb

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)


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

    rf = RandomForestClassifier(**grid_search.best_params_, random_state=42)
    return rf


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

def log_reg(X, y):
    # Set up parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000]  # Increased max_iter to ensure convergence
    }

    # Initialize the model with multi_class parameter
    lr = LogisticRegression(random_state=42, multi_class='ovr')

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        lr,
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

    # Return the best model
    lr = LogisticRegression(**grid_search.best_params_, random_state=42, multi_class='ovr')
    return lr

def neural_network(X, y):
    # Set up parameter grid
    param_grid = {
        'hidden_layer_sizes': [(100,), (200,), (100, 100)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'max_iter': [1000]
    }

    # Initialize the model
    nn = MLPClassifier(random_state=42)

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        nn,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )   

    # Fit GridSearchCV to the training data
    grid_search.fit(X, y)

    # Get the best parameters and score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")
    return MLPClassifier(**grid_search.best_params_, random_state=42)

# def xgb_classifier(X, y):
#     # Set up parameter grid
#     param_grid = {
#         'max_depth': [3, 5, 7],
#         'learning_rate': [0.01, 0.1, 0.3],
#         'n_estimators': [100, 200],
#         'min_child_weight': [1, 3, 5],
#         'subsample': [0.8, 0.9, 1.0],
#         'colsample_bytree': [0.8, 0.9, 1.0],
#         'gamma': [0, 0.1, 0.2]
#     }

#     # Initialize the model
#     xgb_model = xgb.XGBClassifier(random_state=42)

#     # Set up GridSearchCV
#     grid_search = GridSearchCV(
#         xgb_model,
#         param_grid,
#         cv=5,
#         scoring='accuracy',
#         n_jobs=-1  # Use all available cores
#     )

#     # Fit GridSearchCV to the training data
#     grid_search.fit(X, y)

#     # Get the best parameters and score
#     print(f"Best parameters: {grid_search.best_params_}")
#     print(f"Best cross-validation accuracy: {grid_search.best_score_:.2f}")

#     # Return the best model
#     return xgb.XGBClassifier(**grid_search.best_params_, random_state=42)

def knn_classifier(X, y):
    # Set up parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    }

    # Initialize the model
    knn = KNeighborsClassifier()

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        knn,
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

    # Return the best model
    return KNeighborsClassifier(**grid_search.best_params_)

##### for plotting trees from random forest
# fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=900)
# index = 0 
# tree.plot_tree(rf.estimators_[index],
#                feature_names = fn, 
#                class_names=cn,
#                filled = True)
# fig.savefig('rf_tree.png')



def stacking(X, y, estimators):
    '''
    estimators = [
        ('rf', rf),
        ('hgb', hgb)
    ]
    '''
    # Create the stacked model

    # Using logistic regression as the final estimator
    stacked_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5  # Number of folds for cross-validation during stacking
    )

    # Fit the stacked model
    stacked_model.fit(X, y)

    # Evaluate using cross-validation
    cv_scores = cross_val_score(stacked_model, X, y, cv=5, scoring='accuracy')
    print(f"Stacked model cross-validation accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    return stacked_model
