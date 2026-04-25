# Python libraries
import pandas as pd

# Machine learning + eval libraries
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def features_selection(pipeline):

    feature_names = (
        pipeline.named_steps['preprocessing']
        .get_feature_names_out()
    )
    importances = pipeline.named_steps['model'].feature_importances_

    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False)
    print(fi_df.head())

    return fi_df

def build_pipeline(y_train, preprocessing):
    """
    Constructs the classification pipeline.
    """
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('model', XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=11
        ))
    ])     

    return pipeline

def hyper_tuning(pipeline, X_train, y_train):
    """
    Process hyperparameter tuning using RandomizedSearchCV.
    """
    param_grid = {
        'model__n_estimators':     [100, 200, 400],
        'model__max_depth':        [3, 4, 5, 6],
        'model__learning_rate':    [0.01, 0.05, 0.1],
        'model__subsample':        [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0],
        'model__min_child_weight': [1, 3, 5],
        'model__reg_alpha':        [0, 0.1, 1.0],
        'model__reg_lambda':       [1.0, 2.0, 5.0]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=50,
        scoring='roc_auc',
        cv=5,
        random_state=11,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print(f"Cross-validation Best AUC: {search.best_score_:.4f}")
    print(f"Optimal Parameters: {search.best_params_}")

    return search.best_estimator_
