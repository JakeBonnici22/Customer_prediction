from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier

# Algorithms to test with GridSearchCV and StratifiedKFold cross validation (5 folds) for hyperparameter tuning and
# model selection.
# Some of the algorithms were tested and not used due to time constraints or poor performance, similarly RFECV
# (Computationally intensive) was not used for the final model.

algorithms = [
    {
        'name': 'Voting Classifier',
        'model': VotingClassifier(
            estimators=[
                ('brf', BalancedRandomForestClassifier(class_weight='balanced')),
                ('ada', AdaBoostClassifier(base_estimator=RandomForestClassifier(class_weight='balanced'))),
                ('xgb', XGBClassifier(scale_pos_weight=6)),
            ],
            voting='soft'
        ),
        'selector': SelectKBest(f_classif),
        'param_grid': {
            'selector__k': [4, 6, 8]
        }
    },

    {
        'name': 'Bagging Classifier',
        'model': BaggingClassifier(
            base_estimator=RandomForestClassifier(class_weight='balanced'),
            n_estimators=10,
            random_state=42
        ),
        'selector': SelectKBest(),
        'param_grid': {
            'selector__k': [4, 6, 8]
        }
    },

    {
        'name': 'Balanced Bagging Classifier',
        'model': BalancedBaggingClassifier(
            base_estimator=RandomForestClassifier(class_weight='balanced'),
            n_estimators=10,
            random_state=42
        ),
        'selector': SelectKBest(),
        'param_grid': {
            'selector__k': [4, 6, 8]
        }
    },

    {
        'name': 'Balanced Random Forest',
        'model': BalancedRandomForestClassifier(class_weight='balanced'),
        'selector': SelectKBest(),
        'param_grid': {
            'selector__k': [4, 6, 8],
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 4],
            'model__min_samples_split': [2, 5],
            'model__bootstrap': [True]
        }
    },

    {
        'name': 'ADA',
        'model': AdaBoostClassifier(base_estimator=RandomForestClassifier(class_weight='balanced')),
        'selector': SelectKBest(),
        'param_grid': {
            'selector__k': [4, 6, 8],
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1]
        }
    },

    {
        'name': 'Random Forest',
        'model': RandomForestClassifier(class_weight='balanced'),
        'selector': SelectKBest(),
        # 'selector': RFE(estimator=RandomForestClassifier(class_weight='balanced')),
        # 'selector': RFECV(estimator=RandomForestClassifier(class_weight='balanced')),
        'param_grid': {
            # 'selector__score_func': [f_classif],
            'selector__k': [4, 6, 8],
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 4],
            'model__min_samples_split': [2, 5],
            'model__bootstrap': [True]
        }
    }
    # ,
    #
    # {
    #         'name': 'Logistic Regression',
    #         'model': LogisticRegression(class_weight='balanced'),
    #         # 'selector': RFECV(estimator=LogisticRegression(class_weight='balanced')),
    #         'selector': SelectKBest(),
    #         'param_grid': {
    #             'selector__k': [4, 6, 8],
    #             'model__C': [0.1, 1, 10],
    #             'model__penalty': ['l2'],
    #             'model__solver': ['liblinear', 'saga'],
    #             'model__l1_ratio': [None, 0.5]
    #         }
    #     },
    #
    # {
    #     'name': 'XGBoost',
    #     'model': XGBClassifier(scale_pos_weight=10),
    #     'selector': SelectKBest(),
    #     # 'selector': RFECV(estimator=XGBClassifier(scale_pos_weight=10)),
    #     # 'selector': None,
    #     'param_grid': {
    #         'selector__k': [4, 6, 8],
    #         'model__n_estimators': [100, 200],
    #         'model__max_depth': [3, 4],
    #         'model__learning_rate': [0.1, 0.01]
    #     }
    # },
    #
    # {
    #     'name': 'SVM',
    #     'model': SVC(class_weight='balanced', probability=True),
    #     'selector': SelectKBest(),
    #     'param_grid': {
    #         'selector__k': [4, 6, 8],
    #         'model__C': [0.1, 1],
    #         'model__gamma': [0.1, 1],
    #         'model__kernel': ['rbf']
    #     }
    # },

    # ,

    # {
    #     'name': 'K-Nearest Neighbors',
    #     'model': libs.KNeighborsClassifier(),
    #     'selector': None,  # RFECV selector
    #     'param_grid': {
    #         'model__n_neighbors': [3, 5, 7],
    #         'model__weights': ['uniform', 'distance'],
    #         'model__algorithm': ['auto', 'ball_tree', 'kd_tree'],
    #         'model__leaf_size': [10, 30, 50]
    #     }
    # },
    # {
    #     'name': 'Decision Tree',
    #     'model': libs.DecisionTreeClassifier(class_weight='balanced'),
    #     'selector': libs.RFECV(estimator=libs.DecisionTreeClassifier(class_weight='balanced')),  # RFECV selector
    #     'param_grid': {
    #         'model__criterion': ['gini', 'entropy'],
    #         'model__max_depth': [None, 5, 10, 20],
    #         'model__min_samples_split': [2, 5, 10],
    #         'model__max_features': ['auto', 'sqrt', 'log2']
    #     }
    # }
]

print("Algorithms defined")
