import imblearn.pipeline as imbpipeline
import pandas as pd
import algorithms as ml_algos
import feature_eng as fe
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib
import imblearn.pipeline as imbpipeline


results_df = pd.DataFrame(
    columns=['Algorithm', 'Parameters', 'Best Score', 'Best Model', 'Accuracy', 'Precision', 'Recall',
             'F1-Score', 'AUC-ROC', 'AUC-PR', 'Classification Report'])

for algo in ml_algos.algorithms:
    pipeline = imbpipeline.Pipeline([
        ('scaler', StandardScaler()),
        ('selector', algo['selector']),
        ('model', algo['model'])
    ])

    grid_search = GridSearchCV(pipeline, param_grid=algo['param_grid'], cv=StratifiedKFold(n_splits=5),
                                    scoring='roc_auc', verbose=3)
    grid_search.fit(fe.X_train, fe.y_train)

    best_model = grid_search.best_estimator_
    model_name = algo['name']
    model_filename = f'modelswpipe/{model_name}_best_model.pkl'
    joblib.dump(best_model, model_filename)

    y_pred = best_model.predict(fe.X_val)
    report = classification_report(fe.y_val, y_pred)
    accuracy = accuracy_score(fe.y_val, y_pred)
    precision = precision_score(fe.y_val, y_pred)
    recall = recall_score(fe.y_val, y_pred)
    f1 = f1_score(fe.y_val, y_pred)
    auc_roc = roc_auc_score(fe.y_val, y_pred)
    auc_pr = average_precision_score(fe.y_val, y_pred)

    report_filename = f'{model_name}_classification_report.txt'
    with open(report_filename, 'w') as file:
        file.write(report)

    results_df = results_df.append({
        'Algorithm': model_name,
        'Parameters': grid_search.best_params_,
        'Best Score': grid_search.best_score_,
        'Best Model': grid_search.best_estimator_,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'Classification Report': report_filename
    }, ignore_index=True)

    print(f"Predictions for {model_name}:")
    print(y_pred)
    print(f"True labels for {model_name}:")
    print(fe.y_val)
    print(f"Classification report for {model_name}_smote:")
    print(report)

if not os.path.exists('modelswpipe/reports'):
    os.makedirs('modelswpipe/reports')

for index, row in results_df.iterrows():
    model_name = row['Algorithm']
    report_filename = f'modelswpipe/reports/{model_name}_classification_report_smote.txt'

    report_text = open(row['Classification Report'], 'r').read()

    with open(report_filename, 'w') as file:
        file.write(report_text)

    results_df.at[index, 'Classification Report'] = report_filename

results_df.to_csv('modelswpipe/model_results.csv', index=False)

