import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import feature_eng as fe
import os

# Finds the best model and threshold based on average recall and precision on the training data. Apply the best model
# and threshold to the validation data and calculates evaluation metrics. Saves the best model with the specific
# threshold.

models_dir = 'models/'
validation_metrics_file = 'validation_metrics.txt'

best_model = None
best_threshold = None
best_metric_score = 0.0
best_metric_scores = {}

# Iterate over the pkl files in the models directory
for file_name in os.listdir(models_dir):
    if file_name.endswith('.pkl'):
        model = joblib.load(os.path.join(models_dir, file_name))
        predicted_probabilities_train = model.predict_proba(fe.X_train)[:, 1]
        thresholds = np.arange(0.1, 1.0, 0.1)
        metric_scores = {}

        # Iterates over the threshold values and evaluates the metrics
        for threshold in thresholds:
            adjusted_labels_train = np.where(predicted_probabilities_train >= threshold, 1, 0)
            precision_0 = precision_score(fe.y_train, adjusted_labels_train, pos_label=0)
            recall_0 = recall_score(fe.y_train, adjusted_labels_train, pos_label=0)
            f1_0 = f1_score(fe.y_train, adjusted_labels_train, pos_label=0)
            accuracy_0 = accuracy_score(fe.y_train, adjusted_labels_train)

            precision_1 = precision_score(fe.y_train, adjusted_labels_train, pos_label=1)
            recall_1 = recall_score(fe.y_train, adjusted_labels_train, pos_label=1)
            f1_1 = f1_score(fe.y_train, adjusted_labels_train, pos_label=1)
            accuracy_1 = accuracy_score(fe.y_train, adjusted_labels_train)

            # Calculates the average recall and precision for each class
            avg_recall = (recall_0 + recall_1) / 2
            avg_precision = (precision_0 + precision_1) / 2

            metric_scores[threshold] = {
                'Precision (Class 0)': precision_0,
                'Recall (Class 0)': recall_0,
                'F1-Score (Class 0)': f1_0,
                'Accuracy (Class 0)': accuracy_0,
                'Precision (Class 1)': precision_1,
                'Recall (Class 1)': recall_1,
                'F1-Score (Class 1)': f1_1,
                'Accuracy (Class 1)': accuracy_1,
                'Average Recall': avg_recall,
                'Average Precision': avg_precision
            }

        # Finds the threshold with the best average recall and precision
        best_threshold_for_model = max(metric_scores, key=lambda x: metric_scores[x]['Average Recall'] + metric_scores[x]['Average Precision'])
        best_metric_score_for_model = metric_scores[best_threshold_for_model]['Average Recall'] + metric_scores[best_threshold_for_model]['Average Precision']

        # Updates the best model and threshold if necessary
        if best_metric_score_for_model > best_metric_score:
            best_model = model
            best_threshold = best_threshold_for_model
            best_metric_score = best_metric_score_for_model
            best_metric_scores = metric_scores[best_threshold_for_model]

print("Best Model:", best_model)
print("Best Threshold:", best_threshold)
print("Best Metric Scores:")
for metric_name, metric_score in best_metric_scores.items():
    print(metric_name + ":", metric_score)

best_model_file = 'best_model_with_threshold_{}.pkl'.format(best_threshold)
joblib.dump(best_model, best_model_file)
print("Best model with threshold saved as:", best_model_file)

predicted_probabilities_val = best_model.predict_proba(fe.X_val)[:, 1]


adjusted_labels_val = np.where(predicted_probabilities_val >= best_threshold, 1, 0)

precision_val_0 = precision_score(fe.y_val, adjusted_labels_val, pos_label=0)
recall_val_0 = recall_score(fe.y_val, adjusted_labels_val, pos_label=0)
f1_val_0 = f1_score(fe.y_val, adjusted_labels_val, pos_label=0)
accuracy_val_0 = accuracy_score(fe.y_val, adjusted_labels_val)

precision_val_1 = precision_score(fe.y_val, adjusted_labels_val, pos_label=1)
recall_val_1 = recall_score(fe.y_val, adjusted_labels_val, pos_label=1)
f1_val_1 = f1_score(fe.y_val, adjusted_labels_val, pos_label=1)
accuracy_val_1 = accuracy_score(fe.y_val, adjusted_labels_val)

roc_auc_val = roc_auc_score(fe.y_val, predicted_probabilities_val)

print("Metrics with adjusted threshold on the validation data (Class 0):")
print("Precision:", precision_val_0)
print("Recall:", recall_val_0)
print("F1 Score:", f1_val_0)
print("Accuracy:", accuracy_val_0)

print("Metrics with adjusted threshold on the validation data (Class 1):")
print("Precision:", precision_val_1)
print("Recall:", recall_val_1)
print("F1 Score:", f1_val_1)
print("Accuracy:", accuracy_val_1)

print("ROC AUC Score on the validation data:", roc_auc_val)


with open(validation_metrics_file, 'w') as file:
    file.write("Best Model: {}\n".format(best_model))
    file.write("Best Threshold: {}\n".format(best_threshold))
    file.write("Best Metric Scores:\n")
    file.write("Best model with threshold saved as: {}\n".format(best_model_file))
    file.write("\n")
    file.write("Metrics with adjusted threshold on the validation data (Class 0):\n")
    file.write("Precision: {}\n".format(precision_val_0))
    file.write("Recall: {}\n".format(recall_val_0))
    file.write("F1 Score: {}\n".format(f1_val_0))
    file.write("Accuracy: {}\n".format(accuracy_val_0))
    file.write("\n")
    file.write("Metrics with adjusted threshold on the validation data (Class 1):\n")
    file.write("Precision: {}\n".format(precision_val_1))
    file.write("Recall: {}\n".format(recall_val_1))
    file.write("F1 Score: {}\n".format(f1_val_1))
    file.write("Accuracy: {}\n".format(accuracy_val_1))
    file.write("\n")
    file.write("ROC AUC Score on the validation data: {}".format(roc_auc_val))
