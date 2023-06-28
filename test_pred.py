import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import feature_eng as fe
import pandas as pd

# Load the best model and make predictions on the validation data.
model = joblib.load('best_model/best_model_with_threshold_0.6.pkl')
predicted_labels_val = model.predict(fe.X_val)
confusion_mat = confusion_matrix(fe.y_val, predicted_labels_val)
class_labels = ['Class 0', 'Class 1']

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_plot_ada_0,1.png')
plt.show()


X_test_without_customer = fe.df_pca_test.drop(columns=['mk_CurrentCustomer'])
predictions = model.predict(X_test_without_customer)

# Creates a new DataFrame with 'mk_CurrentCustomer' and 'Predictions' columns
result_df = pd.DataFrame()
result_df['mk_CurrentCustomer'] = fe.df_pca_test['mk_CurrentCustomer']
result_df['Predictions'] = predictions

result_df.to_csv('predictions.csv', index=False)
