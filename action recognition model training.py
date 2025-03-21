import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample

# -------------------------------------------------------------------
# 1) Define Paths and Labels
# -------------------------------------------------------------------
action_folders = [
    \bottom forehand\action recognition",
    \bottom backhand\action recognition",
    \top forehand\action recognition",
    \top backhand\action recognition"
]

# Convert folder names to action labels
def folder_to_label(folder_path: str) -> str:
    return os.path.basename(os.path.dirname(folder_path)).replace(" ", "_").lower()

# Prepare results folder
base_dataset_dir = os.path.dirname(action_folders[0])
results_dir = os.path.join(base_dataset_dir, "results_new")
os.makedirs(results_dir, exist_ok=True)

# -------------------------------------------------------------------
# 2) Load & Combine All Processed CSVs
# -------------------------------------------------------------------
all_dfs = []
for action_folder in action_folders:
    csv_files = glob.glob(os.path.join(action_folder, "**", "*.csv"), recursive=True)
    action_label = folder_to_label(action_folder)
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:  # Ensure only non-empty DataFrames are used
                df["action_label"] = action_label
                all_dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

if not all_dfs:
    raise ValueError("No valid CSV files found. Check dataset paths.")

final_dataset = pd.concat(all_dfs, ignore_index=True)

# -------------------------------------------------------------------
# 3) Select Features for Training
# -------------------------------------------------------------------
selected_features = [
    "top_player_to_ball_distance", "bottom_player_to_ball_distance", "ball_to_net_distance", "ball_center_x", "ball_center_y",
    "top_player_x1", "top_player_x2", "top_player_y1", "top_player_y2",
    "bottom_player_x1", "bottom_player_x2", "bottom_player_y1", "bottom_player_y2",
    "top_left_wrist_dist", "top_right_wrist_dist", "bottom_left_wrist_dist", "bottom_right_wrist_dist",
    "net_x1", "net_x2", "net_y1", "net_y2", "players_feet_distance"
]

# Add court keypoints and player keypoints
for i in range(18):  # Court keypoints
    selected_features.append(f"court_kp_{i}_x")
    selected_features.append(f"court_kp_{i}_y")

for i in range(17):  # Player pose keypoints
    selected_features.append(f"top_player_kp{i}_x")
    selected_features.append(f"top_player_kp{i}_y")
    selected_features.append(f"top_player_kp{i}_conf")
    selected_features.append(f"bottom_player_kp{i}_x")
    selected_features.append(f"bottom_player_kp{i}_y")
    selected_features.append(f"bottom_player_kp{i}_conf")

# Ensure only existing features are used
selected_features = [feat for feat in selected_features if feat in final_dataset.columns]

# Extract data and labels
X = final_dataset[selected_features].fillna(0)
y = final_dataset["action_label"].astype("category")

# -------------------------------------------------------------------
# 4) Balance the Dataset
# -------------------------------------------------------------------
min_class_size = y.value_counts().min()
balanced_dfs = [
    resample(final_dataset[final_dataset["action_label"] == label], 
             replace=False, 
             n_samples=min_class_size, 
             random_state=42)
    for label in y.unique()
]

balanced_df = pd.concat(balanced_dfs)
X_balanced = balanced_df[selected_features].fillna(0)
y_balanced = balanced_df["action_label"].astype("category")

# -------------------------------------------------------------------
# 5) Train-Test Split
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# -------------------------------------------------------------------
# 6) Model Training and Saving
# -------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf_pipeline = RandomForestClassifier(n_estimators=100, random_state=42)
clf_pipeline.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(clf_pipeline, os.path.join(results_dir, "trained_action_model.pkl"))
joblib.dump(scaler, os.path.join(results_dir, "scaler.pkl"))

# -------------------------------------------------------------------
# 7) Evaluation and Results
# -------------------------------------------------------------------
y_pred = clf_pipeline.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Save results
report_path = os.path.join(results_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=y.cat.categories, yticklabels=y.cat.categories)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
conf_mat_path = os.path.join(results_dir, "confusion_matrix.png")
plt.savefig(conf_mat_path, dpi=150)
plt.close()

# -------------------------------------------------------------------
# 8) Feature Importance
# -------------------------------------------------------------------
rf_model = clf_pipeline
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

feature_importance_path = os.path.join(results_dir, "feature_importance.csv")
feature_importance_df.to_csv(feature_importance_path, index=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=feature_importance_df.head(10))
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
feature_importance_plot_path = os.path.join(results_dir, "feature_importance.png")
plt.savefig(feature_importance_plot_path, dpi=150)
plt.close()

print(f"Results saved in {results_dir}")
