import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, matthews_corrcoef
from mealpy import FloatVar, PSO

# Prepare data
df=pd.read_parquet("IEEE-fraud-detection.parquet")
X = df.drop(columns=['isFraud']).values
y = df['isFraud'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

n_features = X.shape[1]

def objective_function(solution):
    mask = np.array(solution) > 0.5
    if not mask.any():
        return 0.0
    selected_idx = np.where(mask)[0]
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train[:, selected_idx], y_train)
    preds = clf.predict(X_test[:, selected_idx])
    f1 = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    return np.sqrt(f1 * mcc)

# Define the PSO problem with continuous bounds [0,1]
problem_dict = {
    "bounds": FloatVar(lb=(0.,) * n_features, ub=(1.,) * n_features, name="mask"),
    "obj_func": objective_function,
    "minmax": "max",
}

# Initialize and run the AIW_PSO variant
model = PSO.AIW_PSO(epoch=100, pop_size=50, c1=2.05, c2=20.5, alpha=0.4)
g_best = model.solve(problem_dict)

# Extract the best binary mask and selected feature names
best_solution = np.array(g_best.solution) > 0.5
selected_features = [df.columns[i] for i, flag in enumerate(best_solution) if flag]

print("Best Fitness:", g_best.target.fitness)
print("Selected Features:", selected_features)
