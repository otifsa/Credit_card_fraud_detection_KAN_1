import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, matthews_corrcoef
from mealpy.GWO import GWO_WOA 


# Prepare data
df=pd.read_csv("creditcard.csv")
X = df.drop(columns=['Class']).values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

n_features = X.shape[1]

def fitness_function(solution):
    if np.sum(solution) == 0:
        return 0.0
    selected_idx = [i for i, bit in enumerate(solution) if bit == 1]
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train[:, selected_idx], y_train)
    preds = clf.predict(X_test[:, selected_idx])
    f1 = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    return np.sqrt(f1 * mcc)

# Define the problem
problem_dict = {
    "fit": fitness_function,
    "lb": [0] * n_features,
    "ub": [1] * n_features,
    "minmax": "max",
    "dtype": "int"
}

# Initialize and run the GWO_WOA optimizer
gwo_woa = GWO_WOA(problem_dict, epoch=100, pop_size=50)
best_position, best_fitness = gwo_woa.solve()

# Retrieve selected features
selected_features = [df.columns[i] for i, bit in enumerate(best_position) if bit == 1]

print("Best Fitness:", best_fitness)
print("Selected Features:", selected_features)
