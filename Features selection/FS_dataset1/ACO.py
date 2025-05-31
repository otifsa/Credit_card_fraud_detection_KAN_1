import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

class AntColonyOptimizer:
    def __init__(self, X_train, y_train, X_test, y_test,
                 n_ants=50, n_iterations=100, decay=0.1, alpha=1.0, beta=1.0):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_features = X_train.shape[1]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.pheromone = np.ones(self.n_features)
        self.heuristic = np.ones(self.n_features)

    def fitness(self, mask):
        if not mask.any():
            return 0.0
        idx = np.where(mask)[0]
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(self.X_train[:, idx], self.y_train)
        preds = clf.predict(self.X_test[:, idx])
        f1 = f1_score(self.y_test, preds)
        mcc = matthews_corrcoef(self.y_test, preds)
        return np.sqrt(f1 * mcc)

    def construct_solution(self):
        pheromone_term = self.pheromone ** self.alpha
        heuristic_term = self.heuristic ** self.beta
        probs = pheromone_term * heuristic_term
        probs /= probs.sum()
        mask = np.random.rand(self.n_features) < probs
        if not mask.any():
            mask[np.random.randint(0, self.n_features)] = True
        return mask

    def update_pheromones(self, masks, fits):
        self.pheromone *= (1 - self.decay)
        for mask, fit in zip(masks, fits):
            self.pheromone += fit * mask

    def solve(self):
        best_mask = None
        best_fit = -np.inf
        for _ in range(self.n_iterations):
            masks, fits = [], []
            for _ in range(self.n_ants):
                mask = self.construct_solution()
                fit = self.fitness(mask)
                masks.append(mask)
                fits.append(fit)
                if fit > best_fit:
                    best_fit, best_mask = fit, mask.copy()
            self.update_pheromones(masks, fits)
        return best_mask, best_fit


df=pd.read_csv("creditcard.csv")
X = df.drop(columns=['Class']).values
y = df['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
aco = AntColonyOptimizer(X_train, y_train, X_test, y_test,n_ants=50, n_iterations=100,decay=0.1, alpha=1.0, beta=1.0)
best_mask, best_fit = aco.solve()
selected_features = [df.columns[i] for i, flag in enumerate(best_mask) if flag]
print("Best Fitness:", best_fit)
print("Selected Features:", selected_features)
