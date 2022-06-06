from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier_
from sklearn.model_selection import cross_val_score


class KNeighborsClassifier:
    __name__ = "k_neighbors_classifier"

    def __init__(self, X, y, cv=3, metric="score", input_type="dictionary"):
        super().__init__(metric, input_type)

        self.X = X
        self.y = y
        self.cv = cv

        self.model = KNeighborsClassifier_

        self.search_space = {
            "n_neighbors": list(range(3, 50)),
            "leaf_size": list(range(3, 50)),
        }

    def objective_function_dict(self, params):
        knc = KNeighborsClassifier_(
            n_neighbors=params["n_neighbors"],
            leaf_size=params["leaf_size"],
        )
        print("\n self.X \n ", self.X.shape)
        print("\n self.y \n ", self.y.shape)

        print("\n n_neighbors \n ", params["n_neighbors"])
        print("\n leaf_size \n ", params["leaf_size"])

        scores = cross_val_score(knc, self.X, self.y, cv=self.cv)
        print("\n ----------- cross_val_score finished \n ")

        return scores.mean()