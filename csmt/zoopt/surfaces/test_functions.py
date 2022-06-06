import numpy as np

from .base_objective_function import ObjectiveFunction


class SphereFunction(ObjectiveFunction):
    name = "Sphere Function"
    _name_ = "sphere_function"
    __name__ = "SphereFunction"

    def __init__(self, n_dim, A=1, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = n_dim
        self.A = A

    def objective_function_dict(self, params):
        loss = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss += self.A * x * x

        return self.return_metric(loss)


class AckleyFunction(ObjectiveFunction):
    name = "Ackley Function"
    _name_ = "ackley_function"
    __name__ = "AckleyFunction"

    def __init__(
        self, A=20, angle=2 * np.pi, metric="score", input_type="dictionary", sleep=0
    ):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = -self.A * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
        loss2 = -np.exp(0.5 * (np.cos(self.angle * x) + np.cos(self.angle * y)))
        loss3 = np.exp(1)
        loss4 = self.A

        loss = loss1 + loss2 + loss3 + loss4

        return self.return_metric(loss)


class RastriginFunction(ObjectiveFunction):
    name = "Rastrigin Function"
    _name_ = "rastrigin_function"
    __name__ = "RastriginFunction"

    def __init__(
        self,
        n_dim,
        A=10,
        angle=2 * np.pi,
        metric="score",
        input_type="dictionary",
        sleep=0,
    ):
        super().__init__(metric, input_type, sleep)

        self.n_dim = n_dim
        self.A = A
        self.angle = angle

    def objective_function_dict(self, params):
        loss = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss += x * x - self.A * np.cos(self.angle * x)

        loss = self.A * self.n_dim + loss

        return self.return_metric(loss)


class RosenbrockFunction(ObjectiveFunction):
    name = "Rosenbrock Function"
    _name_ = "rosenbrock_function"
    __name__ = "RosenbrockFunction"

    def __init__(self, A=1, B=100, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = (self.A - x) ** 2 + self.B * (y - x ** 2) ** 2

        return self.return_metric(loss)


class BealeFunction(ObjectiveFunction):
    name = "Beale Function"
    _name_ = "beale_function"
    __name__ = "BealeFunction"

    def __init__(
        self, A=1.5, B=2.25, C=2.652, metric="score", input_type="dictionary", sleep=0
    ):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B
        self.C = C

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = (self.A - x + x * y) ** 2
        loss2 = (self.B - x + x * y ** 2) ** 2
        loss3 = (self.C - x + x * y ** 3) ** 2

        loss = loss1 + loss2 + loss3

        return self.return_metric(loss)


class HimmelblausFunction(ObjectiveFunction):
    name = "Himmelblau's Function"
    _name_ = "himmelblaus_function"
    __name__ = "HimmelblausFunction"

    def __init__(self, A=-11, B=-7, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = (x ** 2 + y + self.A) ** 2
        loss2 = (x + y ** 2 + self.B) ** 2

        loss = loss1 + loss2

        return self.return_metric(loss)


class HölderTableFunction(ObjectiveFunction):
    name = "Hölder Table Function"
    _name_ = "hölder_table_function"
    __name__ = "HölderTableFunction"

    def __init__(self, A=10, angle=1, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = np.sin(self.angle * x) * np.cos(self.angle * y)
        loss2 = np.exp(abs(1 - (np.sqrt(x ** 2 + y ** 2) / np.pi)))

        loss = -np.abs(loss1 * loss2)

        return self.return_metric(loss)


class CrossInTrayFunction(ObjectiveFunction):
    name = "Cross In Tray Function"
    _name_ = "cross_in_tray_function"
    __name__ = "CrossInTrayFunction"

    def __init__(
        self,
        A=-0.0001,
        B=100,
        angle=1,
        metric="score",
        input_type="dictionary",
        sleep=0,
    ):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = np.sin(self.angle * x) * np.sin(self.angle * y)
        loss2 = np.exp(abs(self.B - (np.sqrt(x ** 2 + y ** 2) / np.pi)) + 1)

        loss = -self.A * (np.abs(loss1 * loss2)) ** 0.1

        return self.return_metric(loss)


class SimionescuFunction(ObjectiveFunction):
    name = "Simionescu Function"
    _name_ = "simionescu_function"
    __name__ = "SimionescuFunction"

    def __init__(
        self,
        A=0.1,
        r_T=1,
        r_S=0.2,
        n=8,
        metric="score",
        input_type="dictionary",
        sleep=0,
    ):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.r_T = r_T
        self.r_S = r_S
        self.n = n

    def objective_function_dict(self, params):
        x = params["x0"].reshape(-1)
        y = params["x1"].reshape(-1)

        condition = (self.r_T + self.r_S * np.cos(self.n * np.arctan(x / y))) ** 2

        mask = x ** 2 + y ** 2 <= condition
        mask_int = mask.astype(int)

        loss = self.A * x * y
        loss = mask_int * loss
        loss[~mask] = np.nan

        return self.return_metric(loss)


class EasomFunction(ObjectiveFunction):
    name = "Easom Function"
    _name_ = "easom_function"
    __name__ = "EasomFunction"

    def __init__(
        self, A=-1, B=1, angle=1, metric="score", input_type="dictionary", sleep=0
    ):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

        self.A = A
        self.B = B
        self.angle = angle

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = self.A * np.cos(x * self.angle) * np.cos(y * self.angle)
        loss2 = np.exp(-((x - np.pi / self.B) ** 2 + (y - np.pi / self.B) ** 2))

        loss = loss1 * loss2

        return self.return_metric(loss)


class BoothFunction(ObjectiveFunction):
    name = "Booth Function"
    _name_ = "booth_function"
    __name__ = "BoothFunction"

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = (x + 2 * y - 7) ** 2
        loss2 = (2 * x + y - 5) ** 2

        loss = loss1 * loss2

        return self.return_metric(loss)


class GoldsteinPriceFunction(ObjectiveFunction):
    name = "Goldstein Price Function"
    _name_ = "goldstein_price_function"
    __name__ = "GoldsteinPriceFunction"

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss1 = 1 + (x + y + 1) ** 2 * (
            19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2
        )
        loss2 = 30 + (2 * x - 3 * y) ** 2 * (
            18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2
        )

        loss = loss1 * loss2

        return self.return_metric(loss)


class StyblinskiTangFunction(ObjectiveFunction):
    name = "Styblinski Tang Function"
    _name_ = "styblinski_tang_function"
    __name__ = "StyblinskiTangFunction"

    def __init__(self, n_dim, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = n_dim

    def objective_function_dict(self, params):
        loss = 0
        for dim in range(self.n_dim):
            dim_str = "x" + str(dim)
            x = params[dim_str]

            loss += x ** 4 - 16 * x ** 2 + 5 * x

        loss = loss / 2

        return self.return_metric(loss)


class MatyasFunction(ObjectiveFunction):
    name = "Matyas Function"
    _name_ = "matyas_function"
    __name__ = "MatyasFunction"

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

        return self.return_metric(loss)


class McCormickFunction(ObjectiveFunction):
    name = "Mc Cormick Function"
    _name_ = "mccormick_function"
    __name__ = "McCormickFunction"

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)
        self.n_dim = 2

    def objective_function_dict(self, params):
        x = params["x0"]
        y = params["x1"]

        loss = np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1

        return self.return_metric(loss)