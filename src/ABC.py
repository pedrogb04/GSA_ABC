import numpy as np
import time
import os
import random
import pandas as pd


from typing import Any, List, Mapping, Tuple, Union


class ABC:
    def __init__(
        self,
        objective_function: callable,
        r_dim: int,
        d_dim: int,
        boundaries: Mapping[str, List[Union[Any, Tuple[float, float]]]],
        is_feasible: Union[callable, None] = None,
        limit: int = 5,
    ) -> None:

        self.objective_function = objective_function
        self.is_feasible = is_feasible if is_feasible else lambda _: True
        self.r_dim = r_dim
        self.d_dim = d_dim
        self.boundaries = boundaries
        self.real_boundaries = np.array(boundaries["real"])
        self.discrete_boundaries = np.array(boundaries["discrete"])

        # Límite de intentos antes de que una fuente de comida se agote
        self.limit = limit

        self.solution_history = None
        self.accuracy_history = None
        self.convergence = None

    def _get_initial_positions(self, population_size: int) -> Mapping[str, np.ndarray]:

        # Initialize random positions with boundaries for each individual
        pos_r = np.zeros((population_size, self.r_dim))

        for col_index in range(self.r_dim):
            rd_lb, rd_ub = self.real_boundaries[col_index]
            pos_r[:, col_index] = np.random.uniform(
                low=rd_lb, high=rd_ub, size=population_size
            )

        pos_d = np.zeros((population_size, self.d_dim)).astype(int)
        for col_index in range(self.d_dim):
            dd_lb, dd_ub = self.discrete_boundaries[col_index]
            while True:
                pos_d[:, col_index] = np.random.choice(
                    a=range(dd_lb, dd_ub + 1), size=population_size
                )
                if sum(pos_d[:, col_index]) != 0:
                    break

        initial_pop = {"real": pos_r, "discrete": pos_d}
        for sol in range(population_size):
            solution = {"real": pos_r[sol, :], "discrete": pos_d[sol, :]}
            if not self.is_feasible(solution):
                initial_pop = self._get_initial_positions(population_size)
                break

        return initial_pop

    def _perturb_solution(
        self, i: int, k: int, pos: Mapping[str, np.ndarray], repair_solution: bool
    ) -> Mapping[str, np.ndarray]:
        """
        Generate a new solution mutating the current one relative to a neighbour (k).
        """

        new_solution = {
            "real": np.copy(pos["real"][i]),
            "discrete": np.copy(pos["discrete"][i]),
        }

        # Perturbación de variables REALES (Hiperparámetros)
        if self.r_dim > 0:
            phi_r = np.random.uniform(low=-1, high=1, size=self.r_dim)
            new_solution["real"] = pos["real"][i] + phi_r * (
                pos["real"][i] - pos["real"][k]
            )
            # Aplicar límites
            new_solution["real"] = np.clip(
                new_solution["real"],
                self.real_boundaries[:, 0],
                self.real_boundaries[:, 1],
            )

        # Perturbación de variables DISCRETAS (Features)
        if self.d_dim > 0:
            phi_d = np.random.uniform(low=-1, high=1, size=self.d_dim)
            step = phi_d * (pos["discrete"][i] - pos["discrete"][k])

            move_probs = np.abs(np.tanh(step))
            rand = np.random.rand(self.d_dim)

            flip_mask = rand < move_probs
            new_solution["discrete"][flip_mask] = (
                1 - new_solution["discrete"][flip_mask]
            )
            new_solution["discrete"] = new_solution["discrete"].astype(int)

        if not np.any(new_solution["discrete"]):
            new_solution["discrete"][np.random.randint(0, self.d_dim)] = 1

        # Comprobación de viabilidad y reparación
        if not self.is_feasible(new_solution):
            if not repair_solution:

                new_solution["real"] = np.clip(
                    new_solution["real"],
                    self.real_boundaries[:, 0],
                    self.real_boundaries[:, 1],
                )
                new_solution["discrete"] = np.clip(
                    new_solution["discrete"],
                    self.discrete_boundaries[:, 0],
                    self.discrete_boundaries[:, 1],
                ).astype(int)
            else:
                pass

        return new_solution

    def _employed_bees_phase(
        self,
        n_employed: int,
        pos: Mapping[str, np.ndarray],
        fit: np.ndarray,
        accs: np.ndarray,
        trials: np.ndarray,
        repair_solution: bool,
    ) -> None:
        """Phase definition of employed bees"""
        for i in range(n_employed):
            k = i
            while k == i:
                k = np.random.randint(0, n_employed)

            new_sol = self._perturb_solution(i, k, pos, repair_solution)
            new_fit, new_acc = self.objective_function(new_sol)

            if new_fit > fit[i]:
                pos["real"][i] = new_sol["real"]
                pos["discrete"][i] = new_sol["discrete"]
                fit[i] = new_fit
                accs[i] = new_acc
                trials[i] = 0
            else:
                trials[i] += 1

    def _onlooker_bees_phase(
        self,
        n_employed: int,
        pos: Mapping[str, np.ndarray],
        fit: np.ndarray,
        accs: np.ndarray,
        trials: np.ndarray,
        repair_solution: bool,
    ) -> None:
        """Onlooker bees phase"""
        fit_norm = fit - np.min(fit)
        sum_fit = np.sum(fit_norm)
        probs = fit_norm / sum_fit if sum_fit != 0 else np.ones(n_employed) / n_employed

        t = 0
        i = 0
        while t < n_employed:
            if np.random.rand() < probs[i]:
                t += 1
                k = i
                while k == i:
                    k = np.random.randint(0, n_employed)

                new_sol = self._perturb_solution(i, k, pos, repair_solution)
                new_fit, new_acc = self.objective_function(new_sol)

                if new_fit > fit[i]:
                    pos["real"][i] = new_sol["real"]
                    pos["discrete"][i] = new_sol["discrete"]
                    fit[i] = new_fit
                    accs[i] = new_acc
                    trials[i] = 0
                else:
                    trials[i] += 1
            i = (i + 1) % n_employed

    def _scout_bees_phase(
        self,
        n_employed: int,
        pos: Mapping[str, np.ndarray],
        fit: np.ndarray,
        accs: np.ndarray,
        trials: np.ndarray,
    ) -> None:
        """Scout bees phase"""
        for i in range(n_employed):
            if trials[i] > self.limit:
                new_pos = self._get_initial_positions(1)
                pos["real"][i] = new_pos["real"][0]
                pos["discrete"][i] = new_pos["discrete"][0]

                sol = {"real": pos["real"][i], "discrete": pos["discrete"][i]}
                fit[i], accs[i] = self.objective_function(sol)
                trials[i] = 0

    def optimize(
        self, population_size: int, iters: int, repair_solution: bool = False
    ) -> pd.DataFrame:
        n_employed = population_size

        pos = self._get_initial_positions(n_employed)
        fit = np.zeros(n_employed)
        accs = np.zeros(n_employed)
        trials = np.zeros(n_employed)

        g_best = {"real": np.zeros(self.r_dim), "discrete": np.zeros(self.d_dim)}
        g_best_score = float("-inf")
        best_acc = 0.0

        best_solution_history = []
        best_accuracy_history = []
        convergence_curve = np.zeros(iters)

        print('ABC is optimizing  "' + self.objective_function.__name__ + '"')
        timer_start = time.time()

        history = pd.DataFrame(
            columns=[
                "Iteration",
                "Fitness",
                "Accuracy",
                "ExecutionTime",
                "Discrete",
                "Real",
            ]
        )

        # Evaluación Inicial
        for i in range(n_employed):
            sol = {"real": pos["real"][i], "discrete": pos["discrete"][i]}
            fit[i], accs[i] = self.objective_function(sol)
            if fit[i] > g_best_score:
                g_best_score = fit[i]
                g_best = {
                    "real": sol["real"].copy(),
                    "discrete": sol["discrete"].copy(),
                }
                best_acc = accs[i]

        for current_iter in range(iters):

            # Ejecución secuencial de las tres fases del ABC
            self._employed_bees_phase(
                n_employed, pos, fit, accs, trials, repair_solution
            )

            self._onlooker_bees_phase(
                n_employed, pos, fit, accs, trials, repair_solution
            )

            self._scout_bees_phase(n_employed, pos, fit, accs, trials)

            # Actualización del Mejor Global
            for i in range(n_employed):
                if fit[i] > g_best_score:
                    g_best_score = fit[i]
                    g_best = {
                        "real": pos["real"][i].copy(),
                        "discrete": pos["discrete"][i].copy(),
                    }
                    best_acc = accs[i]

            convergence_curve[current_iter] = g_best_score
            best_solution_history.append(g_best)
            best_accuracy_history.append(best_acc)
            history.loc[len(history)] = [
                current_iter,
                g_best_score,
                best_acc,
                time.time() - timer_start,
                g_best["discrete"],
                g_best["real"],
            ]

            print(f"At iteration {current_iter + 1} the best fitness is {g_best_score}")

        self.convergence = convergence_curve
        self.solution_history = best_solution_history
        self.accuracy_history = best_accuracy_history

        return history
