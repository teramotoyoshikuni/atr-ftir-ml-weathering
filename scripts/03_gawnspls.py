# -*- coding: utf-8 -*-
"""
03_gawnspls.py

Demonstration of Genetic Algorithm-based Wavenumber Selection using PLS (GAWNSPLS)

- Minimal modifications from the original student script for GitHub reproducibility.
- Input CSV is unified to: for_regression.csv
- Wavenumbers are taken from the column names of the CSV (recommended for your dataset format).
- Outputs are saved under: results/03_gawnspls/

Required packages:
- numpy, pandas, matplotlib
- scikit-learn
- deap
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression


# -----------------
# Settings
# -----------------
DATA_FILE = Path("for_regression.csv")   # unified input
OUT_DIR = Path("results") / "03_gawnspls"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
RANDOM_SEED = 42

# GA settings
number_of_population = 100
number_of_generation = 150

# PLS / CV settings
max_number_of_components = 1
fold_number = 16

probability_of_crossover = 0.5
probability_of_mutation = 0.2

# Not used in this script, kept for compatibility with the original header
threshold_of_variable_selection = 0.5

# Wavenumber selection settings
max_width_of_areas = 20
number_of_areas_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50]


# -----------------
# Load dataset
# -----------------
# Expect:
# - index column = sample name
# - first column = y
# - remaining columns = X (column names are wavenumbers)
df = pd.read_csv(DATA_FILE, index_col=0)

y_train = df.iloc[:, 0].to_numpy(dtype=float)
x_train = df.iloc[:, 1:].to_numpy(dtype=float)

# wavenumbers from column names
# (your CSV columns should be like "4000", "3996", ... or numeric strings)
wavenumbers = pd.to_numeric(df.columns[1:], errors="coerce").to_numpy()

# autoscaling
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)


# -----------------
# Helper functions
# -----------------
def create_ind_uniform(min_boundary, max_boundary):
    index = []
    for min_val, max_val in zip(min_boundary, max_boundary):
        index.append(random.uniform(min_val, max_val))
    return index


def evalOneMax(individual):
    # NOTE: uses "number_of_areas" from outer loop (same as your original code)
    individual_array = np.array(np.floor(individual), dtype=int)
    selected_x_variable_numbers = np.zeros(0, dtype=int)

    for area_number in range(number_of_areas):
        start = individual_array[2 * area_number]
        width = individual_array[2 * area_number + 1]

        # Clip start/width to safe bounds (minimal safety guard)
        start = max(0, min(start, autoscaled_x_train.shape[1] - 1))
        width = max(1, width)

        if start + width <= autoscaled_x_train.shape[1]:
            selected_x_variable_numbers = np.r_[selected_x_variable_numbers, np.arange(start, start + width)]
        else:
            selected_x_variable_numbers = np.r_[selected_x_variable_numbers, np.arange(start, autoscaled_x_train.shape[1])]

    # Remove duplicates and sort (keeps behavior but avoids duplicated columns)
    if selected_x_variable_numbers.size > 0:
        selected_x_variable_numbers = np.unique(selected_x_variable_numbers)

    selected_autoscaled_x_train = autoscaled_x_train[:, selected_x_variable_numbers]

    if len(selected_x_variable_numbers):
        # cross-validation
        pls_components = np.arange(
            1,
            min(np.linalg.matrix_rank(selected_autoscaled_x_train) + 1, max_number_of_components + 1),
            1
        )
        r2_cv_all = []
        for pls_component in pls_components:
            model_in_cv = PLSRegression(n_components=pls_component)

            estimated_y_train_in_cv = np.ndarray.flatten(
                model_selection.cross_val_predict(
                    model_in_cv,
                    selected_autoscaled_x_train,
                    autoscaled_y_train,
                    cv=fold_number
                )
            )
            # back to original y scale
            estimated_y_train_in_cv = estimated_y_train_in_cv * y_train.std(ddof=1) + y_train.mean()

            r2_cv_all.append(
                1 - np.sum((y_train - estimated_y_train_in_cv) ** 2) / np.sum((y_train - y_train.mean()) ** 2)
            )
        value = float(np.max(r2_cv_all))
    else:
        value = -999.0

    return value,


# -----------------
# DEAP creator guard (avoid "already exists" error)
# -----------------
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


# -----------------
# Main loop
# -----------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

r2_rmse_results = []

for number_of_areas in number_of_areas_list:
    print(f"\n=== Start: number_of_areas = {number_of_areas} ===")

    # boundaries for GA chromosome
    min_boundary = np.zeros(number_of_areas * 2)
    max_boundary = np.ones(number_of_areas * 2) * x_train.shape[1]
    max_boundary[np.arange(1, number_of_areas * 2, 2)] = max_width_of_areas

    # toolbox
    toolbox = base.Toolbox()
    toolbox.register("create_ind", create_ind_uniform, min_boundary, max_boundary)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.create_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # init population
    pop = toolbox.population(n=number_of_population)

    print("Start of evolution")

    r2_history = []
    rmse_history = []

    # evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print(f"  Evaluated {len(pop)} individuals")

    # GA generations
    for generation in range(number_of_generation):
        print(f"-- Generation {generation + 1} --")

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < probability_of_crossover:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # mutation
        for mutant in offspring:
            if random.random() < probability_of_mutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # re-evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        print(f"  Evaluated {len(invalid_ind)} individuals")

        pop[:] = offspring

        # stats
        y_train_variance = np.var(y_train, ddof=1)
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean_r2 = float(np.mean(fits))
        std_r2 = float(np.std(fits))

        rmse_values = [np.sqrt((1 - fit) * y_train_variance) for fit in fits]
        mean_rmse = float(np.mean(rmse_values))

        r2_history.append(mean_r2)
        rmse_history.append(mean_rmse)

        print(f"  Min {min(fits)}")
        print(f"  Max {max(fits)}")
        print(f"  Avg {mean_r2}")
        print(f"  Std {std_r2}")
        print(f"  Avg RMSE {mean_rmse}")

    print("-- End of evolution --")

    # best individual
    best_individual = tools.selBest(pop, 1)[0]
    best_individual_array = np.array(np.floor(best_individual), dtype=int)

    selected_x_variable_numbers = np.zeros(0, dtype=int)
    for area_number in range(number_of_areas):
        start = best_individual_array[2 * area_number]
        width = best_individual_array[2 * area_number + 1]
        start = max(0, min(start, autoscaled_x_train.shape[1] - 1))
        width = max(1, width)

        if start + width <= autoscaled_x_train.shape[1]:
            selected_x_variable_numbers = np.r_[selected_x_variable_numbers, np.arange(start, start + width)]
        else:
            selected_x_variable_numbers = np.r_[selected_x_variable_numbers, np.arange(start, autoscaled_x_train.shape[1])]

    selected_x_variable_numbers = np.unique(selected_x_variable_numbers)

    print(f"Selected variables (0-based): {selected_x_variable_numbers}")
    print(f"Fitness (best CV r2): {best_individual.fitness.values}")

    # -----------------
    # Plot r2 & RMSE over generations
    # -----------------
    generations = np.arange(1, number_of_generation + 1)

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax1.plot(generations, r2_history, marker="o")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("r2")

    ax2 = ax1.twinx()
    ax2.plot(generations, rmse_history, marker="x")
    ax2.set_ylabel("RMSE")

    plt.title(f"r2 and RMSE over Generations (Areas: {number_of_areas}, Width: {max_width_of_areas})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"r2_rmse_plot_areas_{number_of_areas}_width_{max_width_of_areas}.png", dpi=200)
    plt.show()

    results_df = pd.DataFrame({"Generation": generations, "r2": r2_history, "RMSE": rmse_history})
    results_df.to_csv(
        OUT_DIR / f"r2_and_rmse_over_generations_areas_{number_of_areas}_width_{max_width_of_areas}.csv",
        index=False
    )

    # -----------------
    # PLS with selected variables (fit on full data)
    # -----------------
    pls_model = PLSRegression(n_components=1)
    pls_model.fit(autoscaled_x_train[:, selected_x_variable_numbers], autoscaled_y_train)

    standardized_coefficients = pls_model.coef_.flatten()

    selected_wavenumbers = wavenumbers[selected_x_variable_numbers]
    selected_wavelengths_df = pd.DataFrame({
        "Selected Wavenumbers": selected_wavenumbers,
        "Standardized Coefficients": standardized_coefficients
    })
    selected_wavelengths_df.to_csv(
        OUT_DIR / f"selected_wavenumbers_with_coefficients_areas_{number_of_areas}_width_{max_width_of_areas}.csv",
        index=False
    )

    # bar plot of coefficients (invert x-axis typical for IR)
    plt.figure(figsize=(10, 6))
    plt.bar(selected_wavelengths_df["Selected Wavenumbers"], selected_wavelengths_df["Standardized Coefficients"])
    plt.xlabel("Selected Wavenumbers")
    plt.ylabel("Standardized Coefficients")
    plt.title(f"Selected Wavenumbers and Coefficients (Areas: {number_of_areas}, Width: {max_width_of_areas})")
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"selected_wavenumbers_plot_areas_{number_of_areas}_width_{max_width_of_areas}.png", dpi=200)
    plt.show()

    # -----------------
    # Actual vs Predicted (on the same training set; kept as original behavior)
    # -----------------
    predicted_y_train = pls_model.predict(autoscaled_x_train[:, selected_x_variable_numbers]).flatten()
    predicted_y_train_original = predicted_y_train * y_train.std(ddof=1) + y_train.mean()

    actual_y_train_original = autoscaled_y_train * y_train.std(ddof=1) + y_train.mean()

    plt.figure(figsize=(7, 7))
    plt.scatter(actual_y_train_original, predicted_y_train_original)
    mn = min(actual_y_train_original.min(), predicted_y_train_original.min())
    mx = max(actual_y_train_original.max(), predicted_y_train_original.max())
    plt.plot([mn, mx], [mn, mx], "k-", linewidth=2)
    plt.xlabel("Actual y")
    plt.ylabel("Predicted y")
    plt.title(f"Actual vs Predicted y (Areas: {number_of_areas}, Width: {max_width_of_areas})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"actual_vs_predicted_y_plot_areas_{number_of_areas}_width_{max_width_of_areas}.png", dpi=200)
    plt.show()

    actual_vs_predicted_df = pd.DataFrame({
        "Actual y": actual_y_train_original.flatten(),
        "Predicted y": predicted_y_train_original.flatten()
    })
    actual_vs_predicted_df.to_csv(
        OUT_DIR / f"actual_vs_predicted_y_original_scale_areas_{number_of_areas}_width_{max_width_of_areas}.csv",
        index=False
    )

    # R2 / RMSE on training fit (kept as original behavior)
    r2_value = 1 - np.sum((actual_y_train_original - predicted_y_train_original) ** 2) / np.sum(
        (actual_y_train_original - actual_y_train_original.mean()) ** 2
    )
    rmse_value = np.sqrt(np.mean((actual_y_train_original - predicted_y_train_original) ** 2))

    print(f"R2 (training fit): {r2_value}")
    print(f"RMSE (training fit): {rmse_value}")

    r2_rmse_results.append({
        "number_of_areas": number_of_areas,
        "R2_training_fit": float(r2_value),
        "RMSE_training_fit": float(rmse_value),
        "best_CV_r2_in_GA": float(best_individual.fitness.values[0]),
        "n_selected_variables": int(len(selected_x_variable_numbers)),
    })

# summary
r2_rmse_df = pd.DataFrame(r2_rmse_results)
r2_rmse_df.to_csv(OUT_DIR / f"r2_rmse_summary_width_{max_width_of_areas}.csv", index=False)
print("\nSaved summary to:", OUT_DIR / f"r2_rmse_summary_width_{max_width_of_areas}.csv")
