"""Verify derivatives, GD updates, saved predictions and sensitivity experiments."""

from pathlib import Path

import numpy as np

import gradient_descent_fit as experiment

HERE = Path(__file__).resolve().parent


def read(name):
    return np.genfromtxt(HERE / name, delimiter=",", names=True, dtype=None, encoding="utf-8")


def main():
    data = read("simulated_data.csv")
    coefficients = read("model_coefficients.csv")
    predictions = read("model_predictions.csv")
    errors = read("model_errors.csv")
    sweep = read("sensitivity_results.csv")
    summary = read("error_summary.csv")
    original = np.genfromtxt(HERE.parent / "assign3/model_errors.csv", delimiter=",", names=True)
    assert (len(data), len(coefficients), len(predictions), len(errors), len(sweep)) == (310, 60, 3060, 60, 900)
    assert np.all(errors["status"] == "converged")
    max_gradient_error = 0.0
    for degree in experiment.DEGREES:
        d = data[data["batch"] == 1]
        X = np.polynomial.polynomial.polyvander((d["temperature_C"] - 50) / 30, degree)
        y = d["measured_kOhm"]
        theta = np.random.default_rng(degree).normal(0, 0.2, degree + 1)
        gradient = 2 / len(X) * X.T @ (X @ theta - y)
        numerical = np.empty_like(theta)
        for j in range(len(theta)):
            perturbation = np.zeros_like(theta)
            perturbation[j] = 1e-5
            numerical[j] = (experiment.mse(X @ (theta + perturbation), y)
                            - experiment.mse(X @ (theta - perturbation), y)) / 2e-5
        max_gradient_error = max(max_gradient_error, np.max(np.abs(numerical - gradient)))
        np.testing.assert_allclose(gradient, numerical, rtol=1e-7, atol=1e-8)
        H = 2 / len(X) * X.T @ X
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        L = eigenvalues[-1]
        rng = np.random.default_rng(experiment.INITIALIZATION_SEED + degree)
        init = {"zero": np.zeros(degree + 1), "small_random": rng.normal(0, 0.1, degree + 1),
                "large_random": rng.normal(0, 50, degree + 1)}
        for batch in range(1, 11):
            d = data[data["batch"] == batch]
            np.testing.assert_array_equal(d["temperature_C"], np.arange(20, 81, 2))
            y = d["measured_kOhm"]
            ls = np.linalg.lstsq(X, y, rcond=None)[0]
            p = predictions[(predictions["batch"] == batch) & (predictions["degree"] == degree)]
            c = coefficients[(coefficients["batch"] == batch) & (coefficients["degree"] == degree)][0]
            normalized = np.polynomial.polynomial.polyval((p["temperature_C"] - 50) / 30,
                                                          [c[f"b{k}"] for k in range(degree + 1)])
            converted = np.polyval([c[f"a{k}"] for k in range(degree, -1, -1)], p["temperature_C"])
            np.testing.assert_allclose(normalized, p["GD_prediction_kOhm"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(converted, normalized, rtol=1e-8, atol=1e-8)
            mask = p["is_train"].astype(bool)
            assert mask.sum() == 31 and (~mask).sum() == 20
            actual = [experiment.mse(normalized[mask], y), experiment.mse(normalized[~mask], p["reference_kOhm"][~mask])]
            e = errors[(errors["batch"] == batch) & (errors["degree"] == degree)][0]
            np.testing.assert_allclose([e["train_noisy_MSE_kOhm2"], e["test_reference_MSE_kOhm2"]], actual, rtol=1e-12)
            o = original[(original["batch"] == batch) & (original["degree"] == degree)][0]
            np.testing.assert_allclose(e["train_noisy_MSE_kOhm2"], o["train_noisy_MSE_kOhm2"], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(e["test_reference_MSE_kOhm2"], o["test_reference_MSE_kOhm2"], rtol=3e-6, atol=1e-5)
            all_X = np.polynomial.polynomial.polyvander((p["temperature_C"] - 50) / 30, degree)
            for row in sweep[(sweep["batch"] == batch) & (sweep["degree"] == degree)]:
                factor = row["rate_factor"]
                np.testing.assert_allclose(row["learning_rate"], factor / L, rtol=1e-12)
                if factor > 2:
                    assert row["status"] == "diverged" and np.isnan(row["test_reference_MSE_kOhm2"])
                    continue
                # Independent closed-form trajectory of the quadratic iteration:
                # theta_k = theta* + Q (I-eta Lambda)^k Q.T (theta_0-theta*).
                components = eigenvectors.T @ (init[row["initialization"]] - ls)
                step = min(int(row["iterations"]), experiment.FIXED_BUDGET)
                budget_theta = ls + eigenvectors @ ((1 - factor / L * eigenvalues) ** step * components)
                budget_prediction = all_X @ budget_theta
                np.testing.assert_allclose(row["budget1000_train_MSE_kOhm2"], experiment.mse(budget_prediction[mask], y), rtol=1e-7, atol=1e-8)
                np.testing.assert_allclose(row["budget1000_test_MSE_kOhm2"], experiment.mse(budget_prediction[~mask], p["reference_kOhm"][~mask]), rtol=1e-6, atol=1e-5)
                final_theta = ls + eigenvectors @ ((1 - factor / L * eigenvalues) ** int(row["iterations"]) * components)
                final_prediction = all_X @ final_theta
                np.testing.assert_allclose(row["train_noisy_MSE_kOhm2"], experiment.mse(final_prediction[mask], y), rtol=1e-7, atol=1e-8)
                np.testing.assert_allclose(row["test_reference_MSE_kOhm2"], experiment.mse(final_prediction[~mask], p["reference_kOhm"][~mask]), rtol=1e-6, atol=1e-5)
                if row["status"] == "max_iterations":
                    assert row["iterations"] == experiment.MAX_ITERATIONS
                else:
                    assert row["status"] == "converged" and abs(row["train_loss_gap_kOhm2"]) < 1e-10
        e = errors[errors["degree"] == degree]
        s = summary[summary["degree"] == degree][0]
        np.testing.assert_allclose(s["test_MSE_mean_kOhm2"], e["test_reference_MSE_kOhm2"].mean(), rtol=1e-12)
        np.testing.assert_allclose(s["test_MSE_sample_std_kOhm2"], e["test_reference_MSE_kOhm2"].std(ddof=1), rtol=1e-12)

    # Check the actual update routine on a small independently solvable problem.
    X = np.column_stack((np.ones(9), np.linspace(-1, 1, 9)))
    y = X @ np.array([3.0, -2.0])
    L = np.linalg.eigvalsh(2 / len(X) * X.T @ X)[-1]
    theta, status, _, _, _ = experiment.gd_many(X, np.array([y, y]), np.zeros((2, 2)), np.array([1 / L, 2.2 / L]))
    assert status.tolist() == ["converged", "diverged"]
    np.testing.assert_allclose(theta[0], [3, -2], atol=1e-9)
    print(f"PASS: finite-difference gradients (max error {max_gradient_error:.3g}), GD update routine,")
    print("60 baseline models and coefficient conversions, 900 configurations, independent quadratic trajectories,")
    print("1000-step results, stopping states, all baseline MSEs and agreement with assign3.")


if __name__ == "__main__":
    main()
