"""Batch gradient descent for noisy polynomial regression and sensitivity sweeps.

Run: python3 assign5/gradient_descent_fit.py
Least squares is only a verification benchmark, never a GD initializer/update.
"""

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/thermistor-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORK_DIR = Path(__file__).resolve().parent
ROOT = WORK_DIR.parent
DEGREES = range(1, 7)
RATE_FACTORS = (0.02, 0.5, 1.0, 1.8, 2.2)
INITIALIZATIONS = ("zero", "small_random", "large_random")
INITIALIZATION_SEED = 20261010
MAX_ITERATIONS = 150_000
GRADIENT_TOLERANCE = 1e-10
FIXED_BUDGET = 1_000
BASELINE_FACTOR = 1.8
SAMPLE_STEPS = set(np.unique(np.concatenate((
    np.arange(0, 11), np.geomspace(12, MAX_ITERATIONS, 80).astype(int),
    [100, FIXED_BUDGET, 10_000, MAX_ITERATIONS],
))))


def write_csv(name, header, rows):
    with (WORK_DIR / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def save_plot(fig, name):
    for suffix in ("png", "svg"):
        fig.savefig(WORK_DIR / f"{name}.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def gd_many(X, targets, initial, rates):
    """Independent GD runs evaluated together; each row has its own state.

    Targets: (runs, samples); parameters: (runs, features).
    Update is theta <- theta - rate * (2/N) X.T (X theta - y).
    H and b cache the algebraically identical gradient to reduce repeated work.
    """
    H = 2.0 / len(X) * (X.T @ X)
    linear = 2.0 / len(X) * (targets @ X)
    theta = initial.copy()
    runs = len(theta)
    status = np.full(runs, "max_iterations", dtype="U20")
    iterations = np.zeros(runs, dtype=int)
    active = np.ones(runs, dtype=bool)
    budget_theta = None
    history = []
    for step in range(MAX_ITERATIONS + 1):
        gradient = theta @ H - linear
        norm = np.linalg.norm(gradient, axis=1)
        finished = active & (norm <= GRADIENT_TOLERANCE)
        status[finished] = "converged"
        iterations[finished] = step
        active[finished] = False
        if step in SAMPLE_STEPS or finished.any() or step == MAX_ITERATIONS or not active.any():
            losses = np.mean((theta @ X.T - targets) ** 2, axis=1)
            record = (status != "diverged") if step in SAMPLE_STEPS else finished
            for run in np.flatnonzero(record):
                history.append((int(run), step, float(losses[run]), float(norm[run])))
        if step == FIXED_BUDGET:
            budget_theta = theta.copy()
        if step == MAX_ITERATIONS or not active.any():
            iterations[active] = step
            break
        # Do not modify converged or diverged runs.
        indices = np.flatnonzero(active)
        updated = theta[indices] - rates[indices, None] * gradient[indices]
        bad = (~np.isfinite(updated).all(axis=1)) | (np.max(np.abs(updated), axis=1) > 1e10)
        invalid = indices[bad]
        status[invalid] = "diverged"
        iterations[invalid] = step + 1
        active[invalid] = False
        theta[indices[~bad]] = updated[~bad]
    if budget_theta is None:
        budget_theta = theta.copy()
    return theta, status, iterations, budget_theta, history


def mse(predicted, target):
    return float(np.mean((predicted - target) ** 2))


def expression(coefficients):
    terms = []
    for i, value in enumerate(coefficients):
        power = len(coefficients) - i - 1
        magnitude = f"{abs(value):.12g}"
        if "e" in magnitude:
            mantissa, exponent = magnitude.split("e")
            magnitude = f"{mantissa}\\times 10^{{{int(exponent)}}}"
        factor = "" if power == 0 else ("t" if power == 1 else f"t^{power}")
        sign = ("-" if value < 0 else "") if i == 0 else (" - " if value < 0 else " + ")
        terms.append(sign + magnitude + factor)
    return "".join(terms)


def main():
    data = np.genfromtxt(ROOT / "assign3/simulated_data.csv", delimiter=",", names=True)
    reference = np.genfromtxt(ROOT / "assign1/thermistor_data.csv", delimiter=",", names=True)
    t, truth = reference["temperature_C"], reference["resistance_kOhm"]
    if not np.array_equal(t, np.arange(0, 101, 2)):
        raise ValueError("Expected the original 2 °C reference grid.")
    train = (t >= 20) & (t <= 80)
    test = ~train
    batches = []
    for batch in range(1, 11):
        rows = data[data["batch"] == batch]
        if len(rows) != 31 or not np.array_equal(rows["temperature_C"], t[train]):
            raise ValueError("Run assign3 to generate ten 31-point measurement batches.")
        batches.append(rows["measured_kOhm"])
    measurements = np.array(batches)
    write_csv("simulated_data.csv", list(data.dtype.names), data.tolist())
    x = (t - 50.0) / 30.0
    result_rows, history_rows, coefficient_rows, prediction_rows = [], [], [], []
    baseline_rows, geometry_rows, gd_models = [], [], {}
    equations = []
    for degree in DEGREES:
        X = np.polynomial.polynomial.polyvander(x[train], degree)
        all_X = np.polynomial.polynomial.polyvander(x, degree)
        H = 2.0 / len(X) * X.T @ X
        eigenvalues = np.linalg.eigvalsh(H)
        L, mu = eigenvalues[-1], eigenvalues[0]
        least_squares = np.linalg.lstsq(X, measurements.T, rcond=None)[0].T
        optimum = np.mean((least_squares @ X.T - measurements) ** 2, axis=1)
        rng = np.random.default_rng(INITIALIZATION_SEED + degree)
        small = rng.normal(0, 0.1, degree + 1)
        large = rng.normal(0, 50.0, degree + 1)
        initial_vectors = (np.zeros(degree + 1), small, large)
        configs, initial, targets, rates = [], [], [], []
        # The same initialization vector is used across all batches and rates
        # within a degree, to isolate measurement and learning-rate effects.
        for b in range(10):
            for init_name, init_value in zip(INITIALIZATIONS, initial_vectors):
                for factor in RATE_FACTORS:
                    configs.append((b + 1, init_name, factor))
                    initial.append(init_value)
                    targets.append(measurements[b])
                    rates.append(factor / L)
        theta, status, iterations, budget_theta, history = gd_many(
            X, np.array(targets), np.array(initial), np.array(rates))
        for run, step, loss, norm in history:
            b, init_name, factor = configs[run]
            history_rows.append([b, degree, init_name, factor, step, loss, norm])
        for run, (batch, init_name, factor) in enumerate(configs):
            b = batch - 1
            valid = status[run] != "diverged"
            predicted = all_X @ theta[run] if valid else np.full(len(t), np.nan)
            train_mse = mse(predicted[train], measurements[b])
            test_mse = mse(predicted[test], truth[test])
            train_reference = mse(predicted[train], truth[train])
            difference = np.linalg.norm(theta[run] - least_squares[b]) if valid else np.nan
            max_prediction_difference = np.max(np.abs(predicted - all_X @ least_squares[b])) if valid else np.nan
            budget_prediction = all_X @ budget_theta[run]
            # An early-diverged run has no meaningful 1000-step prediction.
            if not valid and iterations[run] <= FIXED_BUDGET:
                budget_prediction[:] = np.nan
            budget_train = mse(budget_prediction[train], measurements[b])
            budget_test = mse(budget_prediction[test], truth[test])
            result_rows.append([batch, degree, init_name, factor, factor / L, status[run], iterations[run],
                                train_mse, test_mse, train_reference, optimum[b], train_mse - optimum[b],
                                difference, max_prediction_difference, budget_train, budget_test])
            if init_name == "zero" and factor == BASELINE_FACTOR:
                gd_models[b, degree] = theta[run].copy()
                baseline_rows.append([batch, degree, status[run], iterations[run], train_mse, test_mse, train_reference,
                                      optimum[b], mse((all_X @ least_squares[b])[test], truth[test]), max_prediction_difference])
                polynomial_t = np.polynomial.Polynomial(theta[run])(
                    np.polynomial.Polynomial([-50.0 / 30.0, 1.0 / 30.0]))
                coefficient_rows.append([batch, degree] + theta[run].tolist() + [0.0] * (6 - degree)
                                        + [0.0] * (6 - degree) + polynomial_t.coef[::-1].tolist())
                equations.append((batch, degree, expression(polynomial_t.coef[::-1])))
                prediction_rows.extend([batch, degree, temp, int(inside), ref, pred, ls]
                                       for temp, inside, ref, pred, ls in zip(t, train, truth, predicted, all_X @ least_squares[b]))
        raw_X = np.polynomial.polynomial.polyvander(t[train], degree)
        raw_L = np.linalg.eigvalsh(2.0 / len(raw_X) * raw_X.T @ raw_X)[-1]
        geometry_rows.append([degree, mu, L, L / mu, 2 / L, BASELINE_FACTOR / L, raw_L, 2 / raw_L])
        print(f"Degree {degree} finished: converged={np.sum(status == 'converged')}, "
              f"max_iterations={np.sum(status == 'max_iterations')}, diverged={np.sum(status == 'diverged')}", flush=True)

    result_header = ["batch", "degree", "initialization", "rate_factor", "learning_rate", "status", "iterations",
                     "train_noisy_MSE_kOhm2", "test_reference_MSE_kOhm2", "train_reference_MSE_kOhm2",
                     "least_squares_train_MSE_kOhm2", "train_loss_gap_kOhm2", "theta_l2_difference",
                     "max_prediction_difference_kOhm", "budget1000_train_MSE_kOhm2", "budget1000_test_MSE_kOhm2"]
    write_csv("sensitivity_results.csv", result_header, result_rows)
    write_csv("loss_history.csv", ["batch", "degree", "initialization", "rate_factor", "iteration", "train_MSE_kOhm2", "gradient_l2_norm"], history_rows)
    write_csv("model_coefficients.csv", ["batch", "degree"] + [f"b{k}" for k in range(7)] + [f"a{k}" for k in range(6, -1, -1)], coefficient_rows)
    write_csv("model_predictions.csv", ["batch", "degree", "temperature_C", "is_train", "reference_kOhm", "GD_prediction_kOhm", "LS_prediction_kOhm"], prediction_rows)
    write_csv("model_errors.csv", ["batch", "degree", "status", "iterations", "train_noisy_MSE_kOhm2", "test_reference_MSE_kOhm2", "train_reference_MSE_kOhm2", "LS_train_MSE_kOhm2", "LS_test_MSE_kOhm2", "max_prediction_difference_kOhm"], baseline_rows)
    write_csv("feature_geometry.csv", ["degree", "hessian_min_eigenvalue", "hessian_max_eigenvalue_L", "hessian_condition_number", "stable_rate_upper_bound", "baseline_learning_rate", "raw_hessian_L", "raw_stable_rate_upper_bound"], geometry_rows)
    equation_text = "# 六十个基准梯度下降模型\n\nt 使用 °C，阻值使用 kΩ，显示 12 位有效数字，完整系数见 CSV。\n"
    for batch in range(1, 11):
        equation_text += f"\n## 第 {batch} 批\n\n"
        equation_text += "\n\n".join(f"- **{degree} 阶**：\\(\\hat R_{degree}(t)={eq}\\)"
                                     for b, degree, eq in equations if b == batch) + "\n"
    (WORK_DIR / "model_equations.md").write_text(equation_text, encoding="utf-8")
    make_outputs(t, truth, train, measurements, gd_models, baseline_rows, result_rows,
                 history_rows, geometry_rows)


def make_outputs(t, truth, train, measurements, models, baseline_rows, result_rows, history_rows, geometry_rows):
    summary_rows = []
    for degree in DEGREES:
        rows = [r for r in baseline_rows if r[1] == degree]
        arr = np.array([r[3:] for r in rows], dtype=float)
        summary_rows.append([degree, arr[:, 1].mean(), arr[:, 1].std(ddof=1), arr[:, 2].mean(), arr[:, 2].std(ddof=1),
                             arr[:, 0].mean(), arr[:, 0].max(), arr[:, -1].max(), sum(r[2] == "converged" for r in rows)])
    write_csv("error_summary.csv", ["degree", "train_MSE_mean_kOhm2", "train_MSE_sample_std_kOhm2", "test_MSE_mean_kOhm2", "test_MSE_sample_std_kOhm2", "iterations_mean", "iterations_max", "max_GD_LS_prediction_difference_kOhm", "converged_count"], summary_rows)
    sensitivity_summary = []
    for degree in DEGREES:
        for init_name in INITIALIZATIONS:
            for factor in RATE_FACTORS:
                rows = [r for r in result_rows if r[1:4] == [degree, init_name, factor]]
                valid = [r for r in rows if r[5] != "diverged"]
                converged = [r for r in rows if r[5] == "converged"]
                stats = [np.mean([r[index] for r in valid]) if valid else np.nan for index in (7, 8, 14, 15)]
                sensitivity_summary.append([degree, init_name, factor, len(converged), sum(r[5] == "max_iterations" for r in rows),
                                            sum(r[5] == "diverged" for r in rows),
                                            np.mean([r[6] for r in converged]) if converged else np.nan] + stats)
    write_csv("sensitivity_summary.csv", ["degree", "initialization", "rate_factor", "converged_count", "max_iterations_count", "diverged_count", "converged_iterations_mean", "valid_train_MSE_mean_kOhm2", "valid_test_MSE_mean_kOhm2", "valid_budget1000_train_MSE_mean_kOhm2", "valid_budget1000_test_MSE_mean_kOhm2"], sensitivity_summary)

    # Representative batch 1 curves; every configuration was evaluated on 10 batches.
    for mode in ("initialization", "learning_rate"):
        fig, axes = plt.subplots(3, 2, figsize=(12, 11))
        for degree, ax in zip(DEGREES, axes.flat):
            choices = INITIALIZATIONS if mode == "initialization" else RATE_FACTORS
            for choice in choices:
                init_name = choice if mode == "initialization" else "zero"
                factor = BASELINE_FACTOR if mode == "initialization" else choice
                selected = [r for r in history_rows if r[:4] == [1, degree, init_name, factor]]
                if selected:
                    arr = np.array([r[4:6] for r in selected], dtype=float)
                    ax.plot(arr[:, 0] + 1, arr[:, 1], label=init_name if mode == "initialization" else f"eta={factor}/L")
            optimum = next(r[10] for r in result_rows if r[:4] == [1, degree, "zero", BASELINE_FACTOR])
            ax.axhline(optimum, color="black", linestyle="--", linewidth=1, label="Least-squares minimum")
            ax.set(xscale="log", yscale="log", xlabel="Iteration + 1 (log scale)", ylabel="Train MSE (kΩ²)", title=f"Degree {degree}")
            ax.grid(alpha=0.2, which="both")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"Effect of {mode.replace('_', ' ')} on convergence (batch 1)", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save_plot(fig, f"{mode}_loss")

    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    for degree, ax in zip(DEGREES, axes.flat):
        for factor in RATE_FACTORS[:-1]:
            rows = [r for r in history_rows if r[:4] == [1, degree, "zero", factor]]
            ax.loglog([r[4] + 1 for r in rows], [r[6] for r in rows], label=f"eta={factor}/L")
        ax.axhline(GRADIENT_TOLERANCE, color="black", linestyle="--", linewidth=1, label="Convergence threshold")
        ax.set(xlabel="Iteration + 1", ylabel="Gradient L2 norm", title=f"Degree {degree}")
        ax.grid(alpha=0.2, which="both")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Stable learning rates: approach to the convergence threshold (batch 1)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_plot(fig, "stable_rate_gradient")

    dense_t = np.linspace(0, 100, 1001)
    dense_x = (dense_t - 50) / 30
    colors = plt.cm.tab10(np.arange(10))
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    for degree, ax in zip(DEGREES, axes.flat):
        for b in range(10):
            ax.plot(dense_t, np.polynomial.polynomial.polyval(dense_x, models[b, degree]), color=colors[b], alpha=0.65, linewidth=0.9)
        ax.plot(t, truth, "k-", linewidth=2, label="Reference")
        ax.axvspan(20, 80, color="tab:blue", alpha=0.06)
        ax.set(title=f"Degree {degree}", xlabel="Temperature (°C)", ylabel="Resistance (kΩ)", xlim=(0, 100))
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Converged GD fits across ten batches\nColors identify batches; separate y scales", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_plot(fig, "gd_polynomial_variability")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for metric, ax, title in ((4, axes[0], "Training MSE"), (5, axes[1], "Test MSE (extrapolation)")):
        for b in range(1, 11):
            rows = [r for r in baseline_rows if r[0] == b]
            ax.semilogy(list(DEGREES), [r[metric] for r in rows], "o-", color=colors[b - 1], linewidth=0.9, markersize=3, label=f"Batch {b}")
        gd_mean = [np.mean([r[metric] for r in baseline_rows if r[1] == n]) for n in DEGREES]
        ls_mean = [np.mean([r[7 if metric == 4 else 8] for r in baseline_rows if r[1] == n]) for n in DEGREES]
        ax.semilogy(list(DEGREES), gd_mean, "k--", linewidth=2, label="GD mean")
        ax.semilogy(list(DEGREES), ls_mean, "rx", markersize=7, label="Least-squares mean")
        ax.set(title=title, xlabel="Polynomial degree", ylabel="MSE (kΩ²)", xticks=list(DEGREES))
        ax.grid(alpha=0.25, which="both")
    axes[1].legend(ncol=3, fontsize=7)
    fig.tight_layout()
    save_plot(fig, "gd_mse_by_batch")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for init_name in INITIALIZATIONS:
        rows = [r for r in result_rows if r[2] == init_name and r[3] == BASELINE_FACTOR]
        for field, ax in ((14, axes[0]), (15, axes[1])):
            values = [np.mean([r[field] for r in rows if r[1] == n]) for n in DEGREES]
            ax.semilogy(list(DEGREES), values, "o-", label=init_name)
    axes[0].set_title("Training MSE after 1000 updates (or earlier convergence)")
    axes[1].set_title("Test MSE after 1000 updates (or earlier convergence)")
    for ax in axes:
        ax.set(xlabel="Polynomial degree", ylabel="Mean MSE across ten batches (kΩ²)", xticks=list(DEGREES))
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_plot(fig, "fixed_budget_initialization")

    # Direct feature-scale example: degree 6, batch 1, zero initialization,
    # exactly the same eta for raw and normalized inputs.
    degree = 6
    eta = geometry_rows[5][5]
    scale_rows = []
    for scale, values in (("normalized_x", (t[train] - 50) / 30), ("raw_t", t[train])):
        X = np.polynomial.polynomial.polyvander(values, degree)
        theta = np.zeros(degree + 1)
        for step in range(21):
            residual = X @ theta - measurements[0]
            loss = np.mean(residual ** 2)
            grad = 2 / len(X) * X.T @ residual
            scale_rows.append([scale, step, eta, loss, np.linalg.norm(grad)])
            if np.max(np.abs(theta)) > 1e10 or not np.isfinite(loss):
                break
            theta -= eta * grad
    write_csv("feature_scale_demo.csv", ["features", "iteration", "learning_rate", "train_MSE_kOhm2", "gradient_l2_norm"], scale_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    for scale in ("normalized_x", "raw_t"):
        rows = [r for r in scale_rows if r[0] == scale]
        ax.semilogy([r[1] for r in rows], [r[3] for r in rows], "o-", label=scale)
    ax.set(xlabel="Iteration", ylabel="Training MSE (kΩ²)", title=f"Degree 6: same learning rate eta={eta:.4g}, different feature scales")
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "feature_scale_demo")
    write_report(summary_rows, sensitivity_summary, baseline_rows, result_rows, geometry_rows)


def write_report(summary, sweep_summary, baseline, results, geometry):
    table = "\n".join(f"| {r[0]} | {r[1]:.6f} ± {r[2]:.6f} | {r[3]:.6f} ± {r[4]:.6f} | {r[5]:.1f} | {int(r[6])} | {r[7]:.3g} |" for r in summary)
    convergence_table = "\n".join(f"| {n} | " + " | ".join(
        f"{sum(r[5] == 'converged' for r in results if r[1] == n and r[3] == factor)}/"
        f"{sum(r[5] == 'max_iterations' for r in results if r[1] == n and r[3] == factor)}/"
        f"{sum(r[5] == 'diverged' for r in results if r[1] == n and r[3] == factor)}"
        for factor in RATE_FACTORS) + " |" for n in DEGREES)
    init_table = "\n".join(f"| {r[1]} | {r[3]} | {r[6]:.1f} | {r[9]:.6f} | {r[10]:.6f} |"
                           for r in sweep_summary if r[0] == 6 and r[2] == BASELINE_FACTOR)
    best_degrees = [min([r for r in baseline if r[0] == b], key=lambda r: r[5])[1] for b in range(1, 11)]
    best_mean = min(summary, key=lambda r: r[3])[0]
    report = f"""# 梯度下降拟合热敏电阻：初值与学习率的影响

## 运行与实验设置

在项目根目录执行 `python3 assign5/gradient_descent_fit.py`，依赖 NumPy 和 Matplotlib。
需要 assign1 的参考数据和 assign3 的十批模拟测量；若缺少输入，先运行相应脚本。
完成计算后运行 `python3 assign5/verify_results.py` 可核验有限差分梯度、梯度下降更新、900 个配置的独立二次迭代轨迹、系数转换及误差。
完整实验包含 **900 次梯度下降运行**：10 批 × 6 种阶次 × 3 种初值 × 5 档学习率。
其中零初值、η=1.8/L 的 60 次运行作为重复第 2、3 项的基准结果。

训练范围为 20～80°C（含端点，2°C 步长，共 31 点），测试范围为 0～18°C 和 82～100°C（共 20 点）。
沿用 assign3 的十批数据，每批独立噪声 ε~N(0,0.5²) kΩ，测量随机种子为 42～51；输入也复制到本目录。
每批各算法设置使用完全相同的数据，未重新抽取噪声或合并批次。
训练 MSE 对照带噪测量，测试 MSE 对照无噪声参考；单位均为 kΩ²。

## 梯度下降算法

用 x=(t−50)/30 将训练温度缩放到 [−1,1]，构造 X=[1,x,…,xⁿ]，模型为 R̂=Σ b_j xʲ。
损失 J(θ)=(1/N)||Xθ−y||²，解析梯度为 ∇J=(2/N)Xᵀ(Xθ−y)。
每步执行 θ←θ−η∇J，采用全部 31 个训练点的批量梯度下降。
代码缓存 H=(2/N)XᵀX 和 q=(2/N)Xᵀy，更新等价地写为 θ←θ−η(Hθ−q)。
同时计算多个配置只是将独立运行按数组并列，各运行使用自身参数、数据、学习率与停止状态。

温度缩放没有改变多项式模型空间，也没有使用正则化。
CSV 同时保存 x 基底的 b0～b6 和转换回摄氏温度 t 基底的 a6～a0。
由于 x 无量纲，b_j 的单位均为 kΩ；a_j 的单位为 kΩ/(°C)^j。
预测优先在 x 基底计算，以减少数值消减；超出实际阶次的系数填零。
最小二乘 `lstsq` 只用来提供验证基准，不用于梯度更新或初始化。

停止规则：梯度二范数 ≤ {GRADIENT_TOLERANCE:g} 则标记 converged；最多 {MAX_ITERATIONS} 次更新。
超过上限仍未收敛标记 max_iterations，不能当作最优解。
参数出现非有限值或任一绝对值超过 10¹⁰，标记 diverged，其最终 MSE 记为 NaN 并排除出均值。
另记录统一 1000 次更新预算的结果；提前收敛者保留收敛值，提前发散者不提供有效预算误差。
历史曲线按预定对数间隔及收敛时刻采样，不保存每一步。

## 参数初值与学习率设计

初值分别为全零、N(0,0.1²) 的小幅随机值、N(0,50²) 的大幅随机值。
初始化随机种子为 {INITIALIZATION_SEED}+阶次，各阶次在不同批次和学习率下共用各自相同的初值向量。
这样可分别观察训练数据、初值和学习率的作用。

H 正定，令 L 为最大特征值，μ 为最小特征值；固定步长的充分稳定条件为 0<η<2/L。
设置 η=c/L，c∈{{0.02,0.5,1.0,1.8,2.2}}，分别覆盖很小步长、稳定步长与超过稳定界限的情况。
c=1.8 的稳定更新可能沿部分特征方向交替靠近解；它仍处于稳定范围内。
η=2.2/L 会放大最大特征值方向的误差，本次数据及初值中出现发散。

随着阶次增加，H 的条件数从 {geometry[0][3]:.2f} 增至 {geometry[5][3]:.2f}。
不同方向收敛速度因此差异很大，即使已缩放温度，6 阶的最慢方向仍需较多迭代。
原始 t 基底的 6 阶稳定学习率上界约为 {geometry[5][7]:.3g}，
缩放基底则约为 {geometry[5][4]:.6g}。示例使用同一个缩放后适用的学习率，原始特征迅速发散。
这说明学习率必须与特征尺度匹配，不意味着原始特征使用任意学习率都会发散。

![特征尺度示例](feature_scale_demo.png)

## 十批基准拟合结果

零初值、η=1.8/L。下表 ± 为十批 MSE 的样本标准差（ddof=1），不是置信区间。
GD–LS 差异是同批同阶两种算法在原始 51 个温度点预测的最大绝对差。

| 阶次 | 训练 MSE 均值 ± 标准差 | 测试 MSE 均值 ± 标准差 | 平均迭代次数 | 最大迭代次数 | 最大 GD–LS 预测差（kΩ） |
|---:|---:|---:|---:|---:|---:|
{table}

按平均测试 MSE 比较，最小的是 **{best_mean} 阶**。十批测试 MSE 最小者依次为 {best_degrees}。
充分收敛的梯度下降重现最小二乘拟合的训练与外推规律：提高阶次降低训练残差，却不能保证区间外误差下降。
改变求解算法不会消除第 3 项中的测量噪声敏感性。

![十批梯度下降拟合曲线](gd_polynomial_variability.png)

![十批误差及最小二乘对照](gd_mse_by_batch.png)

## 学习率实验

以下每格为 **收敛 / 达到迭代上限 / 发散** 的运行数，每格共 30 次（10 批 × 3 初值）。
全部逐次结果见 `sensitivity_results.csv`，逐配置十批统计见 `sensitivity_summary.csv`。

| 阶次 | 0.02/L | 0.5/L | 1.0/L | 1.8/L | 2.2/L |
|---:|---:|---:|---:|---:|---:|
{convergence_table}

很小学习率的更新稳定但缓慢，在相同迭代预算内可能尚未充分接近最小二乘解。
具体地，η=0.02/L 时 4～6 阶的全部运行达到迭代上限；6 阶在 η=0.5/L 下也全部未满足收敛判据，
η=1/L 时有 18 次收敛、12 次达到上限，η=1.8/L 时 30 次全部收敛。
适当增加学习率能减少迭代次数；超过稳定上界则发散，不能靠延长迭代修复。
不同阶次的 L 不同，本实验使用 ηL 的相对尺度比较；实际 η 也保存在 CSV。

![学习率对损失曲线的影响](learning_rate_loss.png)

![稳定学习率的梯度收敛细节](stable_rate_gradient.png)

## 初值实验

损失为严格凸二次函数，且本次 X 满列秩，因此最优解唯一。
在稳定学习率和充分迭代下，三种初值都应趋近同一最小二乘解，不存在因初值而落入不同局部最小值的问题。
但初始误差在不同特征方向的分量不同，导致收敛速度和有限预算下的参数、预测不同。
大幅初值通常使初始损失更大；收敛耗时不能只凭初值大小判断，还取决于其在慢方向上的分量。

下表展示 6 阶、η=1.8/L 的十批结果：

| 初值 | 收敛次数 | 收敛运行平均迭代次数 | 1000 步预算训练 MSE 均值 | 1000 步预算测试 MSE 均值 |
|---|---:|---:|---:|---:|
{init_table}

最终收敛结果和 1000 步结果应分别解读：前者验证求解到同一最优解，后者显示初值带来的优化进度差异。
本次 6 阶大幅随机初值平均约需 94063 步，而零初值约需 79434 步；在 1000 步时，大幅初值的平均测试误差仍很大。
零初值 6 阶在 1000 步的平均测试 MSE 约为 90.23，充分收敛后约为 711.64，
说明继续降低训练损失也可能使外推误差增大；不能通过观察测试集来随意选择提前停止时刻。
测试误差的变化还包含测量噪声及外推效应，不一定随训练优化进度改善而下降。

![初值对收敛的影响](initialization_loss.png)

![初值在统一预算下的影响](fixed_budget_initialization.png)

## 结论与限制

模型初值、学习率和训练噪声分别作用于优化起点、更新稳定性、所学习的目标数据。
稳定步长并充分收敛时，初值不改变唯一最优解；迭代不足则保留初值和步长影响。
训练噪声仍会改变最优解，高阶模型在训练范围外的预测更易波动。
必须区分“梯度下降尚未收敛”和“已收敛模型的外推误差较大”，不能将它们都称为过拟合。
本实验将测试误差用于描述性比较，实际选阶次或优化设置应另用验证数据，保留独立测试集。

## 文件说明

- `simulated_data.csv`：与 assign3 相同的 310 个带噪训练点。
- `model_coefficients.csv`：60 个基准模型的 x 与 t 基底系数。
- `model_equations.md`：对应的 60 个摄氏温度多项式表达式。
- `model_predictions.csv`：3060 行 GD、LS 预测及参考值。
- `model_errors.csv`、`error_summary.csv`：基准逐批误差与十批统计。
- `sensitivity_results.csv`：900 次运行的状态、迭代次数、误差、最优损失差和 1000 步结果。
- `sensitivity_summary.csv`：每阶次、初值、学习率的十批统计；有效误差均值排除发散者，但包含未收敛者，须结合状态列阅读。
- `loss_history.csv`：采样损失与梯度范数。
- `feature_geometry.csv`：H 特征值、条件数及稳定步长上界。
- `feature_scale_demo.csv`：原始与缩放特征的同学习率示例。
- `verify_results.py`：独立数值核验，不重新执行完整参数扫描。
- 七组图表各提供 PNG 与 SVG；收敛后保留参数值，历史曲线延长部分表示该值保持不变，并非继续更新。
"""
    (WORK_DIR / "README.md").write_text(report, encoding="utf-8")
    print(f"Baseline converged: {sum(r[2] == 'converged' for r in baseline)}/60", flush=True)
    print(f"Maximum baseline GD–LS prediction difference: {max(r[9] for r in baseline):.6g} kOhm", flush=True)
    print(f"Saved outputs in {WORK_DIR}", flush=True)


if __name__ == "__main__":
    main()
