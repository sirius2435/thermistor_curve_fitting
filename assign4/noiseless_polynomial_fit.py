"""Repeat polynomial fitting ten times without noise; compare with assign3."""

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
REPEATS = 10
DEGREES = range(1, 7)


def write_csv(name, header, rows):
    with (WORK_DIR / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def save_plot(fig, name):
    for suffix in ("png", "svg"):
        fig.savefig(WORK_DIR / f"{name}.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def expression(coefficients):
    result = []
    for i, value in enumerate(coefficients):
        power = len(coefficients) - i - 1
        magnitude = f"{abs(value):.12g}"
        if "e" in magnitude:
            mantissa, exponent = magnitude.split("e")
            magnitude = f"{mantissa}\\times 10^{{{int(exponent)}}}"
        factor = "" if power == 0 else ("t" if power == 1 else f"t^{power}")
        sign = ("-" if value < 0 else "") if i == 0 else (" - " if value < 0 else " + ")
        result.append(sign + magnitude + factor)
    return "".join(result)


def main():
    reference = np.genfromtxt(ROOT / "assign1/thermistor_data.csv", delimiter=",", names=True)
    noisy_errors = np.genfromtxt(ROOT / "assign3/model_errors.csv", delimiter=",", names=True)
    noisy_coeff = np.genfromtxt(ROOT / "assign3/model_coefficients.csv", delimiter=",", names=True)
    t, truth = reference["temperature_C"], reference["resistance_kOhm"]
    if not np.array_equal(t, np.arange(0, 101, 2)):
        raise ValueError("Expected 51 reference temperatures: 0, 2, ..., 100 °C.")
    if len(noisy_errors) != 60 or len(noisy_coeff) != 60:
        raise ValueError("Run assign3/repeated_polynomial_fit.py to generate ten noisy batches first.")
    train = (t >= 20) & (t <= 80)
    test = ~train
    noise = np.zeros((REPEATS, train.sum()))
    measured = truth[train] + noise
    coefficients = {}
    predictions = np.empty((REPEATS, 6, len(t)))
    errors = np.empty((REPEATS, 6, 2))
    coefficient_rows, prediction_rows, error_rows = [], [], []
    for b in range(REPEATS):
        for j, degree in enumerate(DEGREES):
            # Refit on every repetition, although noise=0 leaves the data unchanged.
            coeff = np.polyfit(t[train], measured[b], degree)
            coefficients[b, j] = coeff
            predicted = np.polyval(coeff, t)
            predictions[b, j] = predicted
            errors[b, j] = [np.mean((predicted[train] - measured[b]) ** 2),
                            np.mean((predicted[test] - truth[test]) ** 2)]
            coefficient_rows.append([b + 1, degree] + [0.0] * (6 - degree) + coeff.tolist())
            error_rows.append([b + 1, degree] + errors[b, j].tolist())
            prediction_rows.extend([b + 1, degree, x, int(inside), r, p]
                                   for x, inside, r, p in zip(t, train, truth, predicted))

    # With a deterministic solver and identical inputs, every repetition agrees exactly.
    if not all(np.array_equal(predictions[b], predictions[0]) for b in range(REPEATS)):
        raise RuntimeError("Identical noiseless inputs unexpectedly produced different predictions.")
    write_csv("simulated_data.csv", ["batch", "temperature_C", "reference_kOhm", "noise_kOhm", "measured_kOhm"],
              ([b + 1, x, r, eps, y] for b in range(REPEATS)
               for x, r, eps, y in zip(t[train], truth[train], noise[b], measured[b])))
    write_csv("model_coefficients.csv", ["batch", "degree"] + [f"a{k}" for k in range(6, -1, -1)], coefficient_rows)
    write_csv("model_predictions.csv", ["batch", "degree", "temperature_C", "is_train", "reference_kOhm", "prediction_kOhm"], prediction_rows)
    write_csv("model_errors.csv", ["batch", "degree", "train_reference_MSE_kOhm2", "test_reference_MSE_kOhm2"], error_rows)
    # Identical values have exactly zero mathematical dispersion. Copy the common
    # value rather than introducing rounding artifacts by averaging ten copies.
    summary_rows = [[degree, errors[0, j, 0], 0.0, errors[0, j, 1], 0.0,
                     np.ptp(predictions[:, j], axis=0).max(),
                     np.ptp(np.array([coefficients[b, j] for b in range(REPEATS)]), axis=0).max()]
                    for j, degree in enumerate(DEGREES)]
    write_csv("error_summary.csv", ["degree", "train_MSE_mean_kOhm2", "train_MSE_sample_std_kOhm2", "test_MSE_mean_kOhm2", "test_MSE_sample_std_kOhm2", "max_prediction_range_kOhm", "max_coefficient_range"], summary_rows)

    comparisons = []
    for j, degree in enumerate(DEGREES):
        rows = noisy_errors[noisy_errors["degree"] == degree]
        if len(rows) != REPEATS:
            raise ValueError(f"Expected ten noisy models for degree {degree}.")
        comparisons.append([degree, errors[0, j, 0], errors[0, j, 1],
                            rows["train_noisy_MSE_kOhm2"].mean(), rows["train_reference_MSE_kOhm2"].mean(),
                            rows["test_reference_MSE_kOhm2"].mean(), rows["test_reference_MSE_kOhm2"].std(ddof=1),
                            rows["test_reference_MSE_kOhm2"].mean() / errors[0, j, 1]])
    write_csv("noise_comparison.csv", ["degree", "noiseless_train_reference_MSE_kOhm2", "noiseless_test_reference_MSE_kOhm2", "noisy_train_measured_MSE_mean_kOhm2", "noisy_train_reference_MSE_mean_kOhm2", "noisy_test_reference_MSE_mean_kOhm2", "noisy_test_MSE_sample_std_kOhm2", "noisy_to_noiseless_test_MSE_ratio"], comparisons)
    comparison = np.array(comparisons)
    best = int(np.argmin(errors[0, :, 1]) + 1)
    best_noisy = int(np.argmin(comparison[:, 5]) + 1)
    write_csv("best_degree_by_batch.csv", ["batch", "lowest_test_MSE_degree", "lowest_test_MSE_kOhm2"],
              ([b + 1, best, errors[b, best - 1, 1]] for b in range(REPEATS)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, truth, "k-", label="Reference")
    for b in range(REPEATS):
        ax.scatter(t[train], measured[b], s=18, color="tab:blue", label="10 identical noiseless batches" if b == 0 else None, zorder=3)
    ax.axvspan(20, 80, color="tab:blue", alpha=0.06)
    ax.set(xlabel="Temperature (°C)", ylabel="Resistance (kΩ)", title="Zero noise: all ten measurement batches overlap")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "noiseless_measurements")

    dense_t = np.linspace(0, 100, 1001)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    for j, ax in enumerate(axes.flat):
        degree = j + 1
        for b in range(REPEATS):
            ax.plot(dense_t, np.polyval(coefficients[b, j], dense_t), color="tab:orange", linewidth=1.4,
                    label="Noiseless fit (10 curves overlap)" if b == 0 else None)
        ax.plot(t, truth, "k-", linewidth=1.5, label="Reference")
        ax.scatter(t[train], measured[0], s=14, color="tab:blue", label="Training values", zorder=3)
        ax.scatter(t[test], truth[test], s=18, marker="x", color="tab:green", label="Test reference", zorder=3)
        ax.axvspan(20, 80, color="tab:blue", alpha=0.06)
        ax.set(title=f"Degree {degree}: train MSE={errors[0, j, 0]:.3g}, test MSE={errors[0, j, 1]:.3g}",
               xlabel="Temperature (°C)", ylabel="Resistance (kΩ)", xlim=(0, 100))
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Noiseless polynomial fitting, repeated ten times", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_plot(fig, "noiseless_fits")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for j, ax in enumerate(axes):
        ax.semilogy(list(DEGREES), errors[0, :, j], "o-", color="tab:orange", label="Noiseless (10 identical runs)")
        metric = "train_reference_MSE_kOhm2" if j == 0 else "test_reference_MSE_kOhm2"
        for b in range(1, REPEATS + 1):
            rows = noisy_errors[noisy_errors["batch"] == b]
            ax.semilogy(rows["degree"], rows[metric], color="gray", alpha=0.4, linewidth=0.8,
                        label="Noisy batches (assign3)" if b == 1 else None)
        ax.semilogy(list(DEGREES), comparison[:, 4 if j == 0 else 5], "s--", color="tab:blue", label="Noisy: mean of 10 runs")
        ax.set(title="Training interval: reference MSE" if j == 0 else "Test interval: reference MSE",
               xlabel="Polynomial degree", ylabel="MSE (kΩ², log scale)", xticks=list(DEGREES))
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_plot(fig, "noise_mse_comparison")

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    for j, ax in enumerate(axes.flat):
        degree = j + 1
        rows = noisy_coeff[noisy_coeff["degree"] == degree]
        for i, row in enumerate(rows):
            coeff = [row[f"a{k}"] for k in range(degree, -1, -1)]
            ax.plot(dense_t, np.polyval(coeff, dense_t), color="gray", alpha=0.5, linewidth=0.9,
                    label="Noisy fits (10 batches)" if i == 0 else None)
        ax.plot(t, truth, "k-", linewidth=2, label="Reference")
        ax.plot(dense_t, np.polyval(coefficients[0, j], dense_t), color="tab:orange", linewidth=2, label="Noiseless fit")
        ax.axvspan(20, 80, color="tab:blue", alpha=0.06)
        ax.set(title=f"Degree {degree}", xlabel="Temperature (°C)", ylabel="Resistance (kΩ)", xlim=(0, 100))
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Effect of measurement noise on fitted curves\nShaded region: training; separate y scales", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_plot(fig, "noise_curve_comparison")

    equations = "\n\n".join(f"- **{degree} 阶**：\\(\\hat R_{degree}(t)={expression(coefficients[0, degree - 1])}\\)" for degree in DEGREES)
    results_table = "\n".join(f"| {degree} | {errors[0, j, 0]:.10g} | {errors[0, j, 1]:.10g} | 0 | 0 |" for j, degree in enumerate(DEGREES))
    comparison_table = "\n".join(f"| {int(row[0])} | {row[1]:.10g} | {row[4]:.8f} | {row[2]:.8f} | {row[5]:.8f} ± {row[6]:.8f} | {row[7]:.2f} |" for row in comparisons)
    report = f"""# 零噪声条件下的多项式拟合及重复实验

## 实验方法

在项目根目录运行 `python3 assign4/noiseless_polynomial_fit.py`，依赖 NumPy 和 Matplotlib。
脚本使用 assign1 的参考数据，并读取 assign3 的十批带噪模型作比较；必要时先运行 `python3 assign3/repeated_polynomial_fit.py`。

保持第 2、3 项的条件：步长 2°C，训练温度 20～80°C（含端点，共 31 点），
测试温度 0～18°C 和 82～100°C（共 20 点），用最小二乘拟合 1～6 阶多项式。
区别是将测量噪声方差设为 **0**，即 ε=0，模拟测量值直接等于参考阻值。
每个阶次仍重新拟合十次，共执行 60 次拟合，未将十批数据合并。
无噪声时，训练温度和测量值均不变；十次数据完全相同，并非十批不同的随机样本。

MSE = (1/N) Σ(预测阻值 − 参考阻值)²，单位为 kΩ²。
零噪声时训练测量值等于参考值，因此训练残差 MSE 和训练范围参考 MSE 相同。
原始 CSV 阻值保留六位小数，这种输入舍入仍然存在；“零噪声”指不再添加随机测量噪声。
图中加密点仅用于显示，误差均使用原始 2°C 采样网格。

## 模型与误差

t 使用 °C，阻值使用 kΩ；a_j 的单位为 kΩ/(°C)^j。十次得到的同阶模型完全相同，下列仅列一次。
系数显示 12 位有效数字，CSV 保存完整浮点精度。

{equations}

| 阶次 | 训练 MSE（十次相同） | 测试 MSE（十次相同） | 训练 MSE 跨批标准差 | 测试 MSE 跨批标准差 |
|---:|---:|---:|---:|---:|
{results_table}

同一阶次的系数、预测和误差在十次实验中完全一致，跨批极差和标准差均为零。
脚本检查十组预测逐元素相等；汇总直接取共同值和零标准差，避免计算十个相同浮点数的均值时引入无意义的舍入波动。
本次 1～6 阶范围内，测试 MSE 最小的是 **{best} 阶**，值为 **{errors[0, best - 1, 1]:.10g} kΩ²**，十次均如此。

![无噪声数据](noiseless_measurements.png)

![无噪声拟合曲线](noiseless_fits.png)

## 与第 3 项带噪实验比较

为使比较使用相同对照值，下表两种条件的训练范围误差均对照无噪声参考值。
带噪实验的训练残差（对照带噪测量）也保存在 `noise_comparison.csv`，但它与下表的训练参考误差含义不同。
带噪测试列给出十批均值 ± 样本标准差（ddof=1）；所有 MSE 单位为 kΩ²。

| 阶次 | 无噪声训练参考 MSE | 带噪训练参考 MSE 均值 | 无噪声测试 MSE | 带噪测试 MSE 均值 ± 标准差 | 带噪/无噪声平均测试误差比 |
|---:|---:|---:|---:|---:|---:|
{comparison_table}

![噪声对误差的影响](noise_mse_comparison.png)

![噪声对曲线的影响](noise_curve_comparison.png)

## 观察与讨论

**去除噪声消除了本实验的批次差异。** 第 3 项中不同测量噪声会改变最小二乘系数，训练区间外的预测尤为敏感。
本项温度、阻值和算法全部固定，因此重复训练得到同一组系数和曲线，没有随机训练数据引起的波动。
重复十次只验证可重复性，不增加独立信息，也不会通过重复操作进一步改善模型。

**本次无噪声数据支持更高阶的逼近。** 随阶次从 1 增至 6，训练 MSE 和测试 MSE 均连续下降。
1 阶模型仍有明显欠拟合；增加阶次改善了对真实曲线弯曲形状的表达。
第 3 项中平均测试 MSE 最小者为 {best_noisy} 阶，而本项为 {best} 阶：
带噪时提高阶次可能放大测量扰动，无噪声时这项不稳定性消失，当前范围内提高阶次带来的近似改善占主导。

**噪声对高阶外推的影响更明显。** 6 阶无噪声测试 MSE 为 {errors[0, 5, 1]:.8f} kΩ²，
第 3 项带噪的平均测试 MSE 为 {comparison[5, 5]:.8f} kΩ²，后者约为前者的 {comparison[5, 7]:.2f} 倍。
曲线图显示带噪高阶模型在训练区间内较接近，却在低温端和高温端明显分散；无噪声高阶曲线的外推更接近参考值。
这说明较小的测量噪声也可能通过高次项在外推区间被显著放大。

**零噪声仍不意味着零误差。** 用有限阶多项式近似原来的指数特性曲线，存在模型近似误差；
最小二乘仅优化训练区间，不直接优化区间外预测，因此所有阶次的测试误差仍为正。
本次在 1～6 阶内测试误差下降是实际观察结果，不是“无噪声时任意提高阶次必然改善外推”的普遍定理。
更高阶、更远外推或不同采样条件下仍可能出现数值问题或近似偏差。

本比较揭示了模型表达能力与训练数据噪声的共同作用：低阶模型偏差较大，高阶模型更灵活但更容易放大噪声。
减少测量噪声有助于稳定学习结果，使本例中的高阶拟合更准确；模型评估仍需区分训练拟合与区间外预测。

## 输出文件

- `simulated_data.csv`：十批共 310 个训练点，noise_kOhm 列全为零。
- `model_coefficients.csv`：60 行模型系数，按 a6～a0 排列，超出阶次的项填零。
- `model_predictions.csv`：60 个模型在 51 个温度点上的 3060 行预测。
- `model_errors.csv`、`error_summary.csv`：逐批误差及零波动汇总。
- `best_degree_by_batch.csv`：每批测试 MSE 最小者。
- `noise_comparison.csv`：与 assign3 的逐阶次对照统计。
- 四组图表均提供 PNG 和 SVG。
"""
    (WORK_DIR / "README.md").write_text(report, encoding="utf-8")
    print("degree | noiseless train MSE | noiseless test MSE | noisy mean test MSE (kOhm^2)")
    for row in comparisons:
        print(f"{int(row[0]):6d} | {row[1]:19.10g} | {row[2]:18.10g} | {row[5]:.8f}")
    print(f"All ten repetitions identical; lowest noiseless test MSE: degree {best}.")
    print(f"Saved outputs in {WORK_DIR}")


if __name__ == "__main__":
    main()
