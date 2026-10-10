"""Repeat the assign2 experiment with ten independent measurement batches."""

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/thermistor-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WORK_DIR = Path(__file__).resolve().parent
SOURCE = WORK_DIR.parent / "assign1" / "thermistor_data.csv"
SEEDS = np.arange(42, 52)
DEGREES = np.arange(1, 7)
NOISE_STD_KOHM = 0.5


def write_csv(name, header, rows):
    with (WORK_DIR / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def save_plot(fig, name):
    for suffix in ("png", "svg"):
        fig.savefig(WORK_DIR / f"{name}.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def expression(coeff):
    terms = []
    for i, value in enumerate(coeff):
        power = len(coeff) - i - 1
        magnitude = f"{abs(value):.12g}"
        if "e" in magnitude:
            base, exponent = magnitude.split("e")
            magnitude = f"{base}\\times 10^{{{int(exponent)}}}"
        factor = "" if power == 0 else ("t" if power == 1 else f"t^{power}")
        sign = ("-" if value < 0 else "") if i == 0 else (" - " if value < 0 else " + ")
        terms.append(sign + magnitude + factor)
    return "".join(terms)


def main():
    reference = np.genfromtxt(SOURCE, delimiter=",", names=True)
    t, truth = reference["temperature_C"], reference["resistance_kOhm"]
    if not np.array_equal(t, np.arange(0, 101, 2)):
        raise ValueError("Expected 51 reference temperatures: 0, 2, ..., 100 °C.")
    train = (t >= 20) & (t <= 80)
    test = ~train
    # Separate reproducible streams; first batch reproduces assign2 (seed=42).
    noise = np.array([np.random.default_rng(int(seed)).normal(0, NOISE_STD_KOHM, train.sum()) for seed in SEEDS])
    measured = truth[train] + noise
    predictions = np.empty((len(SEEDS), len(DEGREES), len(t)))
    errors = np.empty((len(SEEDS), len(DEGREES), 3))
    dense_t = np.linspace(0, 100, 1001)
    dense_predictions = np.empty((len(SEEDS), len(DEGREES), len(dense_t)))
    coefficient_rows, error_rows, prediction_rows, equations = [], [], [], []
    coefficients = {}
    for b, seed in enumerate(SEEDS):
        equations.append(f"## 第 {b + 1} 批（seed={seed}）\n")
        for j, degree in enumerate(DEGREES):
            coeff = np.polyfit(t[train], measured[b], int(degree))
            coefficients[b, j] = coeff
            predicted = np.polyval(coeff, t)
            predictions[b, j] = predicted
            dense_predictions[b, j] = np.polyval(coeff, dense_t)
            errors[b, j] = [
                np.mean((predicted[train] - measured[b]) ** 2),
                np.mean((predicted[test] - truth[test]) ** 2),
                np.mean((predicted[train] - truth[train]) ** 2),
            ]
            coefficient_rows.append([b + 1, seed, degree] + [0.0] * (6 - degree) + coeff.tolist())
            error_rows.append([b + 1, seed, degree] + errors[b, j].tolist())
            prediction_rows.extend([b + 1, seed, degree, x, int(inside), r, pred]
                                   for x, inside, r, pred in zip(t, train, truth, predicted))
            equations.append(f"- **{degree} 阶**：\\(\\hat R_{degree}(t)={expression(coeff)}\\)\n")

    write_csv("simulated_data.csv", ["batch", "seed", "temperature_C", "reference_kOhm", "noise_kOhm", "measured_kOhm"],
              ([b + 1, seed, x, r, eps, y] for b, seed in enumerate(SEEDS)
               for x, r, eps, y in zip(t[train], truth[train], noise[b], measured[b])))
    write_csv("model_coefficients.csv", ["batch", "seed", "degree"] + [f"a{i}" for i in range(6, -1, -1)], coefficient_rows)
    write_csv("model_errors.csv", ["batch", "seed", "degree", "train_noisy_MSE_kOhm2", "test_reference_MSE_kOhm2", "train_reference_MSE_kOhm2"], error_rows)
    write_csv("model_predictions.csv", ["batch", "seed", "degree", "temperature_C", "is_train", "reference_kOhm", "prediction_kOhm"], prediction_rows)
    (WORK_DIR / "model_equations.md").write_text(
        "# 十批实验的 60 个多项式模型\n\nt 使用 °C，阻值使用 kΩ。系数显示 12 位有效数字，完整精度见 model_coefficients.csv。\n\n"
        + "\n".join(equations), encoding="utf-8")

    summary_header = ["degree"]
    for metric in ("train_noisy", "test_reference", "train_reference"):
        summary_header.extend(f"{metric}_MSE_{stat}_kOhm2" for stat in ("mean", "std", "median", "min", "max"))
    summary_rows = []
    for j, degree in enumerate(DEGREES):
        row = [degree]
        for metric in range(3):
            values = errors[:, j, metric]
            row.extend([values.mean(), values.std(ddof=1), np.median(values), values.min(), values.max()])
        summary_rows.append(row)
    write_csv("error_summary.csv", summary_header, summary_rows)

    best_degrees = DEGREES[np.argmin(errors[:, :, 1], axis=1)]
    best_rows = [[b + 1, seed, degree, errors[b, degree - 1, 1]] for b, (seed, degree) in enumerate(zip(SEEDS, best_degrees))]
    write_csv("best_degree_by_batch.csv", ["batch", "seed", "lowest_test_MSE_degree", "lowest_test_MSE_kOhm2"], best_rows)
    noise_rows = [[b + 1, seed, noise[b].mean(), noise[b].std(ddof=1)] for b, seed in enumerate(SEEDS)]
    write_csv("noise_statistics.csv", ["batch", "seed", "sample_mean_kOhm", "sample_std_kOhm"], noise_rows)

    coefficient_summary = []
    for j, degree in enumerate(DEGREES):
        values = np.array([coefficients[b, j] for b in range(len(SEEDS))])
        for column, power in enumerate(range(int(degree), -1, -1)):
            coefficient_summary.append([degree, f"a{power}", values[:, column].mean(), values[:, column].std(ddof=1)])
    write_csv("coefficient_summary.csv", ["degree", "coefficient", "mean", "sample_std"], coefficient_summary)

    average_prediction = predictions.mean(axis=0)
    population_variance = predictions.var(axis=0, ddof=0)
    spread_rows = [[degree, x, int(inside), mean, std] for j, degree in enumerate(DEGREES)
                   for x, inside, mean, std in zip(t, train, average_prediction[j], predictions[:, j].std(axis=0, ddof=1))]
    write_csv("prediction_spread.csv", ["degree", "temperature_C", "is_train", "prediction_mean_kOhm", "prediction_sample_std_kOhm"], spread_rows)
    decomposition = []
    for j, degree in enumerate(DEGREES):
        for region, mask in (("train_reference", train), ("test_reference", test)):
            bias2 = np.mean((average_prediction[j, mask] - truth[mask]) ** 2)
            variance = population_variance[j, mask].mean()
            mean_mse = np.mean((predictions[:, j, mask] - truth[mask]) ** 2)
            decomposition.append([degree, region, bias2, variance, mean_mse])
    write_csv("empirical_bias_variance.csv", ["degree", "region", "squared_bias_kOhm2", "variance_ddof0_kOhm2", "mean_reference_MSE_kOhm2"], decomposition)

    colors = plt.cm.tab10(np.arange(10))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t[train], truth[train], "k-", linewidth=2, label="Reference", zorder=5)
    for b in range(10):
        ax.plot(t[train], measured[b], ".-", color=colors[b], linewidth=0.7, alpha=0.65, label=f"Batch {b + 1}")
    ax.set(xlabel="Temperature (°C)", ylabel="Resistance (kΩ)", title="Ten independent measurement batches", xlim=(20, 80))
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    save_plot(fig, "measurement_batches")

    for name, limits, title in (
        ("polynomial_variability", (0, 100), "Ten fitted curves per degree: full range and extrapolation"),
        ("training_variability", (20, 80), "Ten fitted curves per degree: training interval"),
    ):
        fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
        dense_mask = (dense_t >= limits[0]) & (dense_t <= limits[1])
        for j, ax in enumerate(axes.flat):
            for b in range(10):
                ax.plot(dense_t[dense_mask], dense_predictions[b, j, dense_mask], color=colors[b], alpha=0.65, linewidth=0.9)
            mean = dense_predictions[:, j, dense_mask].mean(axis=0)
            ax.plot(dense_t[dense_mask], mean, color="tab:red", linestyle="--", linewidth=2, label="Mean fitted prediction")
            ref_mask = (t >= limits[0]) & (t <= limits[1])
            ax.plot(t[ref_mask], truth[ref_mask], "k-", linewidth=2, label="Reference", zorder=5)
            ax.axvspan(20, 80, color="tab:blue", alpha=0.05)
            ax.set(xlabel="Temperature (°C)", ylabel="Resistance (kΩ)", title=f"Degree {DEGREES[j]}", xlim=limits)
            ax.grid(alpha=0.25)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(title + "\nColors identify batches; panels use separate y scales", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save_plot(fig, name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for metric, ax, title in ((0, axes[0], "Training MSE (noisy measurements)"), (1, axes[1], "Test MSE (reference; extrapolation)")):
        for b in range(10):
            ax.semilogy(DEGREES, errors[b, :, metric], "o-", color=colors[b], linewidth=1, markersize=3, label=f"Batch {b + 1}")
        ax.set(xlabel="Polynomial degree", ylabel="MSE (kΩ², log scale)", title=title, xticks=DEGREES)
        ax.grid(alpha=0.25, which="both")
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    save_plot(fig, "mse_by_batch")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for metric, ax, title in ((0, axes[0], "Training MSE distribution"), (1, axes[1], "Test MSE distribution")):
        ax.boxplot(errors[:, :, metric], positions=DEGREES, showfliers=True)
        ax.plot(DEGREES, errors[:, :, metric].mean(axis=0), "s--", color="tab:red", label="Mean across 10 batches")
        for b in range(10):
            ax.scatter(DEGREES + (b - 4.5) * 0.025, errors[b, :, metric], color=colors[b], s=13, alpha=0.7, zorder=3)
        ax.set_yscale("log")
        ax.set(xlabel="Polynomial degree", ylabel="MSE (kΩ², log scale)", title=title, xticks=DEGREES)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save_plot(fig, "mse_distribution")

    mean_errors = errors.mean(axis=0)
    std_errors = errors.std(axis=0, ddof=1)
    best_mean = int(DEGREES[np.argmin(mean_errors[:, 1])])
    summary_table = "\n".join(f"| {degree} | {mean_errors[j, 0]:.6f} ± {std_errors[j, 0]:.6f} | {mean_errors[j, 1]:.6f} ± {std_errors[j, 1]:.6f} | {np.median(errors[:, j, 1]):.6f} | {mean_errors[j, 2]:.6f} |"
                              for j, degree in enumerate(DEGREES))
    batch_table = "\n".join(f"| {batch} | {seed} | {degree} | {value:.6f} |" for batch, seed, degree, value in best_rows)
    count_text = "、".join(f"{degree} 阶 {np.sum(best_degrees == degree)} 次" for degree in DEGREES if np.any(best_degrees == degree))
    bv_table = "\n".join(f"| {degree} | {bias:.6f} | {var:.6f} | {total:.6f} |" for degree, region, bias, var, total in decomposition if region == "test_reference")
    report = f"""# 十批训练数据对热敏电阻多项式学习结果的影响

## 实验设计与复现

在项目根目录运行 `python3 assign3/repeated_polynomial_fit.py`。依赖 NumPy 和 Matplotlib，脚本也可从其他目录运行。
本实验重复第 2 项 **10 次**，每次都重新生成噪声，再独立拟合 1～6 阶多项式，共获得 **60 个模型**。
输入为 `assign1/thermistor_data.csv`，温度步长保持 2°C。
训练范围为 20～80°C（包含两端，共 31 点），测试范围为 0～18°C 和 82～100°C（共 20 点）。
参考值在全部批次中固定，拟合只用带噪训练数据，不利用物理模型公式。

每批噪声独立生成自 N(0, 0.5²)，单位为 kΩ，即标准差 500Ω。
随机种子依次为 42～51，保证可复现；第 1 批与 assign2 相同，其余 9 批使用新噪声。
每批内部的六种模型共用同一组测量数据，不同批次分别训练，未将 10 批数据合并。
温度采样位置保持不变，因此本实验只考查测量噪声造成的训练数据变化。
零均值指生成分布的期望，实际 31 个噪声样本的均值不必恰好为零；各批噪声统计见 `noise_statistics.csv`。

使用 `numpy.polyfit` 做最小二乘拟合，用 `numpy.polyval` 预测。
MSE = (1/N) Σ(预测阻值 − 对照阻值)²，单位为 kΩ²。
训练 MSE 对照本批带噪测量值；测试 MSE 对照固定的无噪声参考值。
另计算训练区间相对于参考值的 MSE，用来观察对真实曲线的逼近。
所有误差都在原始 2°C 网格上计算，图中加密曲线仅用于显示。

## 十次实验的统计结果

下表的“±”表示十批结果的**样本标准差（ddof=1）**，不是置信区间；所有误差单位为 kΩ²。
完整统计（均值、标准差、中位数、最小值、最大值）见 `error_summary.csv`。

| 阶次 | 训练 MSE 均值 ± 标准差 | 测试 MSE 均值 ± 标准差 | 测试 MSE 中位数 | 训练范围参考 MSE 均值 |
|---:|---:|---:|---:|---:|
{summary_table}

![每批训练与测试误差](mse_by_batch.png)

![误差分布](mse_distribution.png)

每批测试 MSE 最小的阶次如下，属于实验后的描述性比较，不是独立验证的模型选择：

| 批次 | 随机种子 | 测试 MSE 最小的阶次 | 最小测试 MSE（kΩ²） |
|---:|---:|---:|---:|
{batch_table}

出现次数为：**{count_text}**。按十批测试 MSE 的平均值比较，最小的是 **{best_mean} 阶**，均值为 **{mean_errors[best_mean - 1, 1]:.6f} kΩ²**。

具体而言，3 阶的测试 MSE 为 {mean_errors[2, 1]:.6f} ± {std_errors[2, 1]:.6f} kΩ²，
4 阶为 {mean_errors[3, 1]:.6f} ± {std_errors[3, 1]:.6f} kΩ²。
4 阶平均误差略低，但波动明显更大；这十批结果支持“3 阶较稳定、4 阶平均误差略低”的描述，不能据此断言二者差异具有统计显著性。
5 阶和 6 阶的平均测试误差分别为 {mean_errors[4, 1]:.6f} 和 {mean_errors[5, 1]:.6f} kΩ²，且波动很大。

## 对学习结果的影响

**同一阶次的结果取决于训练数据。** 噪声改变每个测量点的位置，最小二乘得到的系数随之改变，预测曲线和 MSE 也发生变化。
`model_coefficients.csv` 给出 60 组系数，`model_equations.md` 列出对应表达式；`coefficient_summary.csv` 给出每个系数跨批次的均值和样本标准差。
不同幂次系数的单位和量级不同（a_j 单位为 kΩ/(°C)^j），不能直接比较它们的绝对标准差来判断模型稳定性，曲线及预测误差更直观。

**训练误差降低不代表外推更可靠。** 每批训练 MSE 都随阶次增加而不升高，因为高阶多项式包含低阶模型的候选解。
但不同批次的测试误差曲线不同，同一批内也可能出现先降后升或其他非单调变化。
1 阶模型表达能力有限，曲线偏差明显；高阶模型虽然更灵活，却对噪声更敏感，尤其在训练区间之外，高次项会放大预测差异。
对比训练区间图和全范围图，可以看到区间内曲线相近并不意味着区间外预测一致。

**一次实验不能确定普遍最优阶次。** assign2 中第 1 批的测试 MSE 最小者为 3 阶，本实验中各批最小者的分布为 {count_text}。
应结合平均误差和波动程度讨论模型；较低的单次误差可能只是该批噪声的结果。
测试误差均值还可能被少量较大的外推误差影响，因此同时报告中位数和全部批次结果。

## 参考误差的经验偏差与方差分解

为量化训练噪声影响，在每个固定温度点计算十个模型预测的平均值与跨批方差：
平均参考 MSE = 平均预测相对于参考曲线的平方偏差 + 预测的跨批方差。
此处方差使用 **ddof=0**，使有限十批的分解精确成立；它不同于上表统计误差时使用的样本标准差。
以下在测试集 20 个温度点上取平均，单位均为 kΩ²：

| 阶次 | 经验平方偏差 | 经验预测方差 | 二者之和（平均测试 MSE） |
|---:|---:|---:|---:|
{bv_table}

该分解针对无噪声参考值，不另加测试测量噪声项。
低阶模型的系统性曲线偏差和高阶模型的训练噪声敏感性可分别通过这两项观察。
在本次结果中，1 阶测试误差主要来自平方偏差；6 阶的经验预测方差为 {population_variance[5, test].mean():.6f} kΩ²，
占其平均测试 MSE 的 {population_variance[5, test].mean() / mean_errors[5, 1] * 100:.2f}%，说明更换训练噪声导致的预测波动是其误差的主要来源。
这只是十批实现的经验统计，不能当作无限次实验的精确期望。
`prediction_spread.csv` 还列出各阶次在每个温度点的预测均值和样本标准差。

测试点全在训练范围外，误差反映外推能力，也受到多项式近似的局限影响，不能把全部误差归因于过拟合。
如实际选择模型，应在训练数据内另设验证机制，并保留独立测试集作最终评估。
本实验只有十批，结论适用于当前噪声水平、训练范围与数据采样；固定种子用于复现，不消除学习结果的随机性。

![十批模拟测量](measurement_batches.png)

![训练范围内模型变化](training_variability.png)

![全范围模型变化](polynomial_variability.png)

## 输出文件

- `simulated_data.csv`：310 行，十批模拟测量。
- `noise_statistics.csv`：每批噪声样本均值和标准差。
- `model_coefficients.csv`、`model_equations.md`：60 个模型的系数和表达式，CSV 系数按 a6～a0 排列，超出阶次的项填零。
- `model_errors.csv`：60 行，逐批逐阶次的三种 MSE。
- `model_predictions.csv`：3060 行，每批每阶次在 51 个原始温度上的预测。
- `error_summary.csv`、`best_degree_by_batch.csv`：误差汇总和每批测试误差最小者。
- `coefficient_summary.csv`、`prediction_spread.csv`：系数及预测随批次变化的统计。
- `empirical_bias_variance.csv`：训练范围与测试范围的经验偏差、方差分解。
- 五组图表均提供 PNG 与 SVG。
"""
    (WORK_DIR / "README.md").write_text(report, encoding="utf-8")
    print("degree | train MSE mean ± std | test MSE mean ± std (kOhm^2)")
    for j, degree in enumerate(DEGREES):
        print(f"{degree:6d} | {mean_errors[j, 0]:.6f} ± {std_errors[j, 0]:.6f} | {mean_errors[j, 1]:.6f} ± {std_errors[j, 1]:.6f}")
    print(f"Best test degrees by batch: {best_degrees.tolist()}; lowest mean test MSE: degree {best_mean}")
    print(f"Saved outputs in {WORK_DIR}")


if __name__ == "__main__":
    main()
