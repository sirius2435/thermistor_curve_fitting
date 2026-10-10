"""Fit degree 1–6 polynomials to simulated thermistor measurements.

Run from any directory with: python3 assign2/thermistor_polynomial_fit.py
The physical model is used only to supply reference data, never for fitting.
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
SOURCE = WORK_DIR.parent / "assign1" / "thermistor_data.csv"
SEED = 42
NOISE_STD_KOHM = 0.5
DEGREES = range(1, 7)


def write_csv(name, header, rows):
    with (WORK_DIR / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def save_plot(figure, name):
    figure.savefig(WORK_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    figure.savefig(WORK_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)


def mse(prediction, target):
    return float(np.mean((prediction - target) ** 2))


def polynomial_text(coefficients):
    terms = []
    degree = len(coefficients) - 1
    for index, coefficient in enumerate(coefficients):
        power = degree - index
        factor = "" if power == 0 else ("t" if power == 1 else f"t^{power}")
        magnitude = f"{abs(coefficient):.12g}"
        if "e" in magnitude:
            mantissa, exponent = magnitude.split("e")
            magnitude = f"{mantissa}\\times 10^{{{int(exponent)}}}"
        if index == 0:
            terms.append(("-" if coefficient < 0 else "") + magnitude + factor)
        else:
            terms.append((" - " if coefficient < 0 else " + ") + magnitude + factor)
    return "".join(terms)


def main():
    reference = np.genfromtxt(SOURCE, delimiter=",", names=True)
    temperature = reference["temperature_C"]
    truth = reference["resistance_kOhm"]
    if not np.array_equal(temperature, np.arange(0, 101, 2)):
        raise ValueError("Source must contain the 51 temperatures 0, 2, ..., 100 °C.")
    train = (temperature >= 20) & (temperature <= 80)
    test = ~train
    train_t = temperature[train]
    train_truth = truth[train]
    noise = np.random.default_rng(SEED).normal(0, NOISE_STD_KOHM, train.sum())
    measured = train_truth + noise

    # All degrees use exactly the same measurements. Test values have no noise.
    coefficients = {}
    error_rows = []
    prediction_columns = [temperature, truth, train.astype(int)]
    for degree in DEGREES:
        coeff = np.polyfit(train_t, measured, degree)
        coefficients[degree] = coeff
        predicted = np.polyval(coeff, temperature)
        prediction_columns.append(predicted)
        error_rows.append([
            degree,
            mse(predicted[train], measured),
            mse(predicted[test], truth[test]),
            mse(predicted[train], train_truth),
        ])

    write_csv("simulated_data.csv", ["temperature_C", "reference_kOhm", "noise_kOhm", "measured_kOhm"],
              zip(train_t, train_truth, noise, measured))
    # Explicit column names a6...a0 avoid ambiguity about coefficient ordering.
    write_csv("model_coefficients.csv", ["degree"] + [f"a{i}" for i in range(6, -1, -1)],
              ([degree] + [0.0] * (6 - degree) + coefficients[degree].tolist() for degree in DEGREES))
    write_csv("model_errors.csv", ["degree", "train_noisy_MSE_kOhm2", "test_reference_MSE_kOhm2", "train_reference_MSE_kOhm2"], error_rows)
    write_csv("model_predictions.csv", ["temperature_C", "reference_kOhm", "is_train"] + [f"degree_{n}_prediction_kOhm" for n in DEGREES],
              np.column_stack(prediction_columns))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(temperature, truth, color="black", label="Reference characteristic")
    ax.scatter(train_t, measured, s=24, color="tab:blue", label="Measurements (Gaussian noise, sigma=0.5 kOhm)", zorder=3)
    ax.axvspan(20, 80, color="tab:blue", alpha=0.07, label="Training interval [20, 80] °C")
    ax.set(xlabel="Temperature (°C)", ylabel="Resistance (kΩ)", title="Simulated thermistor measurements")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "simulated_measurements")

    dense_t = np.linspace(0, 100, 1001)
    # A dense line is only for display; all MSEs use the original 2 °C grid.
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    for ax, row in zip(axes.flat, error_rows):
        degree, train_error, test_error, _ = row
        ax.plot(temperature, truth, color="black", linewidth=1.5, label="Reference")
        ax.plot(dense_t, np.polyval(coefficients[degree], dense_t), color="tab:orange", label="Polynomial")
        ax.scatter(train_t, measured, s=13, color="tab:blue", label="Measurements", zorder=3)
        ax.scatter(temperature[test], truth[test], s=18, marker="x", color="tab:green", label="Test reference", zorder=3)
        ax.axvspan(20, 80, color="tab:blue", alpha=0.07)
        ax.axvline(20, color="gray", linestyle=":", linewidth=0.8)
        ax.axvline(80, color="gray", linestyle=":", linewidth=0.8)
        ax.set(title=f"Degree {degree}: train MSE={train_error:.4g}, test MSE={test_error:.4g}",
               xlabel="Temperature (°C)", ylabel="Resistance (kΩ)", xlim=(0, 100))
        ax.grid(alpha=0.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Polynomial fits and extrapolation (shaded region: training)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_plot(fig, "polynomial_fits")

    fig, ax = plt.subplots(figsize=(9, 5))
    local_t = np.linspace(20, 80, 601)
    ax.plot(train_t, train_truth, "k--", linewidth=2, label="Reference")
    ax.scatter(train_t, measured, color="black", s=18, label="Measurements", zorder=4)
    for degree in DEGREES:
        ax.plot(local_t, np.polyval(coefficients[degree], local_t), label=f"Degree {degree}")
    ax.set(xlabel="Temperature (°C)", ylabel="Resistance (kΩ)", title="Polynomial fits within the training interval", xlim=(20, 80))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    save_plot(fig, "training_fits")

    errors = np.array(error_rows)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(errors[:, 0], errors[:, 1], "o-", label="Train MSE (noisy measurements)")
    ax.semilogy(errors[:, 0], errors[:, 2], "s-", label="Test MSE (reference, outside training interval)")
    ax.semilogy(errors[:, 0], errors[:, 3], "^--", label="Train-interval MSE (reference, diagnostic)")
    ax.set(xlabel="Polynomial degree", ylabel="Mean squared error (kΩ², log scale)",
           title="Error versus polynomial degree", xticks=list(DEGREES))
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "mse_vs_degree")

    best = int(errors[np.argmin(errors[:, 2]), 0])
    equations = "\n\n".join(f"- **{degree} 阶**：\\(\\hat R_{degree}(t)={polynomial_text(coefficients[degree])}\\)" for degree in DEGREES)
    table = "\n".join(f"| {int(n)} | {tr:.8f} | {te:.8f} | {ref:.8f} |" for n, tr, te, ref in error_rows)
    report = f"""# 热敏电阻多项式拟合与误差分析

## 运行方法与文件

在项目根目录执行 `python3 assign2/thermistor_polynomial_fit.py`。依赖 `numpy` 和 `matplotlib`。
脚本从自身位置定位输入和输出，亦可在其他目录运行。所有输出保存在 `assign2/`。

- `simulated_data.csv`：31 个训练点的参考阻值、噪声和模拟测量值。
- `model_coefficients.csv`：各模型系数，列按 a6 至 a0 排列，超出模型阶次的项填零。
- `model_errors.csv`：六种模型的 MSE，单位均为 kΩ²。
- `model_predictions.csv`：51 个原始温度点的预测结果；is_train=1 表示训练点。
- `simulated_measurements.png/svg`：模拟测量数据。
- `polynomial_fits.png/svg`：六种模型在 0～100°C 上的曲线。
- `training_fits.png/svg`：训练范围内的拟合细节。
- `mse_vs_degree.png/svg`：误差随阶次变化，纵轴采用对数刻度。

## 数据与方法

本实验读取第 1 项的 `assign1/thermistor_data.csv`，其阻值保留六位小数。
参考数据来自 R(t)=22 exp[3100(1/(t+273.15)−1/298.15)] kΩ，但拟合过程只使用温度和模拟测量值，不使用该公式。

训练温度为 20, 22, …, 80°C，共 31 个点；测试温度为 0, 2, …, 18°C 和 82, 84, …, 100°C，共 20 个点。
训练、测试集无重叠，合起来覆盖原始 51 个点。20°C 和 80°C 均属于训练集。

对训练阻值加上独立高斯噪声 ε ~ N(0, 0.5²)，单位为 kΩ，即标准差 500Ω。
使用 NumPy `default_rng({SEED})`，且六种模型使用同一组测量数据。
本次噪声的样本均值为 {np.mean(noise):.6f} kΩ，样本标准差（ddof=1）为 {np.std(noise, ddof=1):.6f} kΩ。
零均值和 0.5 kΩ 指生成分布的参数；有限样本的统计量无需恰好等于它们，也未对噪声重新中心化。

使用 `numpy.polyfit`（对应 MATLAB 的 `polyfit`）分别求解 1～6 阶多项式的最小二乘问题：

\\[
\\min_{{a_0,\\ldots,a_n}}\\sum_{{i=1}}^{{31}}[\\hat R_n(t_i)-R_i^{{\\mathrm{{measured}}}}]^2,
\\qquad \\hat R_n(t)=a_nt^n+\\cdots+a_1t+a_0.
\\]

这里 t 使用 °C，阻值使用 kΩ；a_j 的单位为 kΩ/(°C)^j。
所有 MSE 在 2°C 的原始采样点上计算，图中加密绘制的曲线点不参与误差计算。

## 六种多项式模型

下列系数展示 12 位有效数字；CSV 保留浮点输出精度，预测和误差计算使用未格式化的系数。

{equations}

## 误差结果

MSE = (1/N) Σ(预测阻值 − 对照阻值)²，单位为 kΩ²。
训练 MSE 对照带噪测量值，测试 MSE 对照未加噪声的参考值。
另列训练范围内对照参考值的误差作为诊断，不参与拟合。

| 阶次 | 训练 MSE（带噪） | 测试 MSE（参考值） | 训练范围 MSE（参考值，辅助） |
|---:|---:|---:|---:|
{table}

![误差随阶次变化](mse_vs_degree.png)

## 结果讨论

训练 MSE 随阶次增加而不升高，因为高阶多项式包含低阶模型的全部候选解。
这一性质只针对同一组数据上的未正则化最小二乘残差，不能保证对真实曲线的预测更好。
1 阶模型无法充分描述阻值随温度变化的弯曲形状，训练误差较大，体现欠拟合。
加入二次和三次项后，训练误差明显下降；更高阶模型继续降低残差，也可能把部分测量噪声纳入曲线。

本次实验中，测试 MSE 最小的是 **{best} 阶**，值为 **{errors[best - 1, 2]:.8f} kΩ²**。
测试误差从 1 阶到 3 阶连续下降，4 阶回升至 {errors[3, 2]:.8f}，
5 阶降至 {errors[4, 2]:.8f}，6 阶又显著增大至 {errors[5, 2]:.8f} kΩ²，呈非单调变化。
与此同时，4～6 阶的训练 MSE 仅有小幅改善，训练范围参考 MSE 却均高于 3 阶，
说明更小的训练残差并未带来更好的真实曲线逼近，表现出拟合噪声的迹象。
高次幂在训练区间外会放大系数和噪声的影响，使外推预测可能大幅偏离参考曲线，甚至产生负阻值。
图中 1 阶模型在高温端预测出负阻值；4～6 阶模型在高温端出现向上弯曲，偏离参考曲线的单调下降趋势。
因此训练残差很小也不意味着模型能准确预测 0～20°C 和 80～100°C 的特性。

这里的测试集全部位于训练区间外，考查的是**外推能力**。
外推误差同时受到多项式表达能力、训练范围和测量噪声影响，不能仅凭测试 MSE 就将全部误差归因于过拟合。
训练误差与测试误差还使用不同的对照值，应注意训练残差包含测量噪声，而测试误差直接衡量对参考曲线的偏离。
辅助的训练范围参考 MSE 可用于观察训练残差下降是否伴随真实曲线误差下降。

最小测试误差的阶次只是这次固定噪声实现下的比较结果，并非所有实验的通用最优阶次。
本作业按要求报告测试结果；若实际用于选择模型阶次，应在训练范围内另设验证集或采用交叉验证，并保留独立测试集作最终评估。

![六种模型的拟合与外推](polynomial_fits.png)

![训练范围拟合细节](training_fits.png)

![模拟测量数据](simulated_measurements.png)
"""
    (WORK_DIR / "README.md").write_text(report, encoding="utf-8")
    print("degree | train noisy MSE | test reference MSE | train reference MSE (kOhm^2)")
    for degree, train_error, test_error, reference_error in error_rows:
        print(f"{degree:6d} | {train_error:15.8f} | {test_error:18.8f} | {reference_error:.8f}")
    print(f"Lowest test MSE: degree {best}; outputs saved in {WORK_DIR}")


if __name__ == "__main__":
    main()
