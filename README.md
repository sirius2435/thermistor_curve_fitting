# 数据驱动的热敏电阻特性曲线建模

复现热敏电阻特性曲线、带噪多项式最小二乘拟合、十批重复实验、零噪声对照，以及梯度下降初值与学习率实验。

## 目录

| 目录 | 内容 | 入口程序 |
|---|---|---|
| assign1 | 0～100°C，2°C 步长的参考特性曲线 | thermistor_curve.py |
| assign2 | 单批带噪数据的 1～6 阶最小二乘拟合 | thermistor_polynomial_fit.py |
| assign3 | 十批独立噪声实验与统计 | repeated_polynomial_fit.py |
| assign4 | 零噪声十次拟合及带噪对照 | noiseless_polynomial_fit.py |
| assign5 | 梯度下降与 900 次初值、学习率实验 | gradient_descent_fit.py |

每个目录包含实验说明、CSV 数据和 PNG/SVG 图表。`experiment_report.md` 为综合实验报告文字稿。

## 环境与复现

本次验证环境：Python 3.13.13、NumPy 2.5.3、Matplotlib 3.11.2。
在项目根目录按顺序运行：

```bash
python3 -m pip install -r requirements.txt
python3 assign1/thermistor_curve.py
python3 assign2/thermistor_polynomial_fit.py
python3 assign3/repeated_polynomial_fit.py
python3 assign4/noiseless_polynomial_fit.py
python3 assign5/gradient_descent_fit.py
python3 assign5/verify_results.py
```

若希望限制梯度下降实验的 BLAS 线程，可使用：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 assign5/gradient_descent_fit.py
```

## 实验条件

- 参考阻值：25°C 时 22 kΩ；β=3100 K。
- 参考模型中的温度用 K，拟合温度用 °C。
- 训练温度：20～80°C，包含端点，31 点。
- 测试温度：0～18°C、82～100°C，20 点，考查外推。
- 带噪数据：高斯噪声标准差 0.5 kΩ；单批种子 42，十批种子 42～51。
- MSE 单位：kΩ²。训练 MSE 默认对照带噪测量，测试 MSE 对照参考值；辅助参考误差另行标注。
- 梯度下降：温度缩放 x=(t−50)/30，稳定性由 Hessian 最大特征值 L 决定。

## 代码仓库

仓库：https://github.com/sirius2435/thermistor_curve_fitting
