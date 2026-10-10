# 热敏电阻特性曲线建模

本目录复现热敏电阻阻值与温度关系的指数模型：

\[
R(T)=R_{T_0}\exp\left[\beta\left(\frac{1}{T}-\frac{1}{T_0}\right)\right]
\]

参数为：

- 参考温度：`T0 = 25 °C = 298.15 K`
- 参考阻值：`R(T0) = 22 kΩ`
- β：`3100 K`
- 温度步长：`2 °C`
- 温度范围：`0 °C` 至 `100 °C`

## 运行

在本目录执行：

```bash
python3 thermistor_curve.py
```

程序会生成：

- `thermistor_data.csv`：温度和电阻计算数据；
- `thermistor_curve.svg`：热敏电阻特性曲线（浏览器可直接打开）。
- `thermistor_curve.png`：热敏电阻特性曲线（适合作业插图）。

模型计算中的温度必须使用开尔文，绘图横坐标使用摄氏温度。
