"""NTC thermistor resistance-temperature characteristic.

Model:
    R(T) = R0 * exp(beta * (1/T - 1/T0))

Temperatures in the model are absolute temperatures (K).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


WORK_DIR = Path(__file__).resolve().parent
BETA_K = 3100.0
T0_C = 25.0
T0_K = T0_C + 273.15
R0_KOHM = 22.0


def resistance(t_kelvin: np.ndarray) -> np.ndarray:
    """Return resistance in kOhm for absolute temperature in kelvin."""
    return R0_KOHM * np.exp(BETA_K * (1.0 / t_kelvin - 1.0 / T0_K))


def write_svg(temperature_c: np.ndarray, resistance_kohm: np.ndarray, path: Path) -> None:
    """Write a self-contained SVG plot without external plotting packages."""
    width, height = 900, 560
    left, right, top, bottom = 85, 35, 35, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    x_min, x_max = 0.0, 100.0
    y_min, y_max = 0.0, 60.0

    def x_px(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def y_px(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    points = " ".join(f"{x_px(x):.2f},{y_px(y):.2f}" for x, y in zip(temperature_c, resistance_kohm))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="24" text-anchor="middle" font-family="Arial" font-size="19">NTC Thermistor Resistance–Temperature Characteristic</text>',
    ]
    for x in range(0, 101, 20):
        xp = x_px(x)
        parts.append(f'<line x1="{xp:.2f}" y1="{top}" x2="{xp:.2f}" y2="{top + plot_h}" stroke="#dddddd"/>')
        parts.append(f'<text x="{xp:.2f}" y="{height - 48}" text-anchor="middle" font-family="Arial" font-size="13">{x}</text>')
    for y in range(0, 61, 10):
        yp = y_px(y)
        parts.append(f'<line x1="{left}" y1="{yp:.2f}" x2="{left + plot_w}" y2="{yp:.2f}" stroke="#dddddd"/>')
        parts.append(f'<text x="{left - 10}" y="{yp + 5:.2f}" text-anchor="end" font-family="Arial" font-size="13">{y}</text>')
    parts.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<polyline points="{points}" fill="none" stroke="#1769aa" stroke-width="2.2"/>',
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 12}" text-anchor="middle" font-family="Arial" font-size="15">Temperature (°C)</text>',
        f'<text x="18" y="{top + plot_h / 2:.2f}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2:.2f})" font-family="Arial" font-size="15">Resistance (kΩ)</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    temperature_c = np.arange(0.0, 100.0 + 2.0, 2.0)
    temperature_k = temperature_c + 273.15
    resistance_kohm = resistance(temperature_k)

    data = np.column_stack((temperature_c, temperature_k, resistance_kohm))
    np.savetxt(
        WORK_DIR / "thermistor_data.csv",
        data,
        delimiter=",",
        header="temperature_C,temperature_K,resistance_kOhm",
        comments="",
        fmt=["%.0f", "%.2f", "%.6f"],
    )

    write_svg(temperature_c, resistance_kohm, WORK_DIR / "thermistor_curve.svg")

    plt.figure(figsize=(8, 5.2), dpi=150)
    plt.plot(temperature_c, resistance_kohm, "o-", linewidth=1.5, markersize=3.5,
             label=r"$R(T)=R_0 e^{\beta(1/T-1/T_0)}$")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Resistance (kΩ)")
    plt.title("NTC Thermistor Resistance–Temperature Characteristic")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(WORK_DIR / "thermistor_curve.png", bbox_inches="tight")
    plt.close()

    print(f"Saved: {WORK_DIR / 'thermistor_data.csv'}")
    print(f"Saved: {WORK_DIR / 'thermistor_curve.svg'}")
    print(f"Saved: {WORK_DIR / 'thermistor_curve.png'}")
    print(f"R(0 °C)   = {resistance_kohm[0]:.3f} kΩ")
    print(f"R(25 °C)  = {resistance(np.array([T0_K]))[0]:.3f} kΩ")
    print(f"R(100 °C) = {resistance_kohm[-1]:.3f} kΩ")


if __name__ == "__main__":
    main()
