import pandas as pd
import matplotlib.pyplot as plt

acc = pd.read_csv(r"C:\Users\ikili\Downloads\acc-gyr 2026-05-06 23-40-03\Accelerometer.csv")
gyr = pd.read_csv(r"C:\Users\ikili\Downloads\acc-gyr 2026-05-06 23-40-03\Gyroscope.csv")

print("Accelerometer Spalten:", acc.columns.tolist())
print("Gyroscope Spalten:", gyr.columns.tolist())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(acc.iloc[:, 0], acc.iloc[:, 1], label="X")
ax1.plot(acc.iloc[:, 0], acc.iloc[:, 2], label="Y")
ax1.plot(acc.iloc[:, 0], acc.iloc[:, 3], label="Z")
ax1.set_title("Accelerometer")
ax1.set_ylabel("Beschleunigung (m/s²)")
ax1.legend()
ax1.grid(True)

ax2.plot(gyr.iloc[:, 0], gyr.iloc[:, 1], label="X")
ax2.plot(gyr.iloc[:, 0], gyr.iloc[:, 2], label="Y")
ax2.plot(gyr.iloc[:, 0], gyr.iloc[:, 3], label="Z")
ax2.set_title("Gyroskop")
ax2.set_ylabel("Drehrate (rad/s)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()