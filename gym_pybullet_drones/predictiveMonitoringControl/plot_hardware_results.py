import pandas as pd
import matplotlib.pyplot as plt
import sys


folder=sys.argv[1]


data=pd.read_csv(
    folder+"/oracle_log.csv"
)


t=data["time"]-data["time"].iloc[0]


plt.figure(figsize=(8,4))

plt.plot(
    t,
    data["distance"]
)

plt.xlabel("Time (s)")
plt.ylabel("Distance (m)")
plt.grid()

plt.savefig(
    folder+"/distance.png",
    dpi=300
)



plt.figure(figsize=(8,4))

plt.plot(
    t,
    data["risk"]
)

plt.xlabel("Time (s)")
plt.ylabel("Risk")
plt.grid()

plt.savefig(
    folder+"/risk.png",
    dpi=300
)



plt.figure(figsize=(8,4))

plt.plot(
    t,
    data["vx"],
    label="vx"
)

plt.plot(
    t,
    data["vy"],
    label="vy"
)

plt.legend()
plt.grid()

plt.xlabel("Time (s)")
plt.ylabel("Velocity")

plt.savefig(
    folder+"/velocity.png",
    dpi=300
)

print("Figures saved")