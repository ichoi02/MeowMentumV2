import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot Robot Telemetry Data")
    parser.add_argument("csv_file", help="Path to the telemetry log CSV file")
    args = parser.parse_args()

    print(f"Loading data from {args.csv_file}...")
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # Create a figure with 4 subplots (IMUs, Encoders, Commands)
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Robot Telemetry Post-Mortem: {args.csv_file}", fontsize=16)

    t = df["timestamp_s"]

    # --- Plot 1: Front IMU Quaternions ---
    axs[0].plot(t, df["front_qr"], label="qr", alpha=0.8)
    axs[0].plot(t, df["front_qi"], label="qi", alpha=0.8)
    axs[0].plot(t, df["front_qj"], label="qj", alpha=0.8)
    axs[0].plot(t, df["front_qk"], label="qk", alpha=0.8)
    axs[0].set_ylabel("Front IMU (Quat)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, linestyle="--", alpha=0.6)

    # --- Plot 2: Back IMU Quaternions ---
    axs[1].plot(t, df["back_qr"], label="qr", alpha=0.8)
    axs[1].plot(t, df["back_qi"], label="qi", alpha=0.8)
    axs[1].plot(t, df["back_qj"], label="qj", alpha=0.8)
    axs[1].plot(t, df["back_qk"], label="qk", alpha=0.8)
    axs[1].set_ylabel("Back IMU (Quat)")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle="--", alpha=0.6)

    # --- Plot 3: Motor Positions (Radians) ---
    axs[2].plot(t, df["front_m1_rad"], label="Front M1", color='blue')
    axs[2].plot(t, df["front_m2_rad"], label="Front M2", color='cyan')
    axs[2].plot(t, df["back_m1_rad"], label="Back M1", color='red')
    axs[2].plot(t, df["back_m2_rad"], label="Back M2", color='orange')
    axs[2].set_ylabel("Position (Rads)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, linestyle="--", alpha=0.6)

    # --- Plot 4: Motor Commands (PWM -1023 to 1023) ---
    axs[3].plot(t, df["front_m1_cmd"], label="Front M1 Cmd", color='blue', alpha=0.7)
    axs[3].plot(t, df["front_m2_cmd"], label="Front M2 Cmd", color='cyan', alpha=0.7)
    axs[3].plot(t, df["back_m1_cmd"], label="Back M1 Cmd", color='red', alpha=0.7)
    axs[3].plot(t, df["back_m2_cmd"], label="Back M2 Cmd", color='orange', alpha=0.7)
    axs[3].set_ylabel("Command (PWM)")
    axs[3].set_xlabel("Time (Seconds)")
    axs[3].legend(loc="upper right")
    axs[3].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()