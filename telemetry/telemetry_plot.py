import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Read the data into a Pandas DataFrame
df = pd.read_csv("/Users/itak/Documents/CMU/24775_Bioinspired_Robot/MeowMentum/telemetry/telemetry_1776463636.csv")

# Create the plot
plt.figure(figsize=(10, 6))

# cols: Time,F_Q0,F_Q1,F_Q2,F_Q3,F_M1,F_M2,Cmd_F1,Cmd_F2,B_Q0,B_Q1,B_Q2,B_Q3,B_M1,B_M2,Cmd_B1,Cmd_B2

# Convert quaternion to rotation matrix
# front_quat = df[['F_Q0', 'F_Q1', 'F_Q2', 'F_Q3']].values
# front_rot = R.from_quat(front_quat).as_matrix().reshape(-1, 9)
# back_quat = df[['B_Q0', 'B_Q1', 'B_Q2', 'B_Q3']].values
# back_rot = R.from_quat(back_quat).as_matrix().reshape(-1, 9)



# Plot rotation matrices
# plt.subplot(2, 2, 1)
# plt.plot(df['Time'], front_rot, label='Front')
# plt.plot(df['Time'], back_rot, label='Back')

plt.plot(df['F_M1'], label='f1')
plt.plot(df['Cmd_F1'], label='c1')
plt.plot(df['F_M2'], label='f2')
plt.plot(df['Cmd_F2'], label='c2')

# Loop through all columns except 'Time' to plot them
# for col in df.columns:
#     if col != 'Time':
#         plt.plot(df['Time'], df[col], label=col)

# Add labels and styling
plt.xlabel('Time (s)')
plt.ylabel('Values')
plt.title('Cat Robot Kinematics/Commands over Time')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()