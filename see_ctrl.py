import numpy as np
import matplotlib.pyplot as plt

ctrls = np.load("control.npy")
plt.plot(ctrls)
plt.legend(["target_rot", "target_pitch", "target_tail", "current_rot1", "current_pitch", "current_rot2", "current_tail"])
plt.show()