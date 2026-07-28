import os
import torch
import torch.nn as nn
import time

# 1. Define your policy/model architecture. 
# This MUST exactly match the structure used during your RL training loop.
class StudentPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
            nn.Tanh() # Binds outputs to [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

# --- Configuration ---
# Student obs = N_FRAMES stacked frames of [front_proj_grav(3) + joint_angles(4)].
# Must match cat_env/cat_env.py + distillation.py (STUDENT_OBS_DIM = N_FRAMES * 7).
N_FRAMES = 4
FRAME_DIM = 3 + 4
STATE_DIM = N_FRAMES * FRAME_DIM      # 28
ACTION_DIM = 3
PTH_FILE = "student_policy.pth"
ONNX_FILE = f"cat_controller.onnx"

def main():
    # 2. Instantiate the model and load the trained weights
    model = StudentPolicy(STATE_DIM, ACTION_DIM)
    
    # If you saved the full model (torch.save(model)), you would just load it directly.
    # If you saved the state_dict (recommended), load it like this:
    try:
        model.load_state_dict(torch.load(PTH_FILE, map_location=torch.device('cpu')))
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Ensure you are loading a state_dict and the architecture matches.")
        return

    # 3. Set the model to evaluation mode
    # Crucial for freezing layers like BatchNorm and Dropout so they behave deterministically
    model.eval()

    # 4. Create a dummy input tensor
    # This acts as a probe to trace the computational graph. 
    # Shape here is (Batch_Size=1, State_Dimension)
    dummy_input = torch.randn(1, STATE_DIM, requires_grad=True)

    # 5. Export to ONNX
    print(f"Exporting model to {ONNX_FILE} as opset 14...")
    torch.onnx.export(
        model,                      
        dummy_input,                
        ONNX_FILE,                  
        export_params=True,         
        opset_version=14,           # CHANGED: Export to 14 to bypass PyTorch's broken downgrade
        do_constant_folding=True,   
        input_names=['state'],      
        output_names=['action'],    
        dynamic_axes={
            'state': {0: 'batch_size'},    
            'action': {0: 'batch_size'}
        }
    )
    print("PyTorch export complete. Now patching opset metadata...")

    # 6. Manually patch the Opset version down to 11
    import onnx
    
    # Load the newly exported model (pulling in any external tensor data)
    onnx_model = onnx.load(ONNX_FILE, load_external_data=True)

    # Force the metadata to say opset 11
    for imp in onnx_model.opset_import:
        if imp.domain == "" or imp.domain == "ai.onnx":
            imp.version = 11

    # Overwrite as a SINGLE self-contained file (no .onnx.data sidecar) so the Pi
    # only needs to copy one file.
    onnx.save(onnx_model, ONNX_FILE, save_as_external_data=False)
    sidecar = ONNX_FILE + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)

    print(f"Success! {ONNX_FILE} has been safely patched to Opset 11 (single file).")

if __name__ == "__main__":
    main()