import os

VARIANTS = {
    "tail":   {"env_id": "Cat-v0",       "suffix": ""},
    "notail": {"env_id": "CatNoTail-v0", "suffix": "_notail"},
}

# Every trained artifact lives here: teacher .zip, student .pth, exported .onnx.
# Absolute, derived from this file, so a script works the same run from the repo
# root, from docs/, or from hardware/ on the Pi.
POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies")


def policy_dir():
    """POLICY_DIR, created if missing. Call before writing an artifact."""
    os.makedirs(POLICY_DIR, exist_ok=True)
    return POLICY_DIR


def teacher_path(variant):
    """Staged SAC teacher for a variant (policies/cat_controller<suffix>.zip)."""
    return os.path.join(POLICY_DIR, f"cat_controller{VARIANTS[variant]['suffix']}.zip")


def student_path(variant):
    """Staged distilled student (policies/student_policy<suffix>.pth)."""
    return os.path.join(POLICY_DIR, f"student_policy{VARIANTS[variant]['suffix']}.pth")


def onnx_path(variant):
    """Exported student for the robot (policies/cat_controller<suffix>.onnx)."""
    return os.path.join(POLICY_DIR, f"cat_controller{VARIANTS[variant]['suffix']}.onnx")
