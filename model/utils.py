import sys
import os
import contextlib

class CustomLogger:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.file = file

    def write(self, message):
        if message == "\n":
            return
        if "Epoch" in message:
            message += "\n"
            self.terminal.write(message)
        else:
            self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

@contextlib.contextmanager
def custom_logging(output_dir):
    """Context manager for custom logging"""
    os.makedirs(output_dir, exist_ok=True)
    original_stdout = sys.stdout
    log_file = open(f"{output_dir}/model_output.txt", "w")
    try:
        sys.stdout = CustomLogger(log_file)
        yield
    finally:
        sys.stdout = original_stdout
        log_file.close()

def load_partial_state_dict(model, state_dict):
    """
    Loads weights from state_dict into model, SKIPPING layers with shape mismatches.
    This allows loading the 'backbone' from the Tax model while ignoring the 'projector'.
    """
    model_dict = model.state_dict()

    # Filter out unnecessary or mismatched keys
    pretrained_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
            else:
                print(f"Skipping layer {k}: Shape mismatch {v.shape} vs {model_dict[k].shape}")
        else:
            print(f"Skipping layer {k}: Not in new model")

    # Overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # Load the new state dict
    model.load_state_dict(model_dict)
    print(f"Successfully transferred {len(pretrained_dict)} layers.")