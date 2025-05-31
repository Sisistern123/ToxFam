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
