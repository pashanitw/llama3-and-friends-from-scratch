import time
from pathlib import Path
from typing import Optional, Mapping

class TrainingLogger:
    def __init__(self, log_dir: str, filename: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True) # what is this
        if not filename:
            filename = f"log_{int(time.time())}.log"

        self.file_path = self.log_dir / filename

        #open the log file

        self._log_file = open(self.file_path, "a")

        print(f"Writing logs to {self.file_path}")

    def log(self, name:str, data:float, step: int):
        self._log_file.write(f"Step {step} | {name}: {data}\n")
        self._log_file.flush()

    def log_dict(self, payload:Mapping[str, float], step:int):
        entries = " | ".join(f"{name}: {data}" for name, data in payload.items())
        self._log_file.write(f"Step {step} | {entries}\n")
        self._log_file.flush()

    def close(self):
        self._log_file.close()

    def __del__(self):
        self.close()


# Example usage
if __name__ == "__main__":
    log_directory = "logs"  # Specify your log directory
    logger = TrainingLogger(log_directory)  # Create an instance of DiskLogger

    # Log individual metrics
    logger.log("accuracy", 0.95, 1)
    logger.log("loss", 0.05, 1)

    # Log multiple metrics at once
    logger.log_dict({"learning_rate": 0.001, "batch_size": 32}, 2)

    # Close the logger when done
    logger.close()
