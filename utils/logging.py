from __future__ import annotations

import sys


class CustomPrint:
    def __init__(self, file_path: str):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        sys.stdout = self.terminal
        self.log.close()


def log_info(msg: str) -> None:
    print(msg, flush=True)
