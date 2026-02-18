import os
from pathlib import Path
from typing import Final

SRC_DIR: Final[Path] = Path(__file__).parent

PROJECT_ROOT: Final = SRC_DIR.parent

print(SRC_DIR)
print(PROJECT_ROOT)