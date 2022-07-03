from pathlib import Path
from typing import Optional


def format_to_md(s: str) -> str:
    lines = s.split("\n")
    return "    " + "\n    ".join(lines)


def get_log_file(log_dir: Optional[str], file_name: Optional[str]) -> Optional[str]:
    if log_dir is not None:
        if file_name is None:
            raise Exception("Please pass a non-null name to get_log_file")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(log_dir, file_name + ".pt"))
    return None
