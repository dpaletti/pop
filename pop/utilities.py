def format_to_md(s: str):
    lines = s.split("\n")
    return "    " + "\n    ".join(lines)
