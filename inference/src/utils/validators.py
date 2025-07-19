from pathlib import Path


def validate_input_path(path: Path) -> bool:
    """Validate the input directory path"""
    if not path.exists():
        print(f"Error: Directory '{path}' does not exist")
        return False

    if not path.is_dir():
        print(f"Error: '{path}' is not a directory")
        return False

    # Check if directory contains any subdirectories
    if not any(p.is_dir() for p in path.iterdir()):
        print(f"Error: No series directories found in '{path}'")
        return False

    return True
