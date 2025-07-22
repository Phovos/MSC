import tomllib
import sys
import os


def extract_version(file_path):
    try:
        with open(file_path, 'rb') as f:
            pyproject = tomllib.load(f)
        version = pyproject['project']['version']
        print(version)
    except FileNotFoundError:
        print(f"Error: {file_path} file not found.", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing key in {file_path}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
    else:
        # Default to pyproject.toml in the parent directory if no argument is provided
        file_path = os.path.join(os.path.dirname(
            __file__), '..', 'pyproject.toml')

    extract_version(file_path)
