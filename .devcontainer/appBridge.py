import tomllib
import sys


def extract_version():
    with open('pyproject.toml', 'rb') as f:
        pyproject = tomllib.load(f)
    version = pyproject['tool']['poetry']['version']
    print(version)


if __name__ == "__main__":
    extract_version()
