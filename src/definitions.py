import os

SRC_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = f"{ROOT_DIR}/data"


if __name__ == "__main__":
    print(f"Source directory: {SRC_DIR}")
    print(f"Root directory: {ROOT_DIR}")
    print(f"Data directory: {DATA_DIR}")