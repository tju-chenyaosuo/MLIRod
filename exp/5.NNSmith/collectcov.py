import os
import argparse

# Usage: python collectcov.py --project /data/exp/2023.11.19.final.exp/src/mlirsmith

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, help="path of mlir project.", default=10)
args = parser.parse_args()

assert args.project is not None

gcda_list = "gcda-list"
os.system(f"find {args.project} -name *.gcda > {gcda_list}")

with open(gcda_list) as f:
    for l in f.readlines():
        if l.find("mlir") != -1:
            cmd = f"gcov -p {l}"
            print(cmd)
            os.system(cmd)
