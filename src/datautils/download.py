# Copyright (c) 2020-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Download data for calculating question overlap."""
import os
from pathlib import Path

import wget

BASE = Path("data")
DIRS = ["nq", "triviaqa", "webquestions"]

TEST_SETS_TO_DOWNLOAD = [
    ("https://dl.fbaipublicfiles.com/qaoverlap/data/nq-test.qa.csv", DIRS[0], "nq-test.qa.csv"),
    ("https://dl.fbaipublicfiles.com/qaoverlap/data/triviaqa-test.qa.csv", DIRS[1], "triviaqa-test.qa.csv"),
    ("https://dl.fbaipublicfiles.com/qaoverlap/data/webquestions-test.qa.csv", DIRS[2], "webquestions-test.qa.csv"),
]
ANNOTATIONS_TO_DOWNLOAD = [
    ("https://dl.fbaipublicfiles.com/qaoverlap/data/nq-annotations.jsonl", DIRS[0],"nq-annotations.jsonl"),
    ("https://dl.fbaipublicfiles.com/qaoverlap/data/triviaqa-annotations.jsonl", DIRS[1], "triviaqa-annotations.jsonl"),
    ("https://dl.fbaipublicfiles.com/qaoverlap/data/webquestions-annotations.jsonl", DIRS[2], "webquestions-annotations.jsonl")
]

os.makedirs("data/", exist_ok=True)
for link, subdir, dest in TEST_SETS_TO_DOWNLOAD:
    os.makedirs("data/" + subdir, exist_ok=True)
    wget.download(link, str(BASE / subdir / dest))

for link, subdir, dest in ANNOTATIONS_TO_DOWNLOAD:
    wget.download(link, str(BASE / subdir / dest))
