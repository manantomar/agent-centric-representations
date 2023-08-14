# Usage 

The code contains implementations of two algorithms

1. [ACRO](https://arxiv.org/abs/2211.00164) - learns agent centric representations with a multi-step inverse dynamics model.
2. [InfoGating](https://arxiv.org/pdf/2303.06121.pdf) - adds an InfoGating bottleneck over a contrastive variant of the multi-step inverse dynamics model.

# Installation

The provided `requirements.txt` file contains all main dependencies required to run this code. You will also need to download the offline datasets from [v-d4rl paper](https://drive.google.com/drive/folders/15HpW6nlJexJP5A4ygGk-1plqt9XdcWGI) or (depending on which kinds of distractors you want to test with) from the [ACRO paper](https://drive.google.com/drive/folders/1HsksquQ6gKQUDj_1Qe7dc-R_h8m6A1H3) and have them in `./vd4rl` path.

# Running

Simply use the train.py script to run ACRO/InfoGating. Provide a `task_name`, `offline_dir` directory where the offline code is available and which `algo` to run. The `dist_level` argument is only used for naming result files.

Run ACRO
```
python train.py task_name=offline_cheetah_run_expert offline_dir=/path/to/dataset/vd4rl/main/cheetah_run/expert/ seed=1 algo=acro dist_level=none
```

Run InfoGating
```
python train.py task_name=offline_cheetah_run_expert offline_dir=/path/to/dataset/vd4rl/main/cheetah_run/expert/ seed=1 algo=infogating dist_level=none
```