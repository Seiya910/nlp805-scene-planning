# NLP805 Assignment 2

English paragraph-level news framing for SemEval-2027 Task 9, Subtask 1.
The experiment uses 1,301 paragraphs from 151 articles and splits them by
article. The code was prepared with AI assistance.

| Model | Validation micro F1 | Test micro F1 | Test macro F1 |
| --- | ---: | ---: | ---: |
| Word TF-IDF | 0.255 | 0.233 | 0.123 |
| Word + character TF-IDF | 0.308 | 0.290 | 0.193 |
| Word + character, balanced | 0.306 | 0.330 | 0.260 |

The word + character model was selected on validation micro F1. Test scores
are from the official scorer on the local 230-paragraph split.

Download the official training release and starter kit from the
[competition page](https://artist.math.unipd.it/competitions/4/).
Run these commands from `assignment_02`, using Python 3.12 and the paths to
your downloaded archives.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

python prepare_official.py --data-zip "$HOME/Downloads/data-second-batch.zip" \
  --starter-zip "$HOME/Downloads/starter-kit.zip" --out work/official
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python framing.py train --articles work/official/articles \
  --annotations work/official/annotations.tsv --labels work/official/labels.txt \
  --official-scorer work/official/scorer-subtask-1.py \
  --data-kind designated-training --seed 42 --out runs/official
python verify_run.py --run runs/official --data-dir work/official
```
