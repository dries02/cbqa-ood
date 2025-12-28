# TODO

## Data

- [ ] missing data Lewis et al (discuss)
- [ ] labels might be misleading since only one answer is used for training. see Roberts et al., they used first answer

## 🔧

- [ ] MC Dropout only on output layer (similar to Flipout) (?)
- [ ] token-level probability averaging
  - [ ] masking before or after computing MI/divergence?
  - [ ] implement Jensen divergence
- [ ] Multi-label setting model training
- [ ] cleanup some old files (`f1_rms.py`, `sbertdemo.py`)
- [ ] For F1: use tokens rather than words!
- [ ] proper error analysis
- [ ] baselines: log probs, entropy, ...
- [ ] Fix bug wrt tokenization/alignment

## 🏃

- [ ] evaluate on other datasets (TriviaQA and NaturalQuestions)
- [ ] different pretrained models, not just BART-large

## ☕

- [ ] flipout for entire model too expensive
- [ ] better justification for ELBO scaling
- [ ] more UQ methods, see literature
- [ ] analyze relationship uncertainty and performance
- [ ] prior or warm start posterior in Flipout
- [ ] deep ensembles: how to ensure enough diversity?
- [ ] determine reasonable max len for tokenizing questions and answers based on training set
- [ ] finetuning and better model training documentation/experiment tracking
