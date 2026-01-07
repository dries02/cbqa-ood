# TODO

## Data

- [ ] missing data Lewis et al (discuss)
- [ ] labels might be misleading since only one answer is used for training. see Roberts et al., they used first answer

## 🔧

- [ ] MC Dropout only on output layer (similar to Flipout) (?)
- [ ] cleanup some old files (`f1_rms.py`, `sbertdemo.py`)
- [ ] For F1: use tokens rather than words?

## 🏃

- [ ] different pretrained models like T5, not just BART-large
- [ ] train flipout on nq
- [ ] train mcdropout and flipout on triviaqa?

## ☕

- [ ] flipout for entire model too expensive
- [ ] better justification for ELBO scaling
- [ ] more UQ methods, see literature
- [ ] analyze relationship uncertainty and performance
- [ ] prior or warm start posterior in Flipout
- [ ] deep ensembles: how to ensure enough diversity?
- [ ] determine reasonable max len for tokenizing questions and answers based on training set
- [ ] finetuning and better model training documentation/experiment tracking
- [ ] How does Jensen divergence relate to MI?
- [ ] Any thoughts on soft label setting?

## Error analysis

- [ ] included: AUROC for OOD detection
- [ ] in progress: proper analysis of histograms (distribution of uncertainty scores per dataset, method, id/od split)
- [ ] selective prediction (if i answer top K% most confident predictions, what are my metrics?)
- [ ] performance and uncertainty correlation?
- [ ] performance (better with 'ensemble'?)
- [ ] FLOPs?
