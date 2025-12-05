# Package management
uv: ```curl -LsSf https://astral.sh/uv/install.sh | sh```

jq: ```sudo apt-get install jq```

# Datasets
We use three data sets for closed-book question answering, as used by Lewis et al. (2020).

See [Github facebookresearch](https://github.com/facebookresearch/QA-Overlap/blob/main/download.py) for test sets and annotations.

## WebQuestions
| Split | Link | Samples |
|-------|------|------|
| Train | - | -
| Dev | - | -
| Test | [download](https://dl.fbaipublicfiles.com/qaoverlap/data/webquestions-test.qa.csv) | 2,032
| Annotations | [download](https://dl.fbaipublicfiles.com/qaoverlap/data/webquestions-annotations.jsonl) | 2,032

> **Remark**: It is not so clear what the training and dev set is. Lewis et al. (2020): *"We use the development split used in Karpukhin et al. (2020), which was randomly split from the train set."*. It is clear that the entire training set has 3778 samples.

> Then in Karpukhin et al. (2020), Table 1, there is a training set with 3417 samples and a 'filtered' training set with 2474 samples. Then the dev set has 361 samples.

> But when checking [their repo](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download), we get the filtered training set with 2474 samples and a dev set with 278 samples (???)...

> What did Lewis et al. use? The training set with 3417 samples or with 2474 samples? And where is the dev set with 361 samples, why does the repo have a dev set with 278 samples? Are there other papers with a better link?

See https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py

I can see in https://discovery.ucl.ac.uk/id/eprint/10151750/2/Patrick_Lewis_Thesis__for_submission_to_UCL_.pdf that he used 3417, 361 by Lewis, suggesting no filtering.

- https://github.com/facebookresearch/DPR/issues/143 shows that the 278 dev set is filtered. the original 361 seems hard to find

Rest to 1940 samples

## TriviaQA
| Split | Link | Samples |
|-------|------|------|
| Train | [view](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/viewer/triviaqa/train) | 78,785
| Dev | [view](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/viewer/triviaqa/dev) | 8,837
| Test | [download](https://dl.fbaipublicfiles.com/qaoverlap/data/triviaqa-test.qa.csv) | 11,313
| Annotations | [download](https://dl.fbaipublicfiles.com/qaoverlap/data/triviaqa-annotations.jsonl) | 11,313

Reduced to 3948 samples

## Natural Questions
| Split | Link | Samples |
|-------|------|------|
| Train | [view](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/viewer/nq/train) | 79,168
| Dev | [view](https://huggingface.co/datasets/RUC-NLPIR/FlashRAG_datasets/viewer/nq/dev) | 8,757
| Test | [download](https://dl.fbaipublicfiles.com/qaoverlap/data/nq-test.qa.csv) | 3,610
| Annotations | [download](https://dl.fbaipublicfiles.com/qaoverlap/data/nq-annotations.jsonl) | 3,610

Reduced to 1428 samples
