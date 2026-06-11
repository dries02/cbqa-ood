from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from src.train.trainconfig import MODEL_CONFIGS


def load_tokenizer(dataset: str, model_type: str, fraction: float, suffix: str) -> PreTrainedTokenizerBase:
    tokenizer_path = Path("models") / f"{dataset}-{model_type}-mcdropout-{suffix}-{fraction}-0"
    return AutoTokenizer.from_pretrained(tokenizer_path)


def load_stochastic_model(method_type: str, model_path: Path, device: torch.device, model_type: str) -> PreTrainedModel:
    if method_type == "mcdropout":
        return AutoModelForSeq2SeqLM.from_pretrained(model_path).train().to(device)
    if method_type == "flipout":
        model = MODEL_CONFIGS[model_type]["flipout_model"].from_pretrained(model_path).eval().to(device)
        model.lm_head.train()
        return model
    msg = f"Unknown model type: {method_type}"
    raise ValueError(msg)


def load_ensemble(dataset: str, model_type: str, suffix: str,
                  n_ensemble: int, fraction: float, device: torch.device,
                  ) -> list[PreTrainedModel]:
    model_paths = [
        Path("models") / f"{dataset}-{model_type}-mcdropout-{suffix}-{fraction}-{i}"
        for i in range(n_ensemble)
    ]

    for path in model_paths:
        if not path.exists():
            msg = f"Model not found: {path}"
            raise FileNotFoundError(msg)

    print(f"Loading {n_ensemble} models...")
    models = [
        AutoModelForSeq2SeqLM.from_pretrained(path).eval().to(device)
        for path in model_paths
    ]

    print(f"Models loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    return models
