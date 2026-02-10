from dataclasses import dataclass, field
from pathlib import Path

from src.train.flipoutseq2seqbart import FlipoutSeq2SeqBart
from src.train.flipoutseq2seqbase import FlipoutSeq2SeqBase
from src.train.flipoutseq2seqt5 import FlipoutSeq2SeqT5

MODEL_CONFIGS = {
    "bart-large": {
        "hf_name": "facebook/bart-large",
        "prefix": "",
        "remove_bos": True,
        "flipout_model": FlipoutSeq2SeqBart,
    },
    "t5-large-ssm": {
        "hf_name": "google/t5-large-ssm",
        "prefix": "question: ",
        "remove_bos": False,
        "flipout_model": FlipoutSeq2SeqT5,
    },
}


@dataclass
class TrainConfig:
    """Configuration for model training."""

    n_epochs: int
    batch_size: int
    lr: float
    # patience: int
    dropout: float
    dataset: str
    method: str
    model: str
    use_soft_labels: bool
    use_stochastic_labels: bool
    rho: float
    train_path: Path = field(init=False)
    dev_path: Path = field(init=False)
    output_dir: Path = field(init=False)
    hf_name: str = field(init=False)
    prefix: str = field(init=False)
    remove_bos: bool = field(init=False)
    flipout_model: FlipoutSeq2SeqBase = field(init=False)


    def __post_init__(self) -> None:
        """Set some directories."""
        suffix = "soft" if self.use_soft_labels else "hard"
        self.output_dir = Path("models") / f"{self.dataset}-{self.model}-{self.method}-{suffix}-0"
        self.train_path = Path("data") / self.dataset / f"{self.dataset}-train.jsonl"
        self.dev_path = Path("data") / self.dataset / f"{self.dataset}-dev.jsonl"
        for key in MODEL_CONFIGS[self.model]:
            setattr(self, key, MODEL_CONFIGS[self.model][key])
