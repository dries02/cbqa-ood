from dataclasses import dataclass, field
from pathlib import Path

MODEL_CONFIGS = {
    "bart-large": {
        "hf_name": "facebook/bart-large",
        "prefix": "",
        "remove_bos": True,
    },
    "flan-t5-large": {
        "hf_name": "google/flan-t5-large",
        "prefix": "answer briefly: ",
        "remove_bos": False,
    },
    "t5-large-ssm": {
        "hf_name": "google/t5-large-ssm",
        "prefix": "question: ",
        "remove_bos": False,
    },
    "t5-3b-ssm": {
        "hf_name": "google/t5-3b-ssm",
        "prefix": "question: ",
        "remove_bos": False,
    },
}


@dataclass
class TrainConfig:
    """Configuration for model training."""

    n_epochs: int
    batch_size: int
    lr: float
    patience: int
    dropout: float
    dataset: str
    method: str
    model: str
    rho: float = -2.5
    use_soft_labels: bool = True
    use_stochastic_labels: bool = False
    train_path: Path = field(init=False)
    dev_path: Path = field(init=False)
    output_dir: Path = field(init=False)
    hf_name: str = field(init=False)
    prefix: str = field(init=False)
    remove_bos: bool = field(init=False)


    def __post_init__(self) -> None:
        """Set some directories."""
        self.output_dir = Path("models") / f"{self.dataset}-{self.model}-{self.method}"
        self.train_path = Path("data") / self.dataset / f"{self.dataset}-train.jsonl"
        self.dev_path = Path("data") / self.dataset / f"{self.dataset}-dev.jsonl"
        self.hf_name = MODEL_CONFIGS[self.model]["hf_name"]
        self.prefix = MODEL_CONFIGS[self.model]["prefix"]
        self.remove_bos = MODEL_CONFIGS[self.model]["remove_bos"]
