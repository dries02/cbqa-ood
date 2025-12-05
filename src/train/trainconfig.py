from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """
    Configuration for model training.

    Attributes:
        n_epochs (int):
        batch_size (int):
        lr (float):
        dataset (str):
        output_dir (Path | None):
    """

    n_epochs: int
    batch_size: int
    lr: float
    patience: int
    dropout: float
    dataset: str
    method: str
    train_path: Path | None = None
    dev_path: Path | None = None
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        """Set some directories."""
        self.output_dir = Path("models") / f"{self.dataset}-{self.method}-large"
        self.train_path = Path("data") / self.dataset / f"{self.dataset}-train.jsonl"
        self.dev_path = Path("data") / self.dataset / f"{self.dataset}-dev.jsonl"
