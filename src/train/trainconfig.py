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
    dataset: str
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        """Set some directories."""
        self.output_dir = Path("models") / (self.dataset + "-large")
