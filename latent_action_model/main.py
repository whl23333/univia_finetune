from lightning.pytorch.cli import LightningCLI
from genie.dataset import LightningOpenX
from genie.model import DINO_LAM, DINO_LAM_MultiView

cli = LightningCLI(
    DINO_LAM_MultiView,
    LightningOpenX,
    seed_everything_default=42,
)
