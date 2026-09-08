"""Put the repo root on sys.path so `pytest` (collecting from tests/) can
import the top-level packages (features/, inference/, models/, ...)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
