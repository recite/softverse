"""Put `scripts/` on the import path, as running a script directly does.

The scripts import one another by bare name. This lives here rather than in
pytest's `pythonpath` setting because the fleet's wheel test clears that
setting (`-o pythonpath=`) so the package cannot be imported from the
checkout; a conftest is loaded either way, and `scripts/` holds no package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
