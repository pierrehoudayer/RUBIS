import subprocess
import sys


def test_core_api_does_not_require_matplotlib():
    code = """
import sys

sys.modules["matplotlib"] = None

import rubis
from rubis.api import deform
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr