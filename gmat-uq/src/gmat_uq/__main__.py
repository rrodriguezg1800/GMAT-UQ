# Copyright 2026 GMAT-UQ contributors
# Licensed under the Apache License, Version 2.0 (see License.txt in the repository root).
"""Allow ``python -m gmat_uq`` as an alternative to the ``gmat-uq`` command."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
