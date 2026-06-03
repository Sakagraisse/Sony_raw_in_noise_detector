#!/usr/bin/env python3
"""Generate ISO stress overlays using direct marker detection only."""

from stress_test_v4_3_iso_detection import DIRECT_ONLY_OUTPUT_DIR, run_stress


def main() -> None:
    run_stress(DIRECT_ONLY_OUTPUT_DIR, propagate_from_iso100=False)


if __name__ == "__main__":
    main()
