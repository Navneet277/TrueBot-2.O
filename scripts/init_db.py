"""Helper script to initialize the SQLite database."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import init_db  # noqa: E402


def main() -> None:
    init_db()
    print("Database initialized successfully.")


if __name__ == "__main__":
    main()

