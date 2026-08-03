"""Compatibility entry point -- delegates to mammocxr.main after the src-layout
migration. Equivalent to `uv run mammocxr`; kept so `uv run python main.py`
keeps working."""

from mammocxr.main import main

if __name__ == "__main__":
    main()
