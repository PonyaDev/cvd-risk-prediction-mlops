"""Basic smoke tests for the project entry point."""

from src.main import main


def test_main_returns_success_exit_code() -> None:
    assert main() == 0
