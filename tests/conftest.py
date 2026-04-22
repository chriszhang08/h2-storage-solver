from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--series-path",
        action="store",
        default=str(Path(__file__).parent / "data" / "arma_garch_test_series.txt"),
        help="Path to a plain-text file (one price per line) used by test_arma_garch_regressor.",
    )
