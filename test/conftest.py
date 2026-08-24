"""
pytest configuration for the CI model tests.

Tests are parametrized over the pretrained models discovered in the
repository. A test requests one of the ``float_spec`` / ``quantized_spec`` /
``akida_spec`` parameters and gets one instance per matching model.

The ``--models`` option restricts the parametrization to an explicit list of
repo-relative model paths (whitespace or comma separated). This is how the
workflows scope PR runs to only the models the PR adds or modifies. Without
``--models``, every discovered model is tested.
"""

from pathlib import Path
from discover_models import discover_all


def pytest_addoption(parser):
    parser.addoption(
        "--models", default="",
        help="Whitespace/comma-separated repo-relative model paths to test. "
             "Empty (default) tests every discovered model.")


def _selected_specs(config, kind):
    specs = [s for s in discover_all() if s.kind == kind]
    raw = config.getoption("--models").replace(",", " ").split()
    if not raw:
        return specs
    wanted = {Path(p).as_posix() for p in raw}
    return [s for s in specs if s.path in wanted]


def pytest_generate_tests(metafunc):
    kind_by_fixture = {"float_spec": "float", "quantized_spec": "quantized", "akida_spec": "akida"}
    for fixture, kind in kind_by_fixture.items():
        if fixture in metafunc.fixturenames:
            specs = _selected_specs(metafunc.config, kind)
            metafunc.parametrize(fixture, specs, ids=[s.path for s in specs])
