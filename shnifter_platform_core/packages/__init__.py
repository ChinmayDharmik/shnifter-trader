# Generated: 2025-07-04T09:50:39.402105
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Shnifter Engine."""

# flake8: noqa

from pathlib import Path
from typing import List, Optional, Union

from shnifter_core.app.static.app_factory import (
    BaseApp as _BaseApp,
    create_app as _create_app,
)
from shnifter_core.app.static.package_builder import PackageBuilder as _PackageBuilder
from shnifter_core.app.static.reference_loader import ReferenceLoader as _ReferenceLoader

_this_dir = Path(__file__).parent.resolve()


def build(
    modules: Optional[Union[str, List[str]]] = None,
    lint: bool = True,
    verbose: bool = False,
) -> None:
    """Build extension modules.

    Parameters
    ----------
    modules : Optional[List[str]], optional
        The modules to rebuild, by default None
        For example: "/news" or ["/news", "/crypto"]
        If None, all modules are rebuilt.
    lint : bool, optional
        Whether to lint the code, by default True
    verbose : bool, optional
        Enable/disable verbose mode
    """
    _PackageBuilder(_this_dir, lint, verbose).build(modules)


_PackageBuilder(_this_dir).auto_build()
_ReferenceLoader(_this_dir)

try:
    # pylint: disable=import-outside-toplevel
    from shnifter.package.__extensions__ import Extensions as _Extensions

    shnifter: Union[_BaseApp, _Extensions] = _create_app(_Extensions)  # type: ignore
    sdk = shnifter
except (ImportError, ModuleNotFoundError):
    print("Failed to import extensions.")
    shnifter = sdk = _create_app()  # type: ignore
