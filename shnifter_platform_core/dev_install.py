# Generated: 2025-07-04T09:50:39.399407
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Install for development script."""

# flake8: noqa: S603

import subprocess
import sys
from pathlib import Path

from tomlkit import dumps, load, loads

PLATFORM_PATH = Path(__file__).parent.resolve()
LOCK = PLATFORM_PATH / "poetry.lock"
PYPROJECT = PLATFORM_PATH / "pyproject.toml"
Interface_PATH = Path(__file__).parent.parent.resolve() / "interface"
Interface_PYPROJECT = Interface_PATH / "pyproject.toml"
Interface_LOCK = Interface_PATH / "poetry.lock"

LOCAL_DEPS = """
[tool.poetry.dependencies]
python = ">=3.9.21,<3.13"
shnifter-devtools = { path = "./extensions/devtools", develop = true, markers = "python_version >= '3.10'" }
shnifter-core = { path = "./core", develop = true }
shnifter-engine-api = { path = "./extensions/engine_api", develop = true }

shnifter-benzinga = { path = "./providers/benzinga", develop = true }
shnifter-bls = { path = "./providers/bls", develop = true }
shnifter-cftc = { path = "./providers/cftc", develop = true }
shnifter-econdb = { path = "./providers/econdb", develop = true }
shnifter-federal-reserve = { path = "./providers/federal_reserve", develop = true }
shnifter-fmp = { path = "./providers/fmp", develop = true }
shnifter-fred = { path = "./providers/fred", develop = true }
shnifter-imf = { path = "./providers/imf", develop = true }
shnifter-intrinio = { path = "./providers/intrinio", develop = true }
shnifter-oecd = { path = "./providers/oecd", develop = true }
shnifter-polygon = { path = "./providers/polygon", develop = true }
shnifter-sec = { path = "./providers/sec", develop = true }
shnifter-tiingo = { path = "./providers/tiingo", develop = true }
shnifter-tradingeconomics = { path = "./providers/tradingeconomics", develop = true }
shnifter-us-eia = { path = "./providers/eia", develop = true }
shnifter-yfinance = { path = "./providers/yfinance", develop = true }

shnifter-commodity = { path = "./extensions/commodity", develop = true }
shnifter-crypto = { path = "./extensions/crypto", develop = true }
shnifter-currency = { path = "./extensions/currency", develop = true }
shnifter-derivatives = { path = "./extensions/derivatives", develop = true }
shnifter-economy = { path = "./extensions/economy", develop = true }
shnifter-equity = { path = "./extensions/equity", develop = true }
shnifter-etf = { path = "./extensions/etf", develop = true }
shnifter-fixedincome = { path = "./extensions/fixedincome", develop = true }
shnifter-index = { path = "./extensions/index", develop = true }
shnifter-news = { path = "./extensions/news", develop = true }
shnifter-regulators = { path = "./extensions/regulators", develop = true }
shnifter-mcp-server = { path = "./extensions/mcp_server", develop = true, markers = "python_version >= '3.10'" }

# Community dependencies
shnifter-alpha-vantage = { path = "./providers/alpha_vantage", optional = true, develop = true }
shnifter-biztoc = { path = "./providers/biztoc", optional = true, develop = true }
shnifter-cboe = { path = "./providers/cboe", optional = true, develop = true }
shnifter-deribit = { path = "./providers/deribit", optional = true, develop = true }
shnifter-ecb = { path = "./providers/ecb", optional = true, develop = true }
shnifter-finra = { path = "./providers/finra", optional = true, develop = true }
shnifter-finviz = { path = "./providers/finviz", optional = true, develop = true }
shnifter-government-us = { path = "./providers/government_us", optional = true, develop = true }
shnifter-multpl = { path = "./providers/multpl", optional = true, develop = true }
shnifter-nasdaq = { path = "./providers/nasdaq", optional = true, develop = true }
shnifter-seeking-alpha = { path = "./providers/seeking_alpha", optional = true, develop = true }
shnifter-stockgrid = { path = "./providers/stockgrid" , optional = true,  develop = true }
shnifter_tmx = { path = "./providers/tmx", optional = true, develop = true }
shnifter_tradier = { path = "./providers/tradier", optional = true, develop = true }
shnifter-wsj = { path = "./providers/wsj", optional = true, develop = true }

shnifter-charting = { path = "./shnifterject_extensions/charting", optional = true, develop = true }
shnifter-econometrics = { path = "./extensions/econometrics", optional = true, develop = true }
shnifter-quantitative = { path = "./extensions/quantitative", optional = true, develop = true }
shnifter-technical = { path = "./extensions/technical", optional = true, develop = true }
"""


def extract_dependencies(local_dep_path, dev: bool = False):
    """Extract development dependencies from a given package's pyproject.toml."""
    package_pyproject_path = PLATFORM_PATH / local_dep_path
    if package_pyproject_path.exists():
        with open(package_pyproject_path / "pyproject.toml") as f:
            package_pyproject_toml = load(f)
        if dev:
            return (
                package_pyproject_toml.get("tool", {})
                .get("poetry", {})
                .get("group", {})
                .get("dev", {})
                .get("dependencies", {})
            )
        return (
            package_pyproject_toml.get("tool", {})
            .get("poetry", {})
            .get("dependencies", {})
        )
    return {}


def get_all_dev_dependencies():
    """Aggregate development dependencies from all local packages."""
    all_dev_dependencies = {}
    local_deps = loads(LOCAL_DEPS).get("tool", {}).get("poetry", {})["dependencies"]
    for _, package_info in local_deps.items():
        if "path" in package_info:
            dev_deps = extract_dependencies(Path(package_info["path"]), dev=True)
            all_dev_dependencies.update(dev_deps)
    return all_dev_dependencies


def install_engine_local(_extras: bool = False):
    """Install the Engine locally for development purposes."""
    original_lock = LOCK.read_text()
    original_pyproject = PYPROJECT.read_text()

    local_deps = loads(LOCAL_DEPS).get("tool", {}).get("poetry", {})["dependencies"]
    with open(PYPROJECT) as f:
        pyproject_toml = load(f)
    pyproject_toml.get("tool", {}).get("poetry", {}).get("dependencies", {}).update(
        local_deps
    )

    # Extract and add devtools dependencies manually if Python version is 3.9
    if sys.version_info[:2] == (3, 9):
        devtools_deps = extract_dependencies(Path("./extensions/devtools"), dev=False)
        devtools_deps.remove("python")
        pyproject_toml.get("tool", {}).get("poetry", {}).get("dependencies", {}).update(
            devtools_deps
        )

    if _extras:
        dev_dependencies = get_all_dev_dependencies()
        pyproject_toml.get("tool", {}).get("poetry", {}).setdefault(
            "group", {}
        ).setdefault("dev", {}).setdefault("dependencies", {})
        pyproject_toml.get("tool", {}).get("poetry", {})["group"]["dev"][
            "dependencies"
        ].update(dev_dependencies)

    TEMP_PYPROJECT = dumps(pyproject_toml)

    try:
        with open(PYPROJECT, "w", encoding="utf-8", newline="\n") as f:
            f.write(TEMP_PYPROJECT)

        CMD = [sys.executable, "-m", "poetry"]
        extras_args = ["-E", "all"] if _extras else []

        subprocess.run(
            CMD + ["lock"],
            cwd=PLATFORM_PATH,
            check=True,
        )
        subprocess.run(
            CMD + ["install"] + extras_args,
            cwd=PLATFORM_PATH,
            check=True,
        )

    except (Exception, KeyboardInterrupt) as e:
        print(e)  # noqa: T201
        print("Restoring pyproject.toml and poetry.lock")  # noqa: T201

    finally:
        # Revert pyproject.toml and poetry.lock to their original state.
        with open(PYPROJECT, "w", encoding="utf-8", newline="\n") as f:
            f.write(original_pyproject)

        with open(LOCK, "w", encoding="utf-8", newline="\n") as f:
            f.write(original_lock)


def install_engine_interface():
    """Install the Interface locally for development purposes."""
    original_lock = Interface_LOCK.read_text()
    original_pyproject = Interface_PYPROJECT.read_text()

    with open(Interface_PYPROJECT) as f:
        pyproject_toml = load(f)

    # remove "shnifter" from dependencies
    pyproject_toml.get("tool", {}).get("poetry", {}).get("dependencies", {}).pop(
        "shnifter", None
    )

    TEMP_PYPROJECT = dumps(pyproject_toml)

    try:
        with open(Interface_PYPROJECT, "w", encoding="utf-8", newline="\n") as f:
            f.write(TEMP_PYPROJECT)

        CMD = [sys.executable, "-m", "poetry"]

        subprocess.run(
            CMD + ["lock"],
            cwd=Interface_PATH,
            check=True,  # noqa: S603
        )
        subprocess.run(CMD + ["install"], cwd=Interface_PATH, check=True)  # noqa: S603

    except (Exception, KeyboardInterrupt) as e:
        print(e)  # noqa: T201
        print("Restoring pyproject.toml and poetry.lock")  # noqa: T201

    finally:
        # Revert pyproject.toml and poetry.lock to their original state.
        with open(Interface_PYPROJECT, "w", encoding="utf-8", newline="\n") as f:
            f.write(original_pyproject)

        with open(Interface_LOCK, "w", encoding="utf-8", newline="\n") as f:
            f.write(original_lock)


if __name__ == "__main__":
    args = sys.argv[1:]
    extras = any(arg.lower() in ["-e", "--extras"] for arg in args)
    interface = any(arg.lower() in ["-c", "--interface"] for arg in args)
    install_engine_local(extras)
    if interface:
        install_engine_interface()
