# Generated: 2025-07-04T09:50:40.212842
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Script to build the Shnifter engine static assets."""

# flake8: noqa: S603
# pylint: disable=import-outside-toplevel,unused-import
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main():
    """Build the Shnifter engine static assets."""
    try:
        logger.info("Attempting to import the Shnifter package...\n")
        # Try importing shnifter in a subprocess and capture output
        result = subprocess.run(
            [sys.executable, "-c", "import shnifter"],
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(result.stdout)
        building_found = any(
            line.startswith("Building") for line in result.stdout.splitlines()
        )

        if result.returncode != 0:
            raise ModuleNotFoundError(result.stderr)

    except (ModuleNotFoundError, subprocess.CalledProcessError):
        logger.info(
            "\nShnifter build script not found, installing from PyPI...\n",
        )

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "shnifter",
                    "--no-deps",
                    "--force-reinstall",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "Failed to install the Shnifter package. "
                "Is pip installed and available in your environment? "
                f"{e.output} \n\n {e.stderr}"
            ) from None

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import shnifter",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(result.stdout)
            building_found = any(
                line.startswith("Building") for line in result.stdout.splitlines()
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to import the Shnifter package. \n{e.stderr}"
            ) from e

    if not building_found:
        logger.info("Did not build on import, triggering rebuild...\n")
        try:
            import shnifter  # noqa

            shnifter.build()

        except AttributeError:
            logger.info(
                "The Shnifter package does not have a build method, "
                "and may have been uninstalled or corrupted. "
                "Installing with --force-reinstall.\n"
            )
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "shnifter",
                        "--no-deps",
                        "--force-reinstall",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logger.info(result.stdout)
                result = subprocess.run(
                    [sys.executable, "-c", "import shnifter"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logger.info(result.stdout)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"{e.output} -> {e.stderr}") from None

        except Exception as e:  # pylint: disable=broad-except
            raise RuntimeError(  # noqa
                "Failed to build the Shnifter engine static assets. \n"
                f"{e} -> {e.__traceback__.tb_frame.f_code.co_filename}:"  # type:ignore  # pylint: disable=E1101
                f"{e.__traceback__.tb_lineno}"  # type:ignore
                if hasattr(e, "__traceback__")
                and hasattr(e.__traceback__, "tb_frame")  # type:ignore
                and hasattr(
                    e.__traceback__.tb_frame,  # type:ignore
                    "f_code",
                )
                and hasattr(
                    e.__traceback__.tb_frame.f_code,  # type:ignore  # pylint: disable=E1101
                    "co_filename",
                )
                and hasattr(
                    e.__traceback__,  # type:ignore
                    "tb_lineno",
                )
                else f"Failed to build the Shnifter engine static assets. \n{e}"
            ) from e
    sys.exit(0)


if __name__ == "__main__":
    main()
