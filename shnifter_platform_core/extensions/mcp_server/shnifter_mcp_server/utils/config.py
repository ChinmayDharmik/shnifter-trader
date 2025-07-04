# Generated: 2025-07-04T09:50:40.123660
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Configuration loader for MCP Server."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Union

from shnifter_core.app.constants import SHNIFTER_DIRECTORY

from .settings import MCPSettings

logger = logging.getLogger(__name__)


def get_mcp_config_path() -> Path:
    """Get the path to the MCP configuration file."""
    return SHNIFTER_DIRECTORY / "mcp_settings.json"


def load_mcp_settings() -> MCPSettings:
    """Load MCP settings from configuration file or create defaults."""
    config_path = get_mcp_config_path()

    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f)
            return MCPSettings(**config_data)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning(
                "Error loading MCP configuration from %s: %s. Using default settings.",
                config_path,
                e,
            )

    # Create default settings
    settings = MCPSettings()

    # Create the directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Save default settings to file
    save_mcp_settings(settings)

    return settings


def save_mcp_settings(settings: MCPSettings) -> None:
    """Save MCP settings to configuration file."""
    config_path = get_mcp_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(settings.model_dump(), f, indent=2)


def load_settings_from_env() -> Dict[str, Union[str, List[str], bool]]:
    """Load MCP settings from environment variables."""
    env_vars = {
        f.alias: os.environ[f.alias]
        for f in MCPSettings.model_fields.values()
        if f.alias and f.alias in os.environ
    }
    if not env_vars:
        return {}

    env_settings = MCPSettings.model_validate(env_vars, from_attributes=False)
    return env_settings.model_dump(exclude_unset=True)


def load_mcp_settings_with_overrides(**overrides) -> MCPSettings:
    """Load MCP settings with optional overrides."""
    settings = load_mcp_settings()

    # Apply environment variable overrides
    env_settings = load_settings_from_env()
    if env_settings:
        settings_dict = settings.model_dump()
        settings_dict.update(env_settings)
        settings = MCPSettings(**settings_dict)

    # Apply function argument overrides
    if overrides:
        settings_dict = settings.model_dump()
        settings_dict.update(overrides)
        settings = MCPSettings(**settings_dict)

    return settings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Start the Shnifter MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  shnifter-mcp                                        # Start with default settings
  shnifter-mcp --host 0.0.0.0 --port 8001           # Custom host and port
  shnifter-mcp --allowed-categories equity,crypto            # Allowed categories
  shnifter-mcp --default-categories equity,crypto            # Override default (all categories enabled by default)
  shnifter-mcp --transport stdio                     # Use stdio transport
  shnifter-mcp --no-tool-discovery                   # Disable tool discovery (multi-interfaceent safe)

The server can also be configured via:
  - Configuration file: ~/.shnifter_engine/mcp_settings.json
  - Environment variables: SHNIFTER_MCP_HOST, SHNIFTER_MCP_PORT, etc.

Command line arguments override configuration file and environment variables.
        """,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host IP address to bind the server to",
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="Port number to bind the server to"
    )
    parser.add_argument(
        "--allowed-categories",
        type=str,
        default=None,
        help="Comma-separated list of tool categories allowed to be used",
    )
    parser.add_argument(
        "--default-categories",
        type=str,
        default="all",
        help="Comma-separated list of tool categories enabled at startup",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="streamable-http",
        help="MCP transport protocol to use",
        choices=["streamable-http", "stdio", "sse"],
    )
    parser.add_argument(
        "--no-tool-discovery",
        action="store_true",
        help="Disable tool discovery (multi-interfaceent safe)",
    )
    return parser.parse_args()
