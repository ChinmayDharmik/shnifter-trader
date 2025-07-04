# Generated: 2025-07-04T09:50:40.125449
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""MCP Server Settings model."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MCPSettings(BaseModel):
    """MCP Server settings model."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )

    # Basic server configuration
    name: str = Field(
        default="Shnifter MCP",
        alias="SHNIFTER_MCP_NAME",
    )
    description: str = Field(
        default="""All Shnifter REST endpoints exposed as MCP tools. Enables LLM agents
to query financial data, run screeners, and build workflows using
the exact same operations available to REST interfaceents.""",
        alias="SHNIFTER_MCP_DESCRIPTION",
    )

    # Tool category filtering
    default_tool_categories: List[str] = Field(
        default_factory=lambda: ["all"],
        description="Default active tool categories on startup",
        alias="SHNIFTER_MCP_DEFAULT_TOOL_CATEGORIES",
    )
    allowed_tool_categories: Optional[List[str]] = Field(
        default=None,
        description="If set, restricts available tool categories to this list",
        alias="SHNIFTER_MCP_ALLOWED_TOOL_CATEGORIES",
    )

    # Tool discovery configuration
    enable_tool_discovery: bool = Field(
        default=True,
        description="""
            Enable tool discovery, allowing the agent to hot-swap tools at runtime.
            Disable for multi-interfaceent or fixed toolset deployments.
        """,
        alias="SHNIFTER_MCP_ENABLE_TOOL_DISCOVERY",
    )

    # Response configuration
    describe_responses: bool = Field(
        default=True,
        description="Include response types in tool descriptions",
        alias="SHNIFTER_MCP_DESCRIBE_RESPONSES",
    )

    @field_validator(
        "default_tool_categories", "allowed_tool_categories", mode="before"
    )
    @classmethod
    def _split_list(cls, v):
        if isinstance(v, str):
            return [part.strip() for part in v.split(",") if part.strip()]
        return v

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}\n\n" + "\n".join(
            f"{k}: {v}" for k, v in self.model_dump().items()
        )
