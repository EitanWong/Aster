from __future__ import annotations

import pytest

from aster.runtime.mcp_security import MCPCommandValidator, MCPSecurityError


def test_mcp_validator_blocks_inline_python_execution() -> None:
    validator = MCPCommandValidator(check_path_exists=False)

    with pytest.raises(MCPSecurityError):
        validator.validate_command_args("python", ["-c", "print(1)"], "unsafe")


def test_mcp_validator_blocks_encoded_path_traversal() -> None:
    validator = MCPCommandValidator(check_path_exists=False)

    with pytest.raises(MCPSecurityError):
        validator.validate_args(["%2e%2e/secrets"], "unsafe")


def test_mcp_validator_allows_custom_whitelisted_command() -> None:
    validator = MCPCommandValidator(
        custom_whitelist={"custom-mcp"},
        check_path_exists=False,
    )

    validator.validate_command("custom-mcp", "safe")
    validator.validate_args(["--stdio"], "safe")
