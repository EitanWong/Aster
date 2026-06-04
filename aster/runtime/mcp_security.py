# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import posixpath
import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

ALLOW_UNSAFE_ENV_VAR = "ASTER_MCP_ALLOW_UNSAFE"

ALLOWED_COMMANDS = {
    "npx",
    "npm",
    "node",
    "uvx",
    "uv",
    "python",
    "python3",
    "pipx",
    "docker",
    "mcp-server-filesystem",
    "mcp-server-fetch",
    "mcp-server-github",
    "mcp-server-memory",
    "mcp-server-postgres",
    "mcp-server-puppeteer",
    "mcp-server-sqlite",
}

DANGEROUS_PATTERNS = [
    re.compile(r";\s*"),
    re.compile(r"\|\s*"),
    re.compile(r"&&\s*"),
    re.compile(r"\|\|\s*"),
    re.compile(r"`"),
    re.compile(r"\$\("),
    re.compile(r">\s*"),
    re.compile(r"<\s*"),
    re.compile(r"\.\./"),
    re.compile(r"~"),
]

DANGEROUS_ARG_PATTERNS = [
    re.compile(r";\s*"),
    re.compile(r"\|\s*"),
    re.compile(r"&&\s*"),
    re.compile(r"\|\|\s*"),
    re.compile(r"`"),
    re.compile(r"\$\("),
    re.compile(r"\$\{"),
    re.compile(r">\s*/"),
    re.compile(r"<\s*/"),
]

BLOCKED_COMMAND_ARG_RULES = {
    "python": {"-c": "inline Python execution"},
    "python3": {"-c": "inline Python execution"},
    "node": {
        "-e": "inline JavaScript evaluation",
        "--eval": "inline JavaScript evaluation",
        "-p": "JavaScript evaluation/print",
        "--print": "JavaScript evaluation/print",
    },
    "npx": {
        "-c": "shell command execution",
        "--call": "shell command execution",
    },
}

CONTROL_CHARS = ("\n", "\r")


class MCPSecurityError(ValueError):
    pass


class MCPCommandValidator:
    def __init__(
        self,
        *,
        allowed_commands: set[str] | None = None,
        allow_unsafe: bool = False,
        custom_whitelist: set[str] | None = None,
        check_path_exists: bool = True,
    ) -> None:
        self.allow_unsafe = allow_unsafe
        self.allowed_commands = set(allowed_commands or ALLOWED_COMMANDS)
        self.check_path_exists = check_path_exists
        if custom_whitelist:
            self.allowed_commands.update(custom_whitelist)

    def validate_command(self, command: str, server_name: str) -> None:
        if self.allow_unsafe:
            return
        self._check_control_chars(command, "command", server_name)
        self._check_path_traversal(command, "command", server_name)
        self._check_patterns(command, DANGEROUS_PATTERNS, "command", server_name)

        base_command = Path(command).name
        if base_command not in self.allowed_commands:
            if os.path.isabs(command) and Path(command).name in self.allowed_commands:
                if os.path.isfile(command) and os.access(command, os.X_OK):
                    return
            raise MCPSecurityError(
                f"MCP server {server_name!r}: command {base_command!r} is not allowlisted."
            )
        if self.check_path_exists and not os.path.isabs(command) and shutil.which(command) is None:
            raise MCPSecurityError(
                f"MCP server {server_name!r}: command {command!r} was not found in PATH."
            )

    def validate_args(self, args: list[str], server_name: str) -> None:
        if self.allow_unsafe:
            return
        for index, arg in enumerate(args):
            context = f"argument {index}"
            self._check_control_chars(arg, context, server_name)
            self._check_path_traversal(arg, context, server_name)
            self._check_patterns(arg, DANGEROUS_ARG_PATTERNS, context, server_name)

    def validate_command_args(self, command: str, args: list[str], server_name: str) -> None:
        if self.allow_unsafe:
            return
        rules = BLOCKED_COMMAND_ARG_RULES.get(Path(command).name)
        if not rules:
            return
        for index, arg in enumerate(args):
            if arg in rules:
                raise MCPSecurityError(
                    f"MCP server {server_name!r}: argument {index} {arg!r} enables {rules[arg]}."
                )
            if Path(command).name == "node" and arg.startswith("--eval="):
                raise MCPSecurityError(
                    f"MCP server {server_name!r}: argument {index} enables inline JavaScript evaluation."
                )
            if Path(command).name == "npx" and arg.startswith("--call="):
                raise MCPSecurityError(
                    f"MCP server {server_name!r}: argument {index} enables shell command execution."
                )

    def validate_env(self, env: dict[str, str] | None, server_name: str) -> None:
        if self.allow_unsafe or not env:
            return
        blocked_names = {
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
            "PATH",
            "PYTHONPATH",
            "NODE_PATH",
        }
        for key, value in env.items():
            if key.upper() in blocked_names:
                raise MCPSecurityError(
                    f"MCP server {server_name!r}: environment variable {key!r} is not allowed."
                )
            self._check_control_chars(value, f"environment variable {key!r}", server_name)
            self._check_path_traversal(value, f"environment variable {key!r}", server_name)
            self._check_patterns(value, DANGEROUS_ARG_PATTERNS, f"environment variable {key!r}", server_name)

    def validate_url(self, url: str, server_name: str) -> None:
        if self.allow_unsafe:
            return
        self._check_control_chars(url, "url", server_name)
        if not url.startswith(("http://", "https://")):
            raise MCPSecurityError(f"MCP server {server_name!r}: URL must use http:// or https://.")
        parsed = urlparse(url)
        self._check_path_traversal(parsed.path, "url", server_name)
        if parsed.query:
            self._check_control_chars(parsed.query, "url query", server_name)
        self._check_patterns(url, DANGEROUS_PATTERNS, "url", server_name)

    @staticmethod
    def _check_control_chars(value: str, context: str, server_name: str) -> None:
        if any(char in value for char in CONTROL_CHARS):
            raise MCPSecurityError(
                f"MCP server {server_name!r}: {context} contains control characters."
            )

    @staticmethod
    def _check_patterns(
        value: str,
        patterns: list[re.Pattern[str]],
        context: str,
        server_name: str,
    ) -> None:
        for pattern in patterns:
            if pattern.search(value):
                raise MCPSecurityError(
                    f"MCP server {server_name!r}: {context} contains a dangerous pattern."
                )

    @staticmethod
    def _check_path_traversal(value: str, context: str, server_name: str) -> None:
        candidates = [value]
        decoded = unquote(value)
        if decoded != value:
            candidates.append(decoded)
        for candidate in candidates:
            if "/" not in candidate and "\\" not in candidate and "%2e" not in value.lower():
                continue
            normalized = posixpath.normpath(candidate.replace("\\", "/"))
            parts = [part for part in candidate.replace("\\", "/").split("/") if part]
            if normalized == ".." or normalized.startswith("../") or any(part == ".." for part in parts):
                raise MCPSecurityError(
                    f"MCP server {server_name!r}: {context} contains path traversal."
                )


_validator: MCPCommandValidator | None = None


def get_validator() -> MCPCommandValidator:
    global _validator
    if _validator is None:
        _validator = MCPCommandValidator(allow_unsafe=os.getenv(ALLOW_UNSAFE_ENV_VAR) == "1")
    return _validator


def set_validator(validator: MCPCommandValidator | None) -> None:
    global _validator
    _validator = validator


def validate_mcp_server_config(
    *,
    server_name: str,
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    url: str | None = None,
) -> None:
    validator = get_validator()
    if command:
        validator.validate_command(command, server_name)
    if args:
        validator.validate_args(args, server_name)
        if command:
            validator.validate_command_args(command, args, server_name)
    if env:
        validator.validate_env(env, server_name)
    if url:
        validator.validate_url(url, server_name)
