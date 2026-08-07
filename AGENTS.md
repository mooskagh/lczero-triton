# AGENTS.md

- Use complete Python type annotations; strict type checking is required.
- Follow PEP 8 and use Ruff for linting and formatting.
- Use the root `justfile`: `just format` applies formatting, `just check` runs all checks, and `just build-proto` generates protobuf bindings.
- The pre-commit hook runs `just pre-commit`.
