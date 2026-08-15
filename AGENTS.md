# AGENTS.md

- Use complete Python type annotations; strict type checking is required.
- Follow PEP 8 and use Ruff for linting and formatting.
- Use the root `justfile`: `just format` applies formatting, `just check` runs all checks, and `just build-proto` generates protobuf bindings.
- The pre-commit hook runs `just pre-commit`.

Sample BT4 executable build:

```bash
uv run --frozen --package lczero-triton lczero-triton graph \
    --network /home/crem/dev/lc0/build/release/BT4-1024x15x32h-swa-6147500.pb.gz \
    --output /tmp/BT4-1024x15x32h-swa-6147500-sm120.lc0ex \
    --batch-size 169
```
