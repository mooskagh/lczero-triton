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

Build and preload-check lc0ex from the repository root:

```bash
./submodules/lc0/build.sh release \
    -Dlc0=true \
    -Dlc0ex-runtime=true \
    -Dbuild_backends=false \
    -Dgtest=false

printf 'uci\nquit\n' | \
    ./submodules/lc0/build/release/lc0 \
    --config= \
    --preload \
    --weights=/home/crem/dev/lc0/build/release/BT4-1024x15x32h-swa-6147500.pb.gz \
    --backend=lc0ex-cuda \
    --backend-opts='lc0ex=/tmp/BT4-1024x15x32h-swa-6147500-sm120.lc0ex,gpu=0' \
    --logfile='<stderr>'
```

The preload check verifies lc0ex loading, CUDA initialization, and network
fingerprint compatibility; graph inference is not wired into the backend yet.
