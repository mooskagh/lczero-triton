set shell := ["bash", "-euo", "pipefail", "-c"]

proto_sources := "lc0ex.proto lc0ex_metadata.proto net.proto"
proto_output := "packages/lc0ex/src/lc0ex/proto"
python_paths := "packages"

# List available recipes.
default:
    @just --list

# Generate Python protobuf bindings and type stubs.
build-proto:
    mkdir -p "{{ proto_output }}"
    uv run --no-sync python -m grpc_tools.protoc \
        --proto_path="submodules/lc0/proto" \
        --python_out="{{ proto_output }}" \
        --pyi_out="{{ proto_output }}" \
        {{ proto_sources }}

# Format Python source and apply safe lint fixes.
format:
    uv run --no-sync --all-packages ruff check --fix {{ python_paths }}
    uv run --no-sync --all-packages ruff format {{ python_paths }}

# Generate bindings and run all static and behavioral checks.
check: build-proto
    uv run --no-sync --all-packages ruff check {{ python_paths }}
    uv run --no-sync --all-packages ruff format --check {{ python_paths }}
    uv run --no-sync --all-packages mypy
    uv run --no-sync --all-packages pytest

# Run checks required before committing.
pre-commit: check
