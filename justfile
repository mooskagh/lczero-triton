set shell := ["bash", "-euo", "pipefail", "-c"]

proto_source := "submodules/lc0/proto/lc0ex.proto"
proto_include := "submodules/lc0/proto"
proto_output := "src/lc0ex/proto"
python_paths := "src tests"

# List available recipes.
default:
    @just --list

# Generate Python protobuf bindings and type stubs.
build-proto:
    mkdir -p "{{ proto_output }}"
    uv run --frozen python -m grpc_tools.protoc \
        --proto_path="{{ proto_include }}" \
        --python_out="{{ proto_output }}" \
        --pyi_out="{{ proto_output }}" \
        "{{ proto_source }}"

# Format Python source and apply safe lint fixes.
format:
    uv run --frozen ruff check --fix {{ python_paths }}
    uv run --frozen ruff format {{ python_paths }}

# Generate bindings and run all static and behavioral checks.
check: build-proto
    uv run --frozen ruff check {{ python_paths }}
    uv run --frozen ruff format --check {{ python_paths }}
    uv run --frozen mypy
    uv run --frozen pytest

# Run checks required before committing.
pre-commit: check
