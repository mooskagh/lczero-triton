"""Development sandbox for the Lc0 Triton compiler."""

from lc0ex import ExecutableBuilder
from lc0ex.proto import lc0ex_pb2


def main() -> None:
    """Build and serialize a minimal neural executable."""
    builder = ExecutableBuilder().set_target(
        lc0ex_pb2.Target.VENDOR_NVIDIA,
        "sm_80",
    )
    _ = builder.build().SerializeToString()


if __name__ == "__main__":
    main()
