# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that all required assets are packaged in the wheel.

This test builds the wheel and verifies that all Metal shaders and
native libraries are included. This catches missing assets before
they cause runtime import failures.
"""

import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest


# Required assets that must be present in the wheel
REQUIRED_ASSETS = [
    "vllm_apple/metal/kernels/paged_attention_v2.metal",
    "vllm_apple/metal/kernels/paged_attention_fused.metal",
    "vllm_apple/metal/native/libkv_write.dylib",
    "vllm_apple/ops/metal/moe_kernel_v2.metal",
]


@pytest.fixture(scope="module")
def wheel_path():
    """Build wheel in a temporary directory and return path."""
    # tests/unit/test_packaging.py -> go up 3 levels to project root
    project_root = Path(__file__).parent.parent.parent

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build the wheel using sys.executable to ensure correct Python
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", tmpdir],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pytest.fail(f"Failed to build wheel:\n{result.stderr}")

        # Find the wheel file
        wheels = list(Path(tmpdir).glob("*.whl"))
        if not wheels:
            pytest.fail("No wheel file generated")

        yield wheels[0]


def test_required_assets_present(wheel_path):
    """Verify all required assets are included in the wheel."""
    with zipfile.ZipFile(wheel_path, "r") as whl:
        wheel_contents = set(whl.namelist())

        missing_assets = []
        for asset in REQUIRED_ASSETS:
            if asset not in wheel_contents:
                missing_assets.append(asset)

        if missing_assets:
            pytest.fail(
                f"Missing assets in wheel:\n"
                + "\n".join(f"  - {a}" for a in missing_assets)
                + f"\n\nWheel contents ({len(wheel_contents)} files):\n"
                + "\n".join(
                    f"  - {f}"
                    for f in sorted(wheel_contents)
                    if f.endswith((".metal", ".dylib", ".metallib"))
                )
            )


def test_no_old_extension_paths(wheel_path):
    """Verify old/incorrect extension paths are not in the wheel."""
    with zipfile.ZipFile(wheel_path, "r") as whl:
        wheel_contents = set(whl.namelist())

        old_paths = [
            f for f in wheel_contents if "extension/kernels" in f or "extension/compiled" in f
        ]

        if old_paths:
            pytest.fail(
                f"Found deprecated extension paths in wheel:\n"
                + "\n".join(f"  - {p}" for p in old_paths)
            )


def test_metal_files_readable(wheel_path):
    """Verify Metal shader files can be read and have content."""
    metal_files = [a for a in REQUIRED_ASSETS if a.endswith(".metal")]

    with zipfile.ZipFile(wheel_path, "r") as whl:
        for metal_file in metal_files:
            try:
                content = whl.read(metal_file)
                if len(content) < 100:
                    pytest.fail(f"Metal file {metal_file} appears truncated ({len(content)} bytes)")

                # Basic sanity check for Metal shader content
                content_str = content.decode("utf-8")
                if "kernel" not in content_str.lower() and "metal" not in content_str.lower():
                    pytest.fail(f"Metal file {metal_file} doesn't appear to be a valid shader")
            except KeyError:
                pytest.fail(f"Metal file {metal_file} not found in wheel")


def test_dylib_present_and_valid(wheel_path):
    """Verify dylib is present and has valid Mach-O header."""
    dylib_files = [a for a in REQUIRED_ASSETS if a.endswith(".dylib")]

    with zipfile.ZipFile(wheel_path, "r") as whl:
        for dylib_file in dylib_files:
            try:
                content = whl.read(dylib_file)

                # Check for Mach-O magic numbers
                # 0xFEEDFACE (32-bit), 0xFEEDFACF (64-bit), 0xCAFEBABE (universal)
                magic = content[:4]
                valid_magics = [
                    b"\xfe\xed\xfa\xce",  # MH_MAGIC (32-bit)
                    b"\xce\xfa\xed\xfe",  # MH_CIGAM (32-bit, byte-swapped)
                    b"\xfe\xed\xfa\xcf",  # MH_MAGIC_64 (64-bit)
                    b"\xcf\xfa\xed\xfe",  # MH_CIGAM_64 (64-bit, byte-swapped)
                    b"\xca\xfe\xba\xbe",  # FAT_MAGIC (universal)
                    b"\xbe\xba\xfe\xca",  # FAT_CIGAM (universal, byte-swapped)
                ]

                if magic not in valid_magics:
                    pytest.fail(f"dylib {dylib_file} has invalid Mach-O header: {magic.hex()}")
            except KeyError:
                pytest.fail(f"dylib {dylib_file} not found in wheel")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
