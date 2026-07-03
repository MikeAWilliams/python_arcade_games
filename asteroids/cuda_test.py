"""Standalone CUDA sanity check for the local PyTorch install.

Purpose: verify the GPU path works end-to-end BEFORE trusting any real ML run.
The common failure mode on a new box (especially an RTX 50-series / Blackwell
card) is that torch installs cleanly and `torch.cuda.is_available()` returns
True, yet the wheel ships no kernels for the GPU's compute capability (sm_120).
`is_available()` will NOT catch that -- only actually launching a kernel does.

This script writes nothing to disk: no logs, no checkpoints, no weights.
Run it with:  uv run python cuda_test.py
"""

import torch


def main() -> None:
    # --- Build info -------------------------------------------------------
    # Which torch, and which CUDA toolkit version it was compiled against.
    # For a 5070 Ti you want a cu128 build; an older CUDA build won't have
    # sm_120 kernels no matter what the driver reports.
    print("torch      :", torch.__version__)
    print("cuda build :", torch.version.cuda)

    # --- Device visibility ------------------------------------------------
    # Proves the driver + runtime handshake works and torch can see the GPU.
    # NOTE: True here is necessary but NOT sufficient -- see the matmul below.
    available = torch.cuda.is_available()
    print("available  :", available)
    if not available:
        print("FAIL: no CUDA device visible to torch (driver / build mismatch).")
        return

    print("device     :", torch.cuda.get_device_name(0))

    # Compute capability identifies the GPU architecture. Blackwell (50-series)
    # reports (12, 0) == sm_120. This is the arch the wheel must have kernels
    # for; if torch only shipped up to sm_90, the compute step below will fail.
    print("capability :", torch.cuda.get_device_capability(0))

    # --- The check that actually matters ----------------------------------
    # Allocate on-GPU and run a real matmul. This forces a kernel launch and
    # actual compute on the device. synchronize() blocks until the GPU work
    # finishes, so any "no kernel image is available for execution on the
    # device" (missing sm_120 kernels) surfaces HERE rather than silently.
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    # A finite result copied back proves the full round trip: allocate ->
    # compute on GPU -> read back to CPU.
    print("matmul ok  :", tuple(y.shape), "| sum", float(y.sum()))

    # How much GPU memory the run touched -- confirms work happened on-device,
    # not on a silent CPU fallback.
    print("mem MB     :", round(torch.cuda.max_memory_allocated() / 1e6, 1))

    print("\nPASS: CUDA path works end-to-end (visible, sm matches, kernel ran).")


if __name__ == "__main__":
    main()
