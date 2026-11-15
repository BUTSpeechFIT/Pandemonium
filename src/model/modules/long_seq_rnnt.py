"""Train the transducer (RNNT / TDT) losses on long target sequences (beyond ~750 tokens).

NeMo computes the transducer forward/backward (alpha/beta) lattice with numba CUDA kernels that
launch **one block per batch item, `maxU` threads per block** -- one thread per target label
position `u` (`gpu_rnnt_kernel.py`: `u = cuda.threadIdx.x`, and the launch
`compute_tdt_alphas_kernel[minibatch, maxU, ...]` in `gpu_rnnt.py`). The hardware caps a block at
1024 threads, but the real limit is lower and register-driven: the TDT kernels are register-heavy
(~85 registers/thread measured), and once `regs/thread * threads > 65536` (the per-SM register
file) the launch aborts with **`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`** -- measured here between
U=700 (runs) and U=800 (aborts) on an *otherwise-idle* GPU, so it is the kernel's own register
footprint, not GPU contention. This is the whole reason the long-form recipes use CTC, not TDT.

The alpha/beta recursion is an **anti-diagonal wavefront**: the kernels iterate `n = t + u` and,
after each diagonal, `cuda.syncthreads()`. Every cell `(t, u)` depends only on cells on strictly
earlier diagonals (`alpha[t-d, u]`, `alpha[t-d, u-1]`, all with `t-d < t`), so **all cells on one
diagonal are mutually independent**. That is what makes the fix exact rather than approximate: a
thread may own several `u` positions and compute them in any order within a diagonal without
changing a single value.

`enable_long_sequence_transducer()` swaps NeMo's four wavefront kernels
(`compute_alphas_kernel`, `compute_betas_kernel`, `compute_tdt_alphas_kernel`,
`compute_tdt_betas_kernel` -- the TDT path randomly uses the plain RNNT pair too, via `omega`) for
strided copies, and wraps every launch so `kernel[grid, maxU, ...]` clamps its block to
`SAFE_BLOCK_THREADS`. Each thread then strides `for u in range(tid, U, blockDim)`; `syncthreads()`
still separates the diagonals. For `U <= SAFE_BLOCK_THREADS` the block is exactly `U`, one `u` per
thread, i.e. bit for bit the stock kernels; the striding path itself is verified against stock in
`tests/test_long_seq_rnnt.py`. The gradient kernels already grid-stride over a fixed 256-thread
block (`GPU_RNNT_THREAD_SIZE`) over `minibatch * maxT * maxU` blocks, so they carry no such limit
and are left untouched.
"""

from numba import cuda

from nemo.collections.asr.parts.numba.rnnt_loss.utils import rnnt_helper
from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils import gpu_rnnt_kernel as _gpu

# Reused verbatim from the stock kernels so the arithmetic is identical: the log-prob lookups are
# `@cuda.jit(device=True)` and `INF` is the same sentinel the originals clamp against.
logp = _gpu.logp
logp_duration = _gpu.logp_duration
INF = _gpu.INF

# Block width the launches are clamped to. 512, not the 1024 hardware cap: the stock kernels abort
# with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES at ~750-800 threads (register pressure). The strided
# kernels here compile lighter (they launched even at 1024 in testing, i.e. <= 64 regs/thread), so
# 512 leaves a 2x register-file margin (512 * 64 = 32768 < 65536) and was verified bit-identical to
# stock. Drop toward 256 (universally safe: 256 * the 255-register max still fits) only if a future
# toolchain inflates the register count. The forward/backward is a negligible slice of the step, so
# the extra strides a smaller block adds cost nothing that matters.
SAFE_BLOCK_THREADS = 512


@cuda.jit()
def _strided_compute_alphas_kernel(acts, denom, alphas, llForward, xlen, ylen, mlabels,
                                   minibatch, maxT, maxU, alphabet_size, blank_):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x
    T = xlen[b]
    U = ylen[b] + 1
    labels = mlabels[b]
    offset = b * maxT * maxU

    if tid == 0:
        alphas[offset] = 0
    cuda.syncthreads()

    for n in range(1, T + U - 1):
        for u in range(tid, U, nthreads):
            t = n - u
            if u == 0:
                if t > 0 and t < T:
                    alphas[offset + t * maxU + u] = alphas[offset + (t - 1) * maxU + u] + logp(
                        denom, acts, maxT, maxU, alphabet_size, b, t - 1, 0, blank_
                    )
            elif u < U:
                if t == 0:
                    alphas[offset + u] = alphas[offset + u - 1] + logp(
                        denom, acts, maxT, maxU, alphabet_size, b, 0, u - 1, labels[u - 1]
                    )
                elif t > 0 and t < T:
                    no_emit = alphas[offset + (t - 1) * maxU + u] + logp(
                        denom, acts, maxT, maxU, alphabet_size, b, t - 1, u, blank_
                    )
                    emit = alphas[offset + t * maxU + u - 1] + logp(
                        denom, acts, maxT, maxU, alphabet_size, b, t, u - 1, labels[u - 1]
                    )
                    alphas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(emit, no_emit)
        cuda.syncthreads()

    if tid == 0:
        llForward[b] = alphas[offset + (T - 1) * maxU + U - 1] + logp(
            denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_
        )


@cuda.jit()
def _strided_compute_betas_kernel(acts, denom, betas, llBackward, xlen, ylen, mlabels,
                                  minibatch, maxT, maxU, alphabet_size, blank_):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x
    T = xlen[b]
    U = ylen[b] + 1
    labels = mlabels[b]
    offset = b * maxT * maxU

    if tid == 0:
        betas[offset + (T - 1) * maxU + U - 1] = logp(
            denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_
        )
    cuda.syncthreads()

    for n in range(T + U - 2, -1, -1):
        for u in range(tid, U, nthreads):
            t = n - u
            if u == (U - 1):
                if t >= 0 and t < (T - 1):
                    betas[offset + t * maxU + U - 1] = betas[offset + (t + 1) * maxU + U - 1] + logp(
                        denom, acts, maxT, maxU, alphabet_size, b, t, U - 1, blank_
                    )
            elif u < U:
                if t == T - 1:
                    betas[offset + (T - 1) * maxU + u] = betas[offset + (T - 1) * maxU + u + 1] + logp(
                        denom, acts, maxT, maxU, alphabet_size, b, T - 1, u, labels[u]
                    )
                elif (t >= 0) and (t < T - 1):
                    no_emit = betas[offset + (t + 1) * maxU + u] + logp(
                        denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_
                    )
                    emit = betas[offset + t * maxU + u + 1] + logp(
                        denom, acts, maxT, maxU, alphabet_size, b, t, u, labels[u]
                    )
                    betas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(emit, no_emit)
        cuda.syncthreads()

    if tid == 0:
        llBackward[b] = betas[offset]


@cuda.jit()
def _strided_compute_tdt_alphas_kernel(acts, duration_acts, denom, sigma, alphas, llForward,
                                       xlen, ylen, mlabels, minibatch, maxT, maxU, alphabet_size,
                                       blank_, durations, num_durations):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x
    T = xlen[b]
    U = ylen[b] + 1
    labels = mlabels[b]
    offset = b * maxT * maxU

    if tid == 0:
        alphas[offset] = 0
    cuda.syncthreads()

    for n in range(1, T + U - 1):
        for u in range(tid, U, nthreads):
            t = n - u
            if u == 0:
                if t > 0 and t < T:
                    alphas[offset + t * maxU + u] = -INF
                    for i in range(num_durations):
                        if durations[i] == 0:
                            continue
                        if t >= durations[i]:
                            alphas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(
                                alphas[offset + t * maxU + u],
                                alphas[offset + (t - durations[i]) * maxU + u]
                                + logp(denom, acts, maxT, maxU, alphabet_size, b, t - durations[i], u, blank_)
                                - sigma
                                + logp_duration(duration_acts, maxT, maxU, num_durations, b, t - durations[i], u, i),
                            )
                        else:
                            break
            elif u < U:
                if t == 0:
                    if durations[0] == 0:
                        alphas[offset + u] = (
                            alphas[offset + u - 1]
                            + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u - 1, labels[u - 1])
                            - sigma
                            + logp_duration(duration_acts, maxT, maxU, num_durations, b, t, u - 1, 0)
                        )
                    else:
                        alphas[offset + u] = -INF
                elif t > 0 and t < T:
                    no_emit = -INF
                    for i in range(num_durations):
                        if durations[i] == 0:
                            continue
                        if t >= durations[i]:
                            no_emit = rnnt_helper.log_sum_exp(
                                no_emit,
                                alphas[offset + (t - durations[i]) * maxU + u]
                                + logp(denom, acts, maxT, maxU, alphabet_size, b, t - durations[i], u, blank_)
                                - sigma
                                + logp_duration(duration_acts, maxT, maxU, num_durations, b, t - durations[i], u, i),
                            )
                        else:
                            break
                    emit = -INF
                    for i in range(num_durations):
                        if t >= durations[i]:
                            emit = rnnt_helper.log_sum_exp(
                                emit,
                                alphas[offset + (t - durations[i]) * maxU + u - 1]
                                + logp(denom, acts, maxT, maxU, alphabet_size, b, t - durations[i], u - 1, labels[u - 1])
                                - sigma
                                + logp_duration(duration_acts, maxT, maxU, num_durations, b, t - durations[i], u - 1, i),
                            )
                        else:
                            break
                    alphas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(emit, no_emit)
        cuda.syncthreads()

    if tid == 0:
        loglike = -INF
        for i in range(num_durations):
            if durations[i] == 0:
                continue
            if durations[i] == 1:
                loglike = (
                    alphas[offset + (T - 1) * maxU + U - 1]
                    + logp(denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_)
                    - sigma
                    + logp_duration(duration_acts, maxT, maxU, num_durations, b, T - 1, U - 1, i)
                )
                continue
            if T >= durations[i]:
                big_blank_loglike = (
                    alphas[offset + (T - durations[i]) * maxU + U - 1]
                    + logp(denom, acts, maxT, maxU, alphabet_size, b, T - durations[i], U - 1, blank_)
                    - sigma
                    + logp_duration(duration_acts, maxT, maxU, num_durations, b, T - durations[i], U - 1, i)
                )
                loglike = rnnt_helper.log_sum_exp(loglike, big_blank_loglike)
            else:
                break
        llForward[b] = loglike


@cuda.jit()
def _strided_compute_tdt_betas_kernel(acts, duration_acts, denom, sigma, betas, llBackward,
                                      xlen, ylen, mlabels, minibatch, maxT, maxU, alphabet_size,
                                      blank_, durations, num_durations):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x
    T = xlen[b]
    U = ylen[b] + 1
    labels = mlabels[b]
    offset = b * maxT * maxU

    if tid == 0:
        if durations[0] == 1:
            betas[offset + (T - 1) * maxU + U - 1] = (
                logp(denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_)
                - sigma
                + logp_duration(duration_acts, maxT, maxU, num_durations, b, T - 1, U - 1, 0)
            )
        elif durations[1] == 1:
            betas[offset + (T - 1) * maxU + U - 1] = (
                logp(denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_)
                - sigma
                + logp_duration(duration_acts, maxT, maxU, num_durations, b, T - 1, U - 1, 1)
            )
    cuda.syncthreads()

    for n in range(T + U - 2, -1, -1):
        for u in range(tid, U, nthreads):
            t = n - u
            if u == U - 1:
                if t >= 0 and t + 1 < T:
                    betas[offset + t * maxU + U - 1] = -INF
                    for i in range(num_durations):
                        if durations[i] == 0:
                            continue
                        if t + durations[i] < T:
                            betas[offset + t * maxU + U - 1] = rnnt_helper.log_sum_exp(
                                betas[offset + t * maxU + U - 1],
                                betas[offset + (t + durations[i]) * maxU + U - 1]
                                + logp(denom, acts, maxT, maxU, alphabet_size, b, t, U - 1, blank_)
                                + logp_duration(duration_acts, maxT, maxU, num_durations, b, t, U - 1, i)
                                - sigma,
                            )
                        elif t + durations[i] == T:
                            betas[offset + t * maxU + U - 1] = rnnt_helper.log_sum_exp(
                                betas[offset + t * maxU + U - 1],
                                logp(denom, acts, maxT, maxU, alphabet_size, b, t, U - 1, blank_)
                                + logp_duration(duration_acts, maxT, maxU, num_durations, b, t, U - 1, i)
                                - sigma,
                            )
            elif u < U - 1:
                if t == T - 1:
                    if durations[0] == 0:
                        betas[offset + (T - 1) * maxU + u] = (
                            betas[offset + (T - 1) * maxU + u + 1]
                            + logp(denom, acts, maxT, maxU, alphabet_size, b, T - 1, u, labels[u])
                            + logp_duration(duration_acts, maxT, maxU, num_durations, b, T - 1, u, 0)
                            - sigma
                        )
                    else:
                        betas[offset + (T - 1) * maxU + u] = -INF
                elif t >= 0 and t < T - 1:
                    no_emit = -INF
                    for i in range(num_durations):
                        if durations[i] == 0:
                            continue
                        if t + durations[i] < T:
                            no_emit = rnnt_helper.log_sum_exp(
                                no_emit,
                                betas[offset + (t + durations[i]) * maxU + u]
                                + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_)
                                + logp_duration(duration_acts, maxT, maxU, num_durations, b, t, u, i)
                                - sigma,
                            )
                    emit = -INF
                    for i in range(num_durations):
                        if t + durations[i] < T:
                            emit = rnnt_helper.log_sum_exp(
                                emit,
                                betas[offset + (t + durations[i]) * maxU + u + 1]
                                + logp(denom, acts, maxT, maxU, alphabet_size, b, t, u, labels[u])
                                + logp_duration(duration_acts, maxT, maxU, num_durations, b, t, u, i)
                                - sigma,
                            )
                    betas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(emit, no_emit)
        cuda.syncthreads()

    if tid == 0:
        llBackward[b] = betas[offset]


class _BlockClampedKernel:
    """Numba-kernel proxy that clamps the block (threads-per-block) dimension of every launch.

    NeMo launches these kernels as `kernel[minibatch, maxU, stream, shmem]`, binding the block
    width to `maxU`. Wrapping the kernel lets `maxU > 1024` fall back to a full 1024-thread block
    while the strided kernel body covers the remaining label positions -- with no edit to NeMo's
    launch site, which resolves the kernel by module attribute at call time.
    """

    def __init__(self, kernel, cap=SAFE_BLOCK_THREADS):
        self._kernel = kernel
        self._cap = cap

    def __getitem__(self, config):
        griddim, blockdim, stream, shmem = config
        if isinstance(blockdim, int) and blockdim > self._cap:
            blockdim = self._cap
        return self._kernel[griddim, blockdim, stream, shmem]


def enable_long_sequence_transducer(cap: int = SAFE_BLOCK_THREADS) -> None:
    """Replace NeMo's four wavefront kernels with strided, block-clamped versions.

    Idempotent and process-global: it swaps attributes on the shared `gpu_rnnt_kernel` module, so
    every `GPURNNT`/`GPUTDT` built afterwards (and the loss objects that hold them) picks up the
    long-sequence kernels. Numerically identical to the stock kernels for `U <= 1024`.
    """
    if getattr(_gpu, "_long_sequence_enabled", False):
        return
    _gpu.compute_alphas_kernel = _BlockClampedKernel(_strided_compute_alphas_kernel, cap)
    _gpu.compute_betas_kernel = _BlockClampedKernel(_strided_compute_betas_kernel, cap)
    _gpu.compute_tdt_alphas_kernel = _BlockClampedKernel(_strided_compute_tdt_alphas_kernel, cap)
    _gpu.compute_tdt_betas_kernel = _BlockClampedKernel(_strided_compute_tdt_betas_kernel, cap)
    _gpu._long_sequence_enabled = True
