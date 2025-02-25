import numpy as np
import time
import ctypes
import pyopencl as cl

# Parameters
NUM_TRADES = 1_000
NUM_SIMULATIONS = 1_000_000
MU = 0.001   # Daily drift (assumed mean return)
SIGMA = 0.02 # Daily volatility
WARMUP_ITERS = 8

# Monte Carlo Portfolio Simulation in Python (CPU)
def monte_carlo_portfolio(num_trades, num_simulations, mu, sigma):
    np.random.seed(42)
    initial_prices = np.random.uniform(90, 110, num_trades).astype(np.float64)
    shocks = np.random.normal(mu, sigma, (num_simulations, num_trades)).astype(np.float64)
    simulated_prices = initial_prices * np.exp(shocks)
    pnl = np.sum(simulated_prices - initial_prices, axis=1)
    return np.mean(pnl)

# Load Rust shared library
rust_lib = ctypes.CDLL("./target/release/libmonte_carlo.dylib")

# Setup Rust FFI CPU Implementation
rust_lib.monte_carlo_portfolio.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double]
rust_lib.monte_carlo_portfolio.restype = ctypes.c_double

# Setup Rust FFI GPU Implementation
rust_lib.monte_carlo_portfolio_gpu.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double]
rust_lib.monte_carlo_portfolio_gpu.restype = ctypes.c_double

# Warmup loop for Rust FFI CPU Implementation
for i in range(WARMUP_ITERS):
    start = time.perf_counter()
    pnl_rust_ffi = rust_lib.monte_carlo_portfolio(NUM_TRADES, NUM_SIMULATIONS, MU, SIGMA)
    end = time.perf_counter()
    if i == WARMUP_ITERS - 1:
        rust_time = end - start

# Warmup loop for Rust FFI GPU Implementation
for i in range(WARMUP_ITERS):
    start = time.perf_counter()
    pnl_rust_gpu = rust_lib.monte_carlo_portfolio_gpu(NUM_TRADES, NUM_SIMULATIONS, MU, SIGMA)
    end = time.perf_counter()
    if i == WARMUP_ITERS - 1:
        gpu_time = end - start

# Warmup loop for Python CPU Implementation
for i in range(WARMUP_ITERS):
    start = time.perf_counter()
    pnl_python = monte_carlo_portfolio(NUM_TRADES, NUM_SIMULATIONS, MU, SIGMA)
    end = time.perf_counter()
    if i == WARMUP_ITERS - 1:
        python_time = end - start

# Create an OpenCL context and queue (using a GPU device)
ctx = cl.Context(dev_type=cl.device_type.GPU)
queue = cl.CommandQueue(ctx)

# OpenCL kernel that performs the Monte Carlo simulation:
kernel_code = """
__kernel void monte_carlo(
    const int num_trades,
    const int num_simulations,
    const float mu,
    const float sigma,
    __global const float* initial_prices,
    __global float* results,
    const uint seed_offset)
{
    int gid = get_global_id(0);
    if(gid >= num_simulations) return;

    // Simple linear congruential generator (LCG) parameters
    uint seed = seed_offset + gid;
    uint a = 1664525;
    uint c = 1013904223;

    float sum = 0.0f;
    for (int i = 0; i < num_trades; i++) {
        // Generate two uniform random numbers via LCG.
        seed = a * seed + c;
        float u1 = (seed & 0x00FFFFFFu) / (float)0x01000000u;
        seed = a * seed + c;
        float u2 = (seed & 0x00FFFFFFu) / (float)0x01000000u;
        // Box–Muller transform to generate a normally distributed value.
        float r = sqrt(-2.0f * log(u1));
        float theta = 6.28318530718f * u2;
        float z = r * cos(theta);
        float shock = mu + sigma * z;
        float price = initial_prices[i];
        float new_price = price * exp(shock);
        sum += (new_price - price);
    }
    results[gid] = sum;
}
"""

# Build the OpenCL program.
prg = cl.Program(ctx, kernel_code).build()

# Prepare input data for Python GPU (OpenCL)
np.random.seed(42)
initial_prices = np.random.uniform(90, 110, NUM_TRADES).astype(np.float32)
results = np.empty(NUM_SIMULATIONS, dtype=np.float32)

mf = cl.mem_flags
prices_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=initial_prices)
results_buf = cl.Buffer(ctx, mf.WRITE_ONLY, results.nbytes)

global_size = (NUM_SIMULATIONS,)
seed_offset = np.uint32(42)

# Warmup loop for Python GPU (OpenCL)
for i in range(WARMUP_ITERS):
    start = time.perf_counter()
    prg.monte_carlo(
        queue, global_size, None,
        np.int32(NUM_TRADES),
        np.int32(NUM_SIMULATIONS),
        np.float32(MU),
        np.float32(SIGMA),
        prices_buf,
        results_buf,
        seed_offset
    )
    queue.finish()
    cl.enqueue_copy(queue, results, results_buf)
    mean_pnl = np.mean(results)
    end = time.perf_counter()
    if i == WARMUP_ITERS - 1:
        pgpu_time = end - start

# Print final results and timings from the last iteration
print(f"Python CPU Expected P&L: {pnl_python:.2f} (Time: {python_time:.6f}s)")
print(f"Rust FFI CPU Expected P&L: {pnl_rust_ffi:.2f} (Time: {rust_time:.6f}s)")
print("Python GPU (OpenCL) Expected P&L: {:.2f} (Time: {:.6f}s)".format(mean_pnl, pgpu_time))
print(f"Rust GPU Expected P&L: {pnl_rust_gpu:.2f} (Time: {gpu_time:.6f}s)")
print("")

# Print speedup comparisons
speedup_cpu = python_time / rust_time
speedup_gpu = python_time / gpu_time
speedup_gpu_vs_cpu = rust_time / gpu_time
speedup_pgpu_vs_python = python_time / pgpu_time
speedup_gpu_vs_pgpu = pgpu_time / gpu_time

print(f"Python GPU (OpenCL) is roughly {speedup_pgpu_vs_python:.1f}x faster than Python CPU")
print(f"Rust CPU is roughly {speedup_cpu:.1f}x faster than Python CPU")
print(f"Rust GPU is roughly {speedup_gpu:.1f}x faster than Python CPU")
print(f"Rust GPU is roughly {speedup_gpu_vs_cpu:.1f}x faster than Rust CPU")
print(f"Rust GPU is roughly {speedup_gpu_vs_pgpu:.1f}x faster than Python GPU")
