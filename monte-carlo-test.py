import numpy as np
import time
import ctypes
import pyopencl as cl
import os
import platform
import sys
from tabulate import tabulate

# Parameters
NUM_TRADES = 1_000
NUM_SIMULATIONS = 1_000_000
MU = 0.001   # Daily drift (assumed mean return)
SIGMA = 0.02 # Daily volatility
WARMUP_ITERS = 8

# Determine the proper library extension based on the platform
if platform.system() == "Windows":
    LIB_EXT = ".dll"
elif platform.system() == "Darwin":  # macOS
    LIB_EXT = ".dylib"
else:  # Linux and others
    LIB_EXT = ".so"

# Library path
LIB_PATH = os.path.join(".", "target", "release", f"libmonte_carlo{LIB_EXT}")

# Python CPU Monte Carlo Portfolio Simulation
def monte_carlo_portfolio(num_trades, num_simulations, mu, sigma):
    np.random.seed(42)
    initial_prices = np.random.uniform(90, 110, num_trades).astype(np.float64)
    shocks = np.random.normal(mu, sigma, (num_simulations, num_trades)).astype(np.float64)
    simulated_prices = initial_prices * np.exp(shocks)
    pnl = np.sum(simulated_prices - initial_prices, axis=1)
    return np.mean(pnl)

# Function to create an OpenCL context
def create_opencl_context():
    # Try to get a GPU device first
    try:
        ctx = cl.Context(dev_type=cl.device_type.GPU)
        return ctx, "GPU"
    except:
        # Fall back to CPU if GPU is not available
        try:
            ctx = cl.Context(dev_type=cl.device_type.CPU)
            return ctx, "CPU"
        except:
            # Last resort - use the default platform's first device
            platform = cl.get_platforms()[0]
            devices = platform.get_devices()
            ctx = cl.Context(devices=devices)
            return ctx, "Default"

# Python GPU (OpenCL) implementation
def monte_carlo_portfolio_opencl(num_trades, num_simulations, mu, sigma):
    # Create an OpenCL context and queue
    ctx, device_type = create_opencl_context()
    queue = cl.CommandQueue(ctx)

    # Get device information
    devices = ctx.get_info(cl.context_info.DEVICES)
    device_name = devices[0].name
    print(f"Using OpenCL device: {device_name} ({device_type})")

    # OpenCL kernel that performs the Monte Carlo simulation
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

    # Build the OpenCL program
    prg = cl.Program(ctx, kernel_code).build()

    # Prepare input data for OpenCL
    np.random.seed(42)
    initial_prices = np.random.uniform(90, 110, num_trades).astype(np.float32)
    results = np.empty(num_simulations, dtype=np.float32)

    # Calculate optimal work group size
    max_work_group_size = devices[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    work_group_size = min(256, max_work_group_size)  # Use 256 or device max

    # Create buffers
    mf = cl.mem_flags
    prices_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=initial_prices)
    results_buf = cl.Buffer(ctx, mf.WRITE_ONLY, results.nbytes)

    # Set up global and local work sizes
    global_size = ((num_simulations + work_group_size - 1) // work_group_size * work_group_size,)
    local_size = (work_group_size,)
    seed_offset = np.uint32(42)

    # Execute the kernel
    prg.monte_carlo(
        queue, global_size, local_size,
        np.int32(num_trades),
        np.int32(num_simulations),
        np.float32(mu),
        np.float32(sigma),
        prices_buf,
        results_buf,
        seed_offset
    )
    queue.finish()

    # Read back results
    cl.enqueue_copy(queue, results, results_buf)
    mean_pnl = np.mean(results)

    return mean_pnl

def format_time(seconds):
    """Format time in a human-readable way based on magnitude."""
    if seconds < 0.001:
        return f"{seconds*1000000:.2f} µs"
    elif seconds < 1:
        return f"{seconds*1000:.2f} ms"
    else:
        return f"{seconds:.4f} s"

def run_benchmark():
    """Run all benchmarks and display results in a formatted table."""
    print("\nMonte Carlo Portfolio Risk Simulation Benchmark")
    print(f"Platform: {platform.system()} {platform.machine()} - {platform.processor()}")
    print(f"Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"\nParameters:")
    print(f"  - Trades: {NUM_TRADES:,}")
    print(f"  - Simulations: {NUM_SIMULATIONS:,}")
    print(f"  - Daily drift (μ): {MU}")
    print(f"  - Daily volatility (σ): {SIGMA}")
    print(f"  - Warmup iterations: {WARMUP_ITERS}")
    print("\n" + "="*80 + "\n")

    # Dictionary to store results
    results = {}
    implementations = [
        ("Python CPU", monte_carlo_portfolio, [NUM_TRADES, NUM_SIMULATIONS, MU, SIGMA]),
        ("Python GPU (OpenCL)", monte_carlo_portfolio_opencl, [NUM_TRADES, NUM_SIMULATIONS, MU, SIGMA]),
    ]

    # Try to load Rust library
    try:
        rust_lib = ctypes.CDLL(LIB_PATH)

        # Setup Rust FFI CPU Implementation
        rust_lib.monte_carlo_portfolio.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double
        ]
        rust_lib.monte_carlo_portfolio.restype = ctypes.c_double

        # Setup Rust FFI GPU Implementation
        rust_lib.monte_carlo_portfolio_gpu.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double
        ]
        rust_lib.monte_carlo_portfolio_gpu.restype = ctypes.c_double

        # Add Rust implementations to the list
        implementations.extend([
            ("Rust CPU", rust_lib.monte_carlo_portfolio, [NUM_TRADES, NUM_SIMULATIONS, MU, SIGMA]),
            ("Rust GPU", rust_lib.monte_carlo_portfolio_gpu, [NUM_TRADES, NUM_SIMULATIONS, MU, SIGMA]),
        ])

        print("Successfully loaded Rust library")
    except Exception as e:
        print(f"Failed to load Rust library: {e}")
        print(f"Make sure to build with 'cargo build --release' first")

    # Run all implementations
    for name, func, args in implementations:
        print(f"\nRunning {name} implementation...")
        sys.stdout.flush()  # Ensure output is displayed immediately

        # Warmup phase
        for i in range(WARMUP_ITERS):
            progress = "#" * (i + 1) + "." * (WARMUP_ITERS - i - 1)
            print(f"\r  Warmup [{progress}] iteration {i+1}/{WARMUP_ITERS}", end="")
            sys.stdout.flush()

            start = time.perf_counter()
            result = func(*args)
            end = time.perf_counter()

            if i == WARMUP_ITERS - 1:
                execution_time = end - start
                results[name] = {"time": execution_time, "result": result}
                print(f"\r  Result: {result:.4f}, Time: {format_time(execution_time)}{' '*20}")

    # Print summary table
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    # Create table data
    table_data = []
    baseline_time = results["Python CPU"]["time"] if "Python CPU" in results else 1.0

    for name, data in sorted(results.items(), key=lambda x: x[1]["time"]):
        speedup = baseline_time / data["time"]
        table_data.append([
            name,
            format_time(data["time"]),
            f"{data['result']:.4f}",
            f"{speedup:.1f}x"
        ])

    # Print table
    print(tabulate(
        table_data,
        headers=["Implementation", "Execution Time", "Expected P&L", "Speedup vs Python CPU"],
        tablefmt="pretty"
    ))

    # Print numerical consistency check
    print("\nNUMERICAL CONSISTENCY CHECK")
    print("-"*80)

    if "Python CPU" in results:
        baseline = results["Python CPU"]["result"]
        consistency_data = []

        for name, data in results.items():
            if name != "Python CPU":
                diff = abs(data["result"] - baseline)
                rel_diff = diff / abs(baseline) * 100 if baseline != 0 else float('inf')
                consistency_data.append([name, f"{diff:.8f}", f"{rel_diff:.4f}%"])

        print(tabulate(
            consistency_data,
            headers=["Implementation", "Absolute Difference", "Relative Difference"],
            tablefmt="pretty"
        ))

    # Display conclusion
    fastest_impl = min(results.items(), key=lambda x: x[1]["time"])
    print(f"\nFastest implementation: {fastest_impl[0]} ({format_time(fastest_impl[1]['time'])})")

    # Display potential performance improvement tips
    if "Rust GPU" in results and "Python CPU" in results:
        speedup = results["Python CPU"]["time"] / results["Rust GPU"]["time"]
        print(f"\nBy using Rust GPU instead of Python CPU, you're achieving a {speedup:.1f}x speedup!")

    print("\n" + "="*80)

if __name__ == "__main__":
    run_benchmark()