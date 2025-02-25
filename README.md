# Monte Carlo Portfolio Simulation

A very simplified example of high-performance Monte Carlo simulation for portfolio risk analysis, with multiple implementations across CPU and GPU platforms.

## Overview

This project demonstrates how to efficiently run Monte Carlo simulations for financial risk assessment. It provides multiple implementations:

- Python CPU (NumPy-based)
- Python GPU (OpenCL)
- Rust CPU (multi-threaded with Rayon)
- Rust GPU (WebGPU via wgpu)

The implementations are designed to be comparable, allowing you to benchmark performance across different languages and compute platforms.

## Features

- Portfolio simulation with configurable number of trades and simulations
- Support for both CPU and GPU acceleration
- Cross-platform compatibility (Windows, macOS, Linux) - only tested on Mac
- Nice summary results
- FFI interface to use Rust implementations from Python

## Requirements

### Rust
- Rust 1.84+ (2021 edition)
- cargo package manager

### Python
- Python 3.7+
- NumPy
- PyOpenCL (for GPU acceleration)
- ctypes (for FFI)
- tabulate (for nice benchmark display)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/joefrost01/monte-carlo
   cd monte-carlo
   ```

2. Build the Rust library:
   ```bash
   cargo build --release
   ```

3. Install Python dependencies:
   ```bash
   pip install numpy pyopencl tabulate
   ```

## Usage

### Running the benchmarks

```bash
python monte-carlo-test.py
```

This will automatically run all available implementations and display a performance comparison table.

### Using the library programmatically in Python

```python
import ctypes
import os
import platform

# Load the library based on platform
if platform.system() == "Windows":
    lib_path = os.path.join("target", "release", "monte_carlo.dll")
elif platform.system() == "Darwin":  # macOS
    lib_path = os.path.join("target", "release", "libmonte_carlo.dylib")
else:  # Linux and others
    lib_path = os.path.join("target", "release", "libmonte_carlo.so")

# Load the library
lib = ctypes.CDLL(lib_path)

# Configure function signatures
lib.monte_carlo_portfolio.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double
]
lib.monte_carlo_portfolio.restype = ctypes.c_double

# Run the simulation
num_trades = 1000
num_simulations = 1_000_000
mu = 0.001  # Daily drift
sigma = 0.02  # Daily volatility

# CPU simulation
result_cpu = lib.monte_carlo_portfolio(num_trades, num_simulations, mu, sigma)
print(f"Expected P&L (CPU): {result_cpu}")

# GPU simulation (if available)
result_gpu = lib.monte_carlo_portfolio_gpu(num_trades, num_simulations, mu, sigma)
print(f"Expected P&L (GPU): {result_gpu}")
```

### Using the Rust library directly

```rust
use monte_carlo::{monte_carlo_portfolio, monte_carlo_portfolio_gpu};

fn main() {
    let num_trades = 1000;
    let num_simulations = 1_000_000;
    let mu = 0.001;  // Daily drift
    let sigma = 0.02;  // Daily volatility
    
    // CPU simulation
    let result_cpu = monte_carlo_portfolio(num_trades, num_simulations, mu, sigma);
    println!("Expected P&L (CPU): {}", result_cpu);
    
    // GPU simulation
    let result_gpu = monte_carlo_portfolio_gpu(num_trades, num_simulations, mu, sigma);
    println!("Expected P&L (GPU): {}", result_gpu);
}
```

## How It Works

### Monte Carlo Simulation

The simulation generates random initial prices for a set of trades, then applies random shocks based on a normal distribution with parameters μ (drift) and σ (volatility). It calculates the resulting profit or loss (P&L) for each simulation run and returns the expected P&L as the average across all simulations.

The core calculation uses the formula:
- New Price = Initial Price * exp(μ + σ * Z)
- Where Z is a random variable from the standard normal distribution

### Implementation Details

#### Rust CPU
- Uses Rayon for parallelization across CPU cores
- PCG64 random number generator for high-quality random numbers
- Each thread handles multiple simulations independently

#### Rust GPU
- Uses wgpu for cross-platform GPU computation
- Implements a simple linear congruential generator (LCG) for random numbers
- Box-Muller transform to convert uniform random numbers to normal distribution
- One GPU work item per simulation

#### Python CPU
- NumPy-based vectorized operations
- Leverages NumPy's fast random number generators

#### Python GPU
- Uses OpenCL for GPU acceleration
- Similar approach to Rust GPU implementation

## Performance

Performance varies by hardware, but generally:
1. GPU implementations (Rust, OpenCL) provide the best performance
2. Multi-threaded Rust CPU implementation is typically faster than Python
3. Python CPU implementation serves as a good baseline

Typical speedups on modern hardware:
- Rust GPU: 10-100x faster than Python CPU
- Rust CPU: 2-10x faster than Python CPU
- Python GPU (OpenCL): 5-50x faster than Python CPU

## Project Structure

```
monte-carlo/
├── src/
│   ├── lib.rs                 # Library entry point
│   ├── monte_carlo_cpu.rs     # CPU implementation
│   └── monte_carlo_gpu.rs     # GPU implementation
├── monte-carlo-test.py        # Python benchmark script
├── Cargo.toml                 # Rust dependencies and configuration
└── README.md                  # This file
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Here are some ways you can contribute:
- Implement additional Monte Carlo methods
- Add support for more complex financial models
- Optimise existing implementations
- Add tests and documentation
- Report bugs and feature requests