pub mod monte_carlo_cpu;
pub mod monte_carlo_gpu;

pub use monte_carlo_cpu::monte_carlo_portfolio;
pub use monte_carlo_gpu::monte_carlo_portfolio_gpu;
