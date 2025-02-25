use rand::prelude::*;
use rand_pcg::Pcg64;
use rand_distr::Normal;
use rayon::prelude::*;
use std::ffi::c_double;
use std::os::raw::c_int;
use rand::SeedableRng;
use libm::exp;

/// Monte Carlo simulation for estimating portfolio risk.
///
/// # Parameters:
/// - `num_trades`: Number of trades in the portfolio.
/// - `num_simulations`: Number of Monte Carlo simulations to run.
/// - `mu`: Expected daily return (drift).
/// - `sigma`: Daily volatility.
///
/// # Returns:
/// - The **expected portfolio P&L** (profit/loss) as a `f64`.
#[no_mangle]
pub extern "C" fn monte_carlo_portfolio(
    num_trades: c_int,
    num_simulations: c_int,
    mu: c_double,
    sigma: c_double,
) -> c_double {
    // Random seed
    let mut rng = Pcg64::seed_from_u64(42);

    // Generate random initial prices
    let initial_prices: Vec<f64> = (0..num_trades)
        .map(|_| rng.random_range(90.0..110.0))
        .collect();

    // Normal distribution
    let normal = Normal::new(mu, sigma).unwrap();

    // Run Monte Carlo simulations in parallel
    let total_pnl: f64 = (0..num_simulations)
        .into_par_iter()
        .map_init(|| (Pcg64::seed_from_u64(42), normal), |(rng, normal), _| {
            let mut total_pnl = 0.0;
            for &price in &initial_prices {
                let shock: f64 = rng.sample(*normal);
                let new_price = price * exp(shock); // Use high-precision exp()
                total_pnl += new_price - price;
            }
            total_pnl
        })
        .sum();

    total_pnl / num_simulations as f64
}