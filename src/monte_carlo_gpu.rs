use bytemuck::{Pod, Zeroable};
use futures::channel::oneshot;
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::ffi::c_double;
use std::os::raw::c_int;
use wgpu::util::DeviceExt;

/// Monte Carlo simulation for estimating portfolio risk using the GPU with wgpu.
///
/// This function sets up the GPU, uploads initial data (prices), and dispatches a compute shader
/// that performs the simulation. The shader uses a basic LCG and Box–Muller transform for random number generation.
/// It computes one simulation per work item and writes the simulation result (total P&L for that run)
/// to an output buffer. The CPU then reads back the results and averages them.
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
pub extern "C" fn monte_carlo_portfolio_gpu(
    num_trades: c_int,
    num_simulations: c_int,
    mu: c_double,
    sigma: c_double,
) -> c_double {
    pollster::block_on(async {
        // Create a wgpu instance using all available backends.
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find an appropriate adapter");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // Create initial prices on CPU: random values between 90.0 and 110.0.
        let mut rng = Pcg64::seed_from_u64(42);
        let initial_prices: Vec<f32> = (0..num_trades)
            .map(|_| rng.random_range(90.0..110.0) as f32)
            .collect();

        // Create a buffer for the initial prices.
        let prices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Prices Buffer"),
            contents: bytemuck::cast_slice(&initial_prices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Prepare an output buffer (one f32 result per simulation).
        let output_size = num_simulations as usize;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Define parameters to pass to the shader.
        #[repr(C)]
        #[derive(Copy, Clone, Pod, Zeroable)]
        struct Params {
            mu: f32,
            sigma: f32,
            num_trades: u32,
            num_simulations: u32,
        }
        let params = Params {
            mu: mu as f32,
            sigma: sigma as f32,
            num_trades: num_trades as u32,
            num_simulations: num_simulations as u32,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // WGSL compute shader.
        let shader_source = r#"
struct Params {
    mu: f32,
    sigma: f32,
    num_trades: u32,
    num_simulations: u32,
};

@group(0) @binding(0)
var<uniform> params: Params;

struct Prices {
    data: array<f32>,
};

@group(0) @binding(1)
var<storage, read> prices: Prices;

struct Results {
    data: array<f32>,
};

@group(0) @binding(2)
var<storage, read_write> results: Results;

// Simple linear congruential generator.
fn lcg(seed: u32) -> u32 {
    let a: u32 = 1664525u;
    let c: u32 = 1013904223u;
    return a * seed + c;
}

// Generate a pseudo-random float in [0, 1).
fn rand(seed: u32) -> f32 {
    let new_seed = lcg(seed);
    return f32(new_seed & 0x00FFFFFFu) / f32(0x01000000u);
}

// Box–Muller transform: converts two uniform random numbers into one normally distributed sample.
fn box_muller(u1: f32, u2: f32) -> f32 {
    let r = sqrt(-2.0 * log(u1));
    let theta = 6.28318530718 * u2;
    return r * cos(theta);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sim_index = global_id.x;
    if (sim_index >= params.num_simulations) {
        return;
    }
    var sum: f32 = 0.0;
    var seed: u32 = sim_index; // Seed per simulation.
    for (var i: u32 = 0u; i < params.num_trades; i = i + 1u) {
        // Generate two uniform random numbers.
        let u1 = rand(seed);
        seed = lcg(seed);
        let u2 = rand(seed);
        seed = lcg(seed);
        let z = box_muller(u1, u2);
        let shock = params.mu + params.sigma * z;
        let price = prices.data[i];
        let new_price = price * exp(shock);
        sum = sum + (new_price - price);
    }
    results.data[sim_index] = sum;
}
"#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Monte Carlo Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout and bind group.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                // Params uniform.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Prices buffer.
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Results buffer.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: prices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create the compute pipeline.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        // Encode commands.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // Dispatch enough workgroups to cover all simulations.
            let workgroup_size = 64;
            let workgroup_count =
                ((num_simulations as f32) / workgroup_size as f32).ceil() as u32;
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy the output buffer to a staging buffer for reading.
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (output_size * std::mem::size_of::<f32>()) as u64,
        );

        // Submit the commands.
        let command_buffer = encoder.finish();
        queue.submit(Some(command_buffer));

        // Map the staging buffer asynchronously using a oneshot channel.
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        receiver.await.unwrap().expect("Failed to map staging buffer");

        let data = buffer_slice.get_mapped_range();
        let results: &[f32] = bytemuck::cast_slice(&data);
        let total: f32 = results.iter().sum();

        // Clean up mapping.
        drop(data);
        staging_buffer.unmap();

        (total / num_simulations as f32) as f64
    })
}