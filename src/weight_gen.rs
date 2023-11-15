use nalgebra as na;
use rand::Rng;
use rand_distr::StandardNormal;

pub fn input_weight(input_dimension: usize, reservoir_dimension: usize) -> na::DMatrix<f64> {
    let mut rng = rand::thread_rng();

    let elements = (0..reservoir_dimension * (input_dimension + 1))
        .map(|_| rng.gen_bool(0.5))
        .map(|b| if b { 0.1 } else { -0.1 })
        .collect::<Vec<_>>();

    na::DMatrix::from_vec(reservoir_dimension, 1 + input_dimension, elements)
}

pub fn reservoir_weight(reservoir_dimension: usize) -> na::DMatrix<f64> {
    let mut rng = rand::thread_rng();

    let elements = (0..reservoir_dimension * reservoir_dimension)
        .map(|_| rng.sample(StandardNormal))
        .collect::<Vec<f64>>();

    let w = na::DMatrix::from_vec(reservoir_dimension, reservoir_dimension, elements);
    // let spectral_radius = w.eigenvalues().unwrap().abs().max();
    let spectral_radius = 0.9;

    w / spectral_radius
}

pub fn output_weight(reservoir_dimension: usize, output_dimension: usize) -> na::DMatrix<f64> {
    let elements = vec![0.0; reservoir_dimension * output_dimension];

    na::DMatrix::from_vec(reservoir_dimension, output_dimension, elements)
}
