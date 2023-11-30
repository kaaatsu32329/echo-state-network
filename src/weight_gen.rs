use na::ComplexField;
use nalgebra as na;
use rand::Rng;
use rand_distr::StandardNormal;

pub fn input_weight(input_dimension: usize, reservoir_dimension: usize) -> na::DMatrix<f64> {
    let mut rng = rand::thread_rng();

    /*
    let elements = (0..reservoir_dimension * (input_dimension + 1))
        .map(|_| rng.gen_bool(0.5))
        .map(|b| if b { 0.1 } else { -0.1 })
        .collect::<Vec<_>>();
     */
    let elements = (0..reservoir_dimension * (input_dimension + 1))
        .map(|_| rng.gen_range(-0.5..0.5))
        .collect::<Vec<_>>();

    na::DMatrix::from_vec(reservoir_dimension, 1 + input_dimension, elements)
}

pub fn reservoir_weight(reservoir_dimension: usize) -> na::DMatrix<f64> {
    let mut rng = rand::thread_rng();

    /*
    let elements = (0..reservoir_dimension * reservoir_dimension)
        .map(|_| rng.sample(StandardNormal))
        .collect::<Vec<f64>>();

    let w = na::DMatrix::from_vec(reservoir_dimension, reservoir_dimension, elements);
    // let spectral_radius = w.eigenvalues().unwrap().abs().max();
    let spectral_radius = 0.9;

    w / spectral_radius
     */

    let elements = (0..reservoir_dimension * reservoir_dimension)
        .map(|_| rng.gen_range(-0.5..0.5))
        .collect::<Vec<_>>();

    let w = na::DMatrix::from_vec(reservoir_dimension, reservoir_dimension, elements);
    let eigens = w.complex_eigenvalues();
    let spectral_radius = eigens
        .iter()
        .map(|c| c.abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    w * 1.25 / spectral_radius
}

pub fn output_weight(reservoir_dimension: usize, output_dimension: usize) -> na::DMatrix<f64> {
    na::DMatrix::zeros(reservoir_dimension, output_dimension)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_input_weight() {
        let input_dimension = 2;
        let reservoir_dimension = 3;

        let w = input_weight(input_dimension, reservoir_dimension);

        println!("{}", w);

        assert_eq!(w.nrows(), reservoir_dimension);
        assert_eq!(w.ncols(), 1 + input_dimension);
    }

    #[test]
    fn test_reservoir_weight() {
        let reservoir_dimension = 3;

        let w = reservoir_weight(reservoir_dimension);

        println!("{}", w);

        assert_eq!(w.nrows(), reservoir_dimension);
        assert_eq!(w.ncols(), reservoir_dimension);
    }

    #[test]
    fn test_output_weight() {
        let reservoir_dimension = 3;
        let output_dimension = 2;

        let w = output_weight(reservoir_dimension, output_dimension);

        println!("{}", w);

        assert_eq!(w.nrows(), reservoir_dimension);
        assert_eq!(w.ncols(), output_dimension);
    }
}
