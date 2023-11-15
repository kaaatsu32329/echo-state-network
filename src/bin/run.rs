pub use echo_state_network::*;

const INPUT_DIMENSION: usize = 1;
const RESERVOIR_DIMENSION: usize = 100;
const OUTPUT_DIMENSION: usize = 1;
const LEAK_RATE: f64 = 0.05;
const REGULARIZATION_COEFFICIENT: f64 = 1e-2;

fn main() {
    let delta = 0.01;
    let gen = InputGen::new(0, 1001, delta);
    let amplitude = 1.0;

    // let input = gen.gen_sin_wave(amplitude);
    let input = gen.get_complex_wave(amplitude);

    let mut esn = EchoStateNetwork::new(
        INPUT_DIMENSION,
        RESERVOIR_DIMENSION,
        OUTPUT_DIMENSION,
        input,
        LEAK_RATE,
        utils::tanh,
        delta,
    );

    esn.train(REGULARIZATION_COEFFICIENT);

    esn.predict(1000);

    esn.plot("test").unwrap();
}
