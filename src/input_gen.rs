use nalgebra as na;

use crate::Input;

#[derive(Debug, Clone, Default)]
pub struct InputGen {
    start_time_step: usize,
    end_time_step: usize,
    delta_time: f64,
}

impl InputGen {
    pub fn new(start_time_step: usize, end_time_step: usize, delta_time: f64) -> Self {
        Self {
            start_time_step,
            end_time_step,
            delta_time,
        }
    }

    pub fn gen_sin_wave(&self, amplitude: f64) -> Input {
        let ncols = self.end_time_step - self.start_time_step;

        let elements: Vec<f64> = (self.start_time_step..self.end_time_step)
            .map(|t| ((t as f64) * self.delta_time).sin() * amplitude)
            .collect::<Vec<_>>();

        Input(na::DMatrix::from_vec(1, ncols, elements))
    }

    pub fn get_complex_wave(&self, amplitude: f64) -> Input {
        fn combined_wave(t: f64) -> f64 {
            (2.0 * t).sin()
                + (2.0 * t * 3.0).sin() / 3.0
                + (2.0 * t * 5.0).sin() / 5.0
                + (2.0 * t * 7.0).sin() / 7.0
        }

        let ncols = self.end_time_step - self.start_time_step;
        let elements = (self.start_time_step..self.end_time_step)
            .map(|t| (t as f64) * self.delta_time)
            .map(|t| amplitude * combined_wave(t))
            .collect::<Vec<_>>();

        Input(na::DMatrix::from_vec(1, ncols, elements))
    }
}
