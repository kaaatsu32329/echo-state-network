use crate::*;
use nalgebra as na;
use weight_gen::*;

pub struct EchoStateNetwork {
    /// Nu x T Matrix
    pub(crate) input: Ut,
    /// Nx x T Matrix
    reservoir_variable: Xt,
    /// Ny x T Matrix
    pub output: Yt,
    /// (1 + Nu + Nx) x T Matrix
    state: Zt,
    /// Nx x (1 + Nu) Matrix
    input_weight: na::DMatrix<f64>,
    /// Nx x Nx Matrix
    reservoir_weight: na::DMatrix<f64>,
    /// Ny x (1 + Nu + Nx) Matrix
    output_weight: na::DMatrix<f64>,
    leak_rate: f64,
    activator: fn(na::DVector<f64>) -> na::DVector<f64>,
    pub(crate) delta: f64,

    ///
    pub predicted: Option<Yt>,
}

impl EchoStateNetwork {
    pub fn new(
        input_dimension: usize,
        reservoir_dimension: usize,
        output_dimension: usize,
        input: Ut,
        leak_rate: f64,
        activator: fn(na::DVector<f64>) -> na::DVector<f64>,
        delta: f64,
    ) -> Self {
        let state_dimention = 1 + input_dimension + reservoir_dimension;
        let input_weight = input_weight(input_dimension, reservoir_dimension);
        let reservoir_weight = reservoir_weight(reservoir_dimension);
        let output_weight = output_weight(output_dimension, state_dimention);

        let reservoir_variable = Reservoir(na::DMatrix::zeros(reservoir_dimension, 1));
        let output = Output(na::DMatrix::zeros(output_dimension, 1));

        let mut state = State(na::DMatrix::zeros(state_dimention, input.0.ncols()));
        state.0.row_mut(0).fill(1.0);

        Self {
            input,
            reservoir_variable,
            output,
            state,
            input_weight,
            reservoir_weight,
            output_weight,
            leak_rate,
            activator,
            delta,
            predicted: None,
        }
    }

    pub fn train(&mut self, regularization_coefficient: f64) {
        let steps = self.input.0.ncols();
        for step in 0..steps - 1 {
            self.update_state(step);
            self.update_reservoir_variables(step);
        }
        self.update_state(steps - 1);
        self.update_output_weight(regularization_coefficient);

        self.result_trained();
    }

    pub fn predict(&mut self, predict_steps: usize) {
        let mut reservoir_variable = self
            .reservoir_variable
            .0
            .column(self.reservoir_variable.0.ncols() - 1)
            .into();

        let size = 1 + self.input.0.nrows() + self.reservoir_variable.0.nrows();

        let mut predicted_output = na::DMatrix::zeros(self.output.0.nrows(), predict_steps);
        predicted_output
            .column_mut(0)
            .copy_from(&self.output.0.column(self.output.0.ncols() - 1));

        for step in 0..predict_steps - 1 {
            let u_t = predicted_output.column(step);
            let u_t = u_t.insert_row(0, 1.0);
            reservoir_variable =
                self.sequence_reservoir_variables(&u_t.clone().into(), &reservoir_variable);

            let mut state = na::DMatrix::zeros(size, 1);
            state.view_mut((0, 0), (u_t.nrows(), 1)).copy_from(&u_t);
            state
                .view_mut((u_t.nrows(), 0), (reservoir_variable.nrows(), 1))
                .copy_from(&reservoir_variable);

            predicted_output
                .column_mut(step + 1)
                .copy_from(&(self.output_weight.clone() * state));
        }
        self.predicted = Some(Output(predicted_output));
    }

    pub fn result_trained(&mut self) {
        self.output.0 = self.output_weight.clone() * self.state.0.clone();
    }

    fn update_reservoir_variables(&mut self, step: usize) {
        let reservoir_variable = self.reservoir_variable.0.clone();
        let u_t = self.input.0.clone().column(step).insert_row(0, 1.0);
        let x_t = reservoir_variable.column(step);
        let x_t1 = self.sequence_reservoir_variables(&u_t, &x_t.into());

        self.append_reservoir_variables(&x_t1, step);
    }

    fn sequence_reservoir_variables(
        &self,
        u_t: &na::DVector<f64>,
        x_t: &na::DVector<f64>,
    ) -> na::DVector<f64> {
        let x_hat = (self.activator)(
            self.input_weight.clone() * u_t + self.reservoir_weight.clone() * x_t.clone(),
        );

        (1. - self.leak_rate) * x_t + self.leak_rate * x_hat
    }

    fn append_reservoir_variables(&mut self, x_t: &na::DVector<f64>, step: usize) {
        self.reservoir_variable = Reservoir({
            let mut mat = self
                .reservoir_variable
                .0
                .clone()
                .insert_column(step + 1, 0.0);
            mat.column_mut(step + 1).copy_from(x_t);
            mat
        });
    }

    fn update_state(&mut self, step: usize) {
        let z_t = self.sequence_state(step);
        self.append_state(&z_t, step);
    }

    fn sequence_state(&self, step: usize) -> na::DVector<f64> {
        let size = 1 + self.input.0.nrows() + self.reservoir_variable.0.nrows();
        let mut z_t = na::DVector::zeros(size);
        z_t[0] = 1.0;
        z_t.view_mut((1, 0), (self.input.0.nrows(), 1))
            .copy_from(&self.input.0.column(step));
        z_t.view_mut(
            (1 + self.input.0.nrows(), 0),
            (self.reservoir_variable.0.nrows(), 1),
        )
        .copy_from(&self.reservoir_variable.0.column(step));

        z_t
    }

    fn append_state(&mut self, z_t: &na::DVector<f64>, step: usize) {
        self.state.0.column_mut(step).copy_from(&z_t);
    }

    fn update_output_weight(&mut self, regularization_coefficient: f64) {
        let size = 1 + self.input.0.nrows() + self.reservoir_variable.0.nrows();
        let identity = regularization_coefficient * na::DMatrix::<f64>::identity(size, size);

        let z_z = self.state.0.clone() * self.state.0.clone().transpose();

        self.output_weight = self.input.0.clone()
            * self.state.0.clone().transpose()
            * (z_z + identity).try_inverse().unwrap();
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use assert_approx_eq::assert_approx_eq;
    use nalgebra as na;

    const INPUT_DIMENSION: usize = 1;
    const RESERVOIR_DIMENSION: usize = 4;
    const OUTPUT_DIMENSION: usize = 1;
    const LEAK_RATE: f64 = 0.05;
    const REGULARIZATION_COEFFICIENT: f64 = 1e-2;

    #[test]
    fn test_train() {
        let step = 10;
        let mut esn = esn_for_test(step);

        esn.train(REGULARIZATION_COEFFICIENT);
    }

    #[test]
    fn test_next_reservoir_variable() {
        let step = 2;
        let mut esn = esn_for_test(step);
        esn.update_reservoir_variables(0);

        let tanh_0_2 = 0.2_f64.tanh();
        let elements = vec![tanh_0_2, -tanh_0_2, tanh_0_2, -tanh_0_2];
        let x_hat = LEAK_RATE * na::DVector::from_vec(elements);
        let mut expected_reservoir = na::DMatrix::zeros(RESERVOIR_DIMENSION, step);
        expected_reservoir.column_mut(1).copy_from(&x_hat);

        for (i, e) in esn.reservoir_variable.0.iter().enumerate() {
            assert_approx_eq!(e, expected_reservoir.data.as_vec()[i]);
        }
    }

    #[test]
    fn test_update_output_weight() {
        let step = 2;
        let mut esn = esn_for_test(step);
        let size = 1 + INPUT_DIMENSION + RESERVOIR_DIMENSION;

        esn.update_reservoir_variables(0);
        esn.update_output_weight(REGULARIZATION_COEFFICIENT);

        let expected_output_weight = esn.input.0 * esn.state.0.transpose() * {
            let z_z = esn.state.0.clone() * esn.state.0.clone().transpose()
                + REGULARIZATION_COEFFICIENT * na::DMatrix::identity(size, size);
            z_z.try_inverse().unwrap()
        };

        for (i, e) in esn.output_weight.data.as_vec().iter().enumerate() {
            assert_approx_eq!(e, expected_output_weight.data.as_vec()[i]);
        }
    }

    fn esn_for_test(step: usize) -> EchoStateNetwork {
        let mut esn = EchoStateNetwork::new(
            INPUT_DIMENSION,
            RESERVOIR_DIMENSION,
            OUTPUT_DIMENSION,
            input_for_test(step),
            LEAK_RATE,
            utils::relu,
            0.1,
        );

        let input = (0..(RESERVOIR_DIMENSION * (1 + INPUT_DIMENSION)))
            .map(|i| if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect::<Vec<_>>();
        esn.input_weight = na::DMatrix::from_vec(RESERVOIR_DIMENSION, 1 + INPUT_DIMENSION, input);
        let reservoir = (0..RESERVOIR_DIMENSION * RESERVOIR_DIMENSION)
            .map(|i| i as f64 / (RESERVOIR_DIMENSION * RESERVOIR_DIMENSION) as f64)
            .collect::<Vec<_>>();
        esn.reservoir_weight =
            na::DMatrix::from_vec(RESERVOIR_DIMENSION, RESERVOIR_DIMENSION, reservoir);

        esn
    }

    fn input_for_test(step: usize) -> Input {
        let elements = (0..step).map(|i| (i + 1) as f64).collect::<Vec<_>>();
        Input(na::DMatrix::from_vec(1, step, elements))
    }
}
