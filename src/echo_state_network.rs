use crate::*;
use nalgebra as na;
use weight_gen::*;

pub struct EchoStateNetwork {
    pub var: Variable,
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
    pub trained: Option<Variable>,
    ///
    pub predicted: Option<Variable>,
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
        let mut state = State(na::DMatrix::zeros(
            1 + input_dimension + reservoir_dimension,
            input.0.ncols(),
        ));
        state.0.row_mut(0).fill(1.0);
        state.0.rows_mut(1, input_dimension).copy_from(&input.0);
        let var = Variable::new(
            state,
            Output(na::DMatrix::zeros(output_dimension, input.0.ncols())),
            input_dimension,
            reservoir_dimension,
            output_dimension,
        );

        let state_dimension = 1 + input_dimension + reservoir_dimension;
        let input_weight = input_weight(input_dimension, reservoir_dimension);
        let reservoir_weight = reservoir_weight(reservoir_dimension);
        let output_weight = output_weight(output_dimension, state_dimension);

        Self {
            var,
            input_weight,
            reservoir_weight,
            output_weight,
            leak_rate,
            activator,
            delta,
            trained: None,
            predicted: None,
        }
    }

    pub fn train(&mut self, regularization_coefficient: f64) {
        let steps = self.var.step();
        let u_0 = self.var.input_with_bias_t(0).clone();
        let x_0 = (self.activator)(self.input_weight.clone() * u_0);
        self.append_reservoir_variables(&x_0, 0);

        for step in 0..steps - 1 {
            self.update_reservoir_variables(step);
        }
        self.update_output_weight(regularization_coefficient);

        self.var.output.0 = self.output_weight.clone() * self.var.state.0.clone();
    }

    pub fn predict(&mut self, predict_steps: usize) {
        let u_1 = self.var.output_t(self.var.output.0.ncols() - 1).clone();

        let x_0 = self
            .var
            .reservoir_t(self.var.reservoir().0.ncols() - 1)
            .clone();

        let mut state = State(na::DMatrix::zeros(self.var.size(), predict_steps));
        state.0.row_mut(0).fill(1.0);
        state
            .0
            .view_mut((1, 1), (self.var.input_dimension, 1))
            .copy_from(&u_1);
        state
            .0
            .view_mut(
                (1 + self.var.input_dimension, 0),
                (self.var.reservoir_dimension, 1),
            )
            .copy_from(&x_0);

        let output = Output(na::DMatrix::zeros(
            self.var.output_dimension,
            predict_steps + 1,
        ));

        let mut predicted = Variable::new(
            state,
            output,
            self.var.input_dimension,
            self.var.reservoir_dimension,
            self.var.output_dimension,
        );

        for step in 0..predict_steps - 1 {
            let x_t = predicted.reservoir_t(step);
            let u_t1 = predicted.input_with_bias_t(step + 1);
            let x_t1 = self.sequence_reservoir_variables(&u_t1, &x_t);

            predicted.replace_reservoir_t(step + 1, &x_t1);

            let y_t1 = self.output_weight.clone() * predicted.state_t(step + 1);
            predicted.replace_output_t(step + 1, &y_t1);
        }

        predicted.state.0 = predicted.state.0.remove_column(0);
        predicted.output.0 = predicted.output.0.remove_column(0);

        self.predicted = Some(predicted);
    }

    pub fn result_trained(&mut self) {
        let size = self.var.size();
        let steps = self.var.step();

        let mut state = State(na::DMatrix::zeros(size, steps));
        state.0.row_mut(0).fill(1.0);
        state
            .0
            .rows_mut(1, self.var.input_dimension)
            .copy_from(&self.var.input().0);
        let mut trained = Variable::new(
            state,
            Output(na::DMatrix::zeros(self.var.output_dimension, steps)),
            self.var.input_dimension,
            self.var.reservoir_dimension,
            self.var.output_dimension,
        );

        let u_0 = trained.input_with_bias_t(0).clone();
        let x_0 = (self.activator)(self.input_weight.clone() * u_0);
        trained.replace_reservoir_t(0, &x_0);

        for step in 0..steps - 1 {
            let x_t = trained.reservoir_t(step);
            let u_t1 = trained.input_with_bias_t(step + 1);
            let x_t1 = self.sequence_reservoir_variables(&u_t1, &x_t);

            trained.replace_reservoir_t(step + 1, &x_t1);
        }

        trained.output.0 = self.output_weight.clone() * trained.state.0.clone();

        self.trained = Some(trained);
    }

    fn update_reservoir_variables(&mut self, step: usize) {
        let x_t = self.var.reservoir_t(step);
        let u_t1 = self.var.input_with_bias_t(step + 1);
        let x_t1 = self.sequence_reservoir_variables(&u_t1, &x_t);

        self.append_reservoir_variables(&x_t1, step + 1);
    }

    /// x(t+1) = (1-a) * x(t) + a * tanh( W_in * u(t+1) + W_res * x(t) )
    /// `a` is leak rate
    fn sequence_reservoir_variables(
        &self,
        u_t1: &na::DVector<f64>,
        x_t: &na::DVector<f64>,
    ) -> na::DVector<f64> {
        let x_hat = (self.activator)(
            self.input_weight.clone() * u_t1 + self.reservoir_weight.clone() * x_t.clone(),
        );

        (1. - self.leak_rate) * x_t + self.leak_rate * x_hat
    }

    fn append_reservoir_variables(&mut self, x_t: &na::DVector<f64>, step: usize) {
        self.var.replace_reservoir_t(step, x_t);
    }

    fn update_output_weight(&mut self, regularization_coefficient: f64) {
        let size = self.var.size();
        let identity = regularization_coefficient * na::DMatrix::<f64>::identity(size, size);

        let z_z = self.var.state.0.clone() * self.var.state.0.clone().transpose();

        self.output_weight = self.var.input().0.clone()
            * self.var.state.0.clone().transpose()
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

    const TEST_STEP: usize = 10;

    #[test]
    fn test_train() {
        let mut esn = esn_for_test(TEST_STEP);

        esn.train(REGULARIZATION_COEFFICIENT);
    }

    #[test]
    fn test_predict() {}

    #[test]
    fn test_result_trained() {}

    #[test]
    fn test_update_reservoir_variables() {}

    #[test]
    fn test_sequence_reservoir_variables() {}

    #[test]
    fn test_append_reservoir_variables() {}

    #[test]
    fn test_update_output_weight() {
        let mut esn = esn_for_test(TEST_STEP);

        esn.input_weight = input_for_test(TEST_STEP).0;
        let element = vec![1.; TEST_STEP * TEST_STEP];
        esn.var.state.0 = na::DMatrix::from_vec(TEST_STEP + 1, TEST_STEP + 1, element);

        esn.update_output_weight(REGULARIZATION_COEFFICIENT);
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
