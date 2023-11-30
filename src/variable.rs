use crate::*;
use nalgebra as na;

#[derive(Debug, Clone)]
pub struct Variable {
    /// (1 + Nu + Nx) x T Matrix
    /// [1; Ut; Xt]
    /// [1; Input; Reservoir]
    pub state: Zt,
    /// Ny x T Matrix
    pub output: Yt,
    /// Nu
    pub input_dimension: usize,
    /// Nx
    pub reservoir_dimension: usize,
    /// Ny
    pub output_dimension: usize,
}

impl Variable {
    pub fn new(
        state: State,
        output: Output,
        input_dimension: usize,
        reservoir_dimension: usize,
        output_dimension: usize,
    ) -> Self {
        Self {
            state,
            output,
            input_dimension: input_dimension,
            reservoir_dimension: reservoir_dimension,
            output_dimension: output_dimension,
        }
    }

    pub fn input(&self) -> Ut {
        let inner = self
            .state
            .0
            .view((1, 0), (self.input_dimension, self.state.0.ncols()))
            .clone();
        Input(inner.into())
    }

    pub fn input_with_bias(&self) -> Ut {
        let inner = self
            .state
            .0
            .view((0, 0), (1 + self.input_dimension, self.state.0.ncols()))
            .clone();
        Input(inner.into())
    }

    pub fn input_t(&self, step: usize) -> na::DVector<f64> {
        self.state
            .0
            .column(step)
            .rows(1, self.input_dimension)
            .into()
    }

    pub fn input_with_bias_t(&self, step: usize) -> na::DVector<f64> {
        self.state
            .0
            .column(step)
            .rows(0, 1 + self.input_dimension)
            .into()
    }

    pub fn reservoir(&self) -> Xt {
        let inner = self
            .state
            .0
            .view(
                (1 + self.input_dimension, 0),
                (self.reservoir_dimension, self.state.0.ncols()),
            )
            .clone();
        Reservoir(inner.into())
    }

    pub fn reservoir_t(&self, step: usize) -> na::DVector<f64> {
        self.state
            .0
            .column(step)
            .rows(1 + self.input_dimension, self.reservoir_dimension)
            .into()
    }

    pub fn replace_reservoir_t(&mut self, step: usize, x_t: &na::DVector<f64>) {
        self.state
            .0
            .column_mut(step)
            .rows_mut(1 + self.input_dimension, self.reservoir_dimension)
            .copy_from(x_t);
    }

    pub fn state_t(&self, step: usize) -> na::DVector<f64> {
        self.state.0.column(step).into()
    }

    pub fn replace_state_t(&mut self, step: usize, z_t: &na::DVector<f64>) {
        self.state.0.column_mut(step).copy_from(z_t);
    }

    pub fn output_t(&self, step: usize) -> na::DVector<f64> {
        self.output.0.column(step).into()
    }

    pub fn replace_output_t(&mut self, step: usize, y_t: &na::DVector<f64>) {
        self.output.0.column_mut(step).copy_from(y_t);
    }

    pub fn step(&self) -> usize {
        self.state.0.ncols()
    }

    /// 1 + Nu + Nx
    pub fn size(&self) -> usize {
        1 + self.input_dimension + self.reservoir_dimension
    }
}
