use nalgebra as na;

#[derive(Debug, Clone)]
pub struct Input(pub na::DMatrix<f64>);

// pub type Ut = Vec<Input>;
pub type Ut = Input;

/// Reservoir variable
/// Multi layer Column vector
#[derive(Debug, Clone)]
pub struct Reservoir(pub na::DMatrix<f64>);

// pub type Xt = Vec<Reservoir>;
pub type Xt = Reservoir;

#[derive(Debug, Clone)]
pub struct Output(pub na::DMatrix<f64>);

// pub type Yt = Vec<Output>;
pub type Yt = Output;

#[derive(Debug, Clone)]
pub struct State(pub na::DMatrix<f64>);

/// [1; Ut; Xt]
pub type Zt = State;
