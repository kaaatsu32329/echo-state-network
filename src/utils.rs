use nalgebra as na;

pub fn relu(x: na::DVector<f64>) -> na::DVector<f64> {
    x.map(|x| x.max(0.0))
}

pub fn tanh(x: na::DVector<f64>) -> na::DVector<f64> {
    x.map(|x| x.tanh())
}
