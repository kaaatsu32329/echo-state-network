use plotters::prelude::*;

use crate::EchoStateNetwork;

pub trait Plot {
    fn plot(&self, name: &str) -> Result<(), Box<dyn std::error::Error>>;
}

impl Plot for EchoStateNetwork {
    /// First element of row
    fn plot(&self, name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let input = self.var.input().0.clone();
        let trained = self.trained.as_ref().unwrap().output.0.clone();

        let mut xs = Vec::new();
        let ys_trained = trained
            .column_iter()
            .map(|m| m[0] as f32)
            .collect::<Vec<_>>();
        let ys_input = input.column_iter().map(|m| m[0] as f32).collect::<Vec<_>>();

        let ys_predict = self
            .predicted
            .as_ref()
            .unwrap()
            .output
            .0
            .column_iter()
            .map(|m| m[0] as f32)
            .collect::<Vec<_>>();

        for i in 0..input.ncols() {
            xs.push(i as f32 * self.delta as f32);
        }
        for i in input.ncols()..input.ncols() + ys_predict.len() {
            xs.push(i as f32 * self.delta as f32);
        }

        let predict_start = input.ncols() as f32 * self.delta as f32;

        let width = 1280;
        let height = 720;
        let path = format!("{}{}{}", "./graph/", name, ".png");
        let root = BitMapBackend::new(&path, (width, height)).into_drawing_area();

        root.fill(&WHITE)?;

        let (y_trained_min, y_trained_max) = ys_trained
            .iter()
            .fold((f32::NAN, f32::NAN), |(m, n), v| (v.min(m), v.max(n)));

        let (y_input_min, y_input_max) = ys_input
            .iter()
            .fold((f32::NAN, f32::NAN), |(m, n), v| (v.min(m), v.max(n)));

        let (y_predict_min, y_predict_max) = ys_predict
            .iter()
            .fold((f32::NAN, f32::NAN), |(m, n), v| (v.min(m), v.max(n)));

        let y_min = y_trained_min.min(y_input_min).min(y_predict_min);
        let y_max = y_trained_max.max(y_input_max).max(y_predict_max);

        let font = ("sans-serif", 32);

        let mut chart;

        if y_min.is_sign_negative() {
            chart = ChartBuilder::on(&root)
                .caption(name, font.into_font())
                .margin(10)
                .x_label_area_size(16)
                .y_label_area_size(42)
                .build_cartesian_2d(
                    (*xs.first().unwrap() - 0.1)..(*xs.last().unwrap() + 0.1),
                    (y_min - 0.1)..(y_max + 0.1),
                )?;
        } else {
            chart = ChartBuilder::on(&root)
                .caption(name, font.into_font())
                .margin(10)
                .x_label_area_size(16)
                .y_label_area_size(42)
                .build_cartesian_2d(
                    (*xs.first().unwrap() - 0.1)..(*xs.last().unwrap() + 0.1),
                    0f32..(y_max + 0.1),
                )?;
        }

        chart.configure_mesh().draw()?;

        let input_line_series =
            LineSeries::new(xs.iter().zip(ys_input.iter()).map(|(x, y)| (*x, *y)), &BLUE)
                .point_size(2);
        let trained_line_series = LineSeries::new(
            xs.iter().zip(ys_trained.iter()).map(|(x, y)| (*x, *y)),
            &RED,
        )
        .point_size(2);
        let predict_line_series = LineSeries::new(
            xs.iter()
                .zip(ys_predict.iter())
                .map(|(x, y)| (*x + predict_start, *y)),
            &GREEN,
        )
        .point_size(2);
        chart.draw_series(input_line_series)?;
        chart.draw_series(trained_line_series)?;
        chart.draw_series(predict_line_series)?;

        Ok(())
    }
}
