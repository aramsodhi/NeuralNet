pub mod neural_network {    use std::f32::consts::E;

use crate::math::math::{Matrix, Vector};

    pub struct NeuralNetwork {
        input_size: usize,
        hidden_size: usize,
        output_size: usize,

        weights_input_hidden: Matrix,
        weights_hidden_output: Matrix,

        bias_hidden: Vector,
        bias_output: Vector,

        learning_rate: f64,
    }
    
    impl NeuralNetwork {
        pub fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
            let weights_input_hidden: Matrix = Matrix::random(hidden_size, input_size);
            let weights_hidden_output: Matrix = Matrix::random(output_size, hidden_size);

            let bias_hidden: Vector = Vector::zeroes(hidden_size);
            let bias_output: Vector = Vector::zeroes(output_size);

            NeuralNetwork {
                input_size,
                hidden_size,
                output_size,
                weights_input_hidden,
                weights_hidden_output,
                bias_hidden,
                bias_output,
                learning_rate,
            }
        }

        pub fn prediction(&self, input: &Vector) -> Vector {
            let hidden_input: Vector = self.weights_input_hidden.multiply_vector(input) + self.bias_hidden.clone();
            let hidden_input: Vector = hidden_input.sigmoid();

            let output_input: Vector = self.weights_hidden_output.multiply_vector(&hidden_input) + self.bias_output.clone();
            output_input.sigmoid()
        }

        pub fn forward_propagation(&self, input: &Vector) -> Vector {
            let hidden_output: Vector = self.weights_input_hidden.multiply_vector(input) + self.bias_hidden.clone();
            let hidden_output: Vector = hidden_output.sigmoid();

            let output_input: Vector = self.weights_hidden_output.multiply_vector(&hidden_output) + self.bias_output.clone();
            output_input.sigmoid()
        }

        pub fn back_propagation(&mut self, input: &Vector, target: Vector) {
            let network_output: Vector = self.forward_propagation(input);

            let output_error: Vector = network_output - target;
            let output_derivative: Vector = output_error.sigmoid_derivative();
            let output_gradient: Vector = output_error * output_derivative;

            let hidden_output: Vector = self.weights_input_hidden.multiply_vector(input).sigmoid();
            let output_gradient_matrix: Matrix = output_gradient.outer_product(&hidden_output);

            self.weights_hidden_output = self.weights_hidden_output.clone() - output_gradient_matrix * self.learning_rate;
            self.bias_output = self.bias_output.clone() - output_gradient.clone() * self.learning_rate;

            let hidden_error: Vector = self.weights_hidden_output.transpose().multiply_vector(&output_gradient);
            let hidden_output_derivative: Vector = hidden_output.sigmoid_derivative();
            let hidden_gradient: Vector = hidden_error * hidden_output_derivative;

            let hidden_gradient_matrix: Matrix = hidden_gradient.outer_product(input);

            self.weights_input_hidden = self.weights_input_hidden.clone() - hidden_gradient_matrix * self.learning_rate;
            self.bias_hidden = self.bias_hidden.clone() - hidden_gradient * self.learning_rate;
        }

        pub fn train(&mut self, inputs: &Vec<Vector>, targets: &Vec<Vector>, epochs: u64) {
            for epoch in 0..epochs {
                let mut total_loss: f64 = 0.0;

                for (input, target) in inputs.iter().zip(targets.iter()) {
                    let network_output: Vector = self.forward_propagation(input);

                    total_loss += (network_output - target.clone()).sum();

                    self.back_propagation(input, target.clone());
                }

                total_loss /= inputs.len() as f64;

                if epoch % 500 == 0 {
                    println!("Epoch {}: Loss = {}", epoch, total_loss);
                }
            }
        }
    }
}