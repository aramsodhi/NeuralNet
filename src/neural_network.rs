pub mod neural_network {
    use crate::math::math::{Matrix, Vector};

    pub struct NeuralNetwork {
        input_size: usize,
        hidden_size: usize,
        output_size: usize,

        weights_input_hidden: Matrix,
        weights_hidden_output: Matrix,

        bias_hidden: Vector,
        bias_output: Vector,
    }
    
    impl NeuralNetwork {
        pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
            let weights_input_hidden: Matrix = Matrix::random(hidden_size, input_size);
            let weights_hidden_output: Matrix = Matrix::random(output_size, hidden_size);

            // originally was using zeroes, consider switching back for simplicity?
            let bias_hidden: Vector = Vector::random(hidden_size);
            let bias_output: Vector = Vector::random(output_size);

            NeuralNetwork {
                input_size,
                hidden_size,
                output_size,
                weights_input_hidden,
                weights_hidden_output,
                bias_hidden,
                bias_output,
            }
        }

        pub fn forward_propagation(&self, input: &Vector) -> Vector {
            let hidden_input: Vector = self.weights_input_hidden.multiply_vector(input) + self.bias_hidden.clone();
            let hideen_output: Vector = hidden_input.sigmoid();

            let output_input: Vector = self.weights_hidden_output.multiply_vector(&hideen_output) + self.bias_output.clone();
            output_input.sigmoid()
        }
    }
}