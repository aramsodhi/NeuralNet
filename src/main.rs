mod neural_network;
mod math;

use neural_network::neural_network::NeuralNetwork;
use math::math::Vector;

fn main() {
    let n: NeuralNetwork = NeuralNetwork::new(3, 3, 2);

    let input: Vector = Vector::from_data(vec![10.0, 1.0, -3.0]);
    let output: Vector = n.forward_propagation(&input);

    for elem in output.data.iter() {
        println!("output = {}", elem);
    }
}
