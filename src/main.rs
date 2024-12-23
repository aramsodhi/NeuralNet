mod neural_network;
mod math;

use neural_network::neural_network::NeuralNetwork;
use math::math::Vector;

fn main() {
    let mut neural_net: NeuralNetwork = NeuralNetwork::new(2, 10, 1, 0.02);

    let inputs: Vec<Vector> = vec![
        Vector::from_data(vec![0.0, 0.0]),
        Vector::from_data(vec![0.0, 1.0]),
        Vector::from_data(vec![1.0, 0.0]),
        Vector::from_data(vec![1.0, 1.0]),
    ];

    let targets: Vec<Vector> = vec![
        Vector::from_data(vec![0.0]),
        Vector::from_data(vec![1.0]),
        Vector::from_data(vec![1.0]),
        Vector::from_data(vec![0.0]),
    ];

    for (inputs, target) in inputs.iter().zip(targets.iter()) {
        let output = neural_net.forward_propagation(&inputs);
        println!("Input: {:?}, Expected: {:?}, Prediction: {:?}", inputs.data, target.data[0], output.data[0]);
    }

    println!();
    neural_net.train(&inputs, &targets, 1000);
    println!();

    for (inputs, target) in inputs.iter().zip(targets.iter()) {
        let output = neural_net.forward_propagation(&inputs);
        println!("Input: {:?}, Expected: {:?}, Prediction: {:?}", inputs.data, target.data[0], output.data[0]);
    }
}
