pub mod math {
    use std::ops::{Add, Sub, Mul};
    use assert_approx_eq::assert_approx_eq;

    #[derive(Debug, Clone)]
    pub struct Vector {
        pub data: Vec<f64>,
    }

    impl PartialEq for Vector {
        fn eq(&self, other: &Self) -> bool {
            self.data == other.data
        }
    }

    #[derive(Debug, Clone)]
    pub struct Matrix {
        pub data: Vec<Vec<f64>>,
    }

    impl PartialEq for Matrix {
        fn eq(&self, other: &Self) -> bool {
            self.data == other.data
        }
    }

    impl Vector {
        pub fn zeroes(size: usize) -> Self {
            Vector {
                data: vec![0.0; size],
            }
        }

        pub fn random(size: usize) -> Self {
            let data: Vec<f64> = (0..size).map(|_: usize| rand::random::<f64>()).collect();
            
            Vector { data }
        }

        pub fn from_data(data: Vec<f64>) -> Self {
            Vector { data }
        }

        pub fn len(&self) -> usize {
            self.data.len()
        }

        pub fn dot(&self, other: &Vector) -> f64 {
            assert_eq!(self.len(), other.len());

            self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum()
        }

        pub fn sigmoid(&self) -> Vector {
            let data: Vec<f64> = self.data.iter().map(|x: &f64| 1.0 / (1.0 + (-x).exp())).collect();

            Vector { data }
        }

        pub fn sigmoid_derivative(&self) -> Vector {
            let sigmoid_values: Vector = self.sigmoid();

            let ones_arr: Vec<f64> = vec![1.0; self.len()];
            let ones: Vector = Vector::from_data(ones_arr);
    
            // sigmoid(x) * (1 - sigmoid(x))
            sigmoid_values.clone() * (ones - sigmoid_values)
        }
    }

    impl Matrix {
        pub fn zeroes(rows: usize, cols: usize) -> Self {
            Matrix {
                data: vec![vec![0.0; cols]; rows],
            }
        }

        pub fn random(rows: usize, cols: usize) -> Self {
            let data: Vec<Vec<f64>> = (0..rows)
                .map(|_: usize| (0..cols).map(|_: usize| rand::random::<f64>()).collect())
                .collect();

            Matrix { data }
        }

        pub fn from_data(data: Vec<Vec<f64>>) -> Self {
            Matrix { data }
        }

        pub fn rows(&self) -> usize {
            self.data.len()
        }

        pub fn cols(&self) -> usize {
            self.data[0].len()
        }

        pub fn transpose(&self) -> Self {
            // not yet implemented
            self.clone()
        }

        pub fn multiply_vector(&self, vector: &Vector) -> Vector {
            let mut result: Vector = Vector::zeroes(self.rows());

            // add dimensions check?
            for entry_index in 0..self.rows() {
                result.data[entry_index] = self.data[entry_index].iter().zip(&vector.data).map(|(a, b)| a * b).sum();
            }

            result
        }
    }

    impl Add for Vector {
        type Output = Vector;

        fn add(self, other: Vector) -> Vector {
            assert_eq!(self.len(), other.len());

            let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect();
            Vector { data }
        }
    }

    impl Sub for Vector {
        type Output = Vector;

        fn sub(self, other: Vector) -> Vector {
            assert_eq!(self.len(), other.len());

            let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a - b).collect();
            Vector { data }
        }
    }

    impl Mul<f64> for Vector {
        type Output = Vector;

        fn mul(self, scalar: f64) -> Vector {
            let data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
            Vector { data }
        }
    }

    // NOT A DOT PRODUCT
    impl Mul for Vector {
        type Output = Vector;

        fn mul(self, other: Vector) -> Vector {
            let data: Vec<f64> = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();
            Vector { data }
        }
    }

    impl Mul for Matrix {
        type Output = Matrix;

        fn mul(self, other: Matrix) -> Matrix {
            let rows: usize = self.rows();
            let cols: usize = other.cols();

            let mut result: Matrix = Matrix::zeroes(rows, cols);

            for row_index in 0..rows {
                for col_index in 0..cols {
                    result.data[row_index][col_index] = (0..self.cols()).map(|k: usize| self.data[row_index][k] * other.data[k][col_index]).sum();
                }
            }

            result
        }
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use math::{Matrix, Vector};

    use super::*;

        // -------------------- VECTOR TEST CASES --------------------
        #[test]
        fn test_zeroes_vector_creation() {
            let v1: Vector = Vector::zeroes(3);

            assert_eq!(v1.len(), 3);
        }

        #[test]
        fn test_random_vector_creation() {
            let v1: Vector = Vector::random(5);

            assert_eq!(v1.len(), 5);
        }

        #[test]
        fn test_vector_dot() {
            let v1: Vector = Vector::from_data(vec![1.0, 2.0, 3.0]);
            let v2: Vector = Vector::from_data(vec![4.0, 5.0, 6.0]);
            
            let result: f64 = v1.dot(&v2);
            let expected: f64 = 32.0;

            assert_eq!(result, expected);
        }

        #[test]
        fn test_vector_add() {
            let v1: Vector = Vector::from_data(vec![1.0, 2.0, 3.0]);
            let v2: Vector = Vector::from_data(vec![4.0, 5.0, 6.0]);

            let result: Vector = v1 + v2;
            let expected: Vector = Vector::from_data(vec![5.0, 7.0, 9.0]);
            
            assert_eq!(result, expected);
        }

        #[test]
        fn test_vector_scalar_multiplication() {
            let v1: Vector = Vector::from_data(vec![1.0, 2.0, 3.0]);

            let result: Vector = v1 * 2.0;
            let expected: Vector = Vector::from_data(vec![2.0, 4.0, 6.0]);

            assert_eq!(result, expected);
        }

        #[test]
        fn test_vector_sigmoid() {
            let v1: Vector = Vector::from_data(vec![2.0, 4.0, -2.0]);

            let result: Vector = v1.sigmoid();

            assert_approx_eq::assert_approx_eq!(result.data[0], 0.880797077978, 1e-10f64);
            assert_approx_eq::assert_approx_eq!(result.data[1], 0.982013790038, 1e-10f64);
            assert_approx_eq::assert_approx_eq!(result.data[2], 0.119202922022, 1e-10f64);
        }

        #[test]
        fn test_vector_sigmoid_derivative() {
            let v1: Vector = Vector::from_data(vec![2.0, 4.0, -2.0]);

            let result: Vector = v1.sigmoid_derivative();
 
            assert_approx_eq::assert_approx_eq!(result.data[0], 0.104993585404, 1e-10f64);
            assert_approx_eq::assert_approx_eq!(result.data[1], 0.017662706213, 1e-10f64);
            assert_approx_eq::assert_approx_eq!(result.data[2], 0.104993585404, 1e-10f64);

        }

        // -------------------- MATRIX TEST CASES --------------------

        #[test]
        fn test_matrix_multiply_vector() {
            let m1: Matrix = Matrix::from_data(vec![
                vec![1.0, 2.0],
                vec![3.0, 4.0],
            ]);

            let v1: Vector = Vector::from_data(vec![1.0, 1.0]);
    
            let result: Vector = m1.multiply_vector(&v1);
            let expected: Vector = Vector::from_data(vec![3.0, 7.0]);

            assert_eq!(result, expected);
        }

        #[test]
        fn test_matrix_dimensions() {
            let m1: Matrix = Matrix::from_data(vec![
                vec![1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0],
            ]);

            assert_eq!(m1.rows(), 2);
            assert_eq!(m1.cols(), 3);
        }

        #[test]
        fn test_zeroes_matrix_creation() {
            let m1: Matrix = Matrix::zeroes(3, 3);

            // check dimensions
            assert_eq!(m1.rows(), 3);
            assert_eq!(m1.cols(), 3);
        }

        #[test]
        fn test_random_matrix_creation() {
            let m1: Matrix = Matrix::random(3, 3);

            // check dimensions
            assert_eq!(m1.rows(), 3);
            assert_eq!(m1.cols(), 3);
        }

        #[test]
        fn test_large_matrix_multiplication() {
            let m1: Matrix = Matrix::from_data(vec![
                vec![4.0, 12.0, 5.0, 34.0, 107.0],
                vec![0.0, 5.0, 2.0, 3.0, 12.0],
                vec![5.0, 3.0, 2.0, 2.0, 8.0],
                vec![45.0, 0.0, 41.0, 0.0, 99.0],
            ]);

            let m2: Matrix = Matrix::from_data(vec![
                vec![6.0, 5.0],
                vec![2.0, 1.0],
                vec![21.0, 57.0],
                vec![19.0, 0.0],
                vec![1.0, 11.0],
            ]);

            let result: Matrix = m1 * m2;

            let expected: Matrix = Matrix::from_data(vec![
                vec![906.0, 1494.0],
                vec![121.0, 251.0],
                vec![124.0, 230.0],
                vec![1230.0, 3651.0],
            ]);

            assert_eq!(result, expected);
        }

}