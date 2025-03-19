package com.neuralnet.loss;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MSETest {
    private final MSE mse = new MSE();

    @Test
    void testComputeWithEqualArrays() {
        double[] predictions = {1.0, 2.0, 3.0};
        double[] targets = {1.0, 2.0, 3.0};
        double expected = 0.0;
        assertEquals(expected, mse.compute(predictions, targets), 1e-10);
    }

    @Test
    void testComputeWithDifferentArrays() {
        double[] predictions = {1.0, 2.0, 3.0};
        double[] targets = {2.0, 3.0, 4.0};
        double expected = 1.0; // ((1-2)² + (2-3)² + (3-4)²) / 3 = 1.0
        assertEquals(expected, mse.compute(predictions, targets), 1e-10);
    }

    @Test
    void testComputeWithNegativeValues() {
        double[] predictions = {-1.0, -2.0, -3.0};
        double[] targets = {-2.0, -3.0, -4.0};
        double expected = 1.0; // ((-1+2)² + (-2+3)² + (-3+4)²) / 3 = 1.0
        assertEquals(expected, mse.compute(predictions, targets), 1e-10);
    }

    @Test
    void testComputeWithDifferentLengths() {
        double[] predictions = {1.0, 2.0};
        double[] targets = {1.0, 2.0, 3.0};
        assertThrows(IllegalArgumentException.class, () -> mse.compute(predictions, targets));
    }

    @Test
    void testDerivativeWithEqualArrays() {
        double[] predictions = {1.0, 2.0, 3.0};
        double[] targets = {1.0, 2.0, 3.0};
        double[] expected = {0.0, 0.0, 0.0};
        assertArrayEquals(expected, mse.derivative(predictions, targets), 1e-10);
    }

    @Test
    void testDerivativeWithDifferentArrays() {
        double[] predictions = {1.0, 2.0, 3.0};
        double[] targets = {2.0, 3.0, 4.0};
        double[] expected = {-2.0/3.0, -2.0/3.0, -2.0/3.0}; // 2 * (prediction - target) / n
        assertArrayEquals(expected, mse.derivative(predictions, targets), 1e-10);
    }

    @Test
    void testDerivativeWithDifferentLengths() {
        double[] predictions = {1.0, 2.0};
        double[] targets = {1.0, 2.0, 3.0};
        assertThrows(IllegalArgumentException.class, () -> mse.derivative(predictions, targets));
    }
} 