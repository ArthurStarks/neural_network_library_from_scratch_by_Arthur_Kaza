package com.neuralnet.activations;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ActivationTest {
    private static final double EPSILON = 1e-10;

    @Test
    void testReLU() {
        ReLU relu = new ReLU();
        
        // Test positive input
        assertEquals(5.0, relu.activate(5.0), EPSILON);
        
        // Test negative input
        assertEquals(0.0, relu.activate(-5.0), EPSILON);
        
        // Test zero input
        assertEquals(0.0, relu.activate(0.0), EPSILON);
        
        // Test derivative
        assertEquals(1.0, relu.derivative(5.0), EPSILON);
        assertEquals(0.0, relu.derivative(-5.0), EPSILON);
    }

    @Test
    void testSigmoid() {
        Sigmoid sigmoid = new Sigmoid();
        
        // Test positive input
        assertEquals(0.9933071490757153, sigmoid.activate(5.0), EPSILON);
        
        // Test negative input
        assertEquals(0.0066928509242848554, sigmoid.activate(-5.0), EPSILON);
        
        // Test zero input
        assertEquals(0.5, sigmoid.activate(0.0), EPSILON);
        
        // Test derivative
        assertEquals(0.006648056670790033, sigmoid.derivative(5.0), EPSILON);
        assertEquals(0.006648056670790033, sigmoid.derivative(-5.0), EPSILON);
    }

    @Test
    void testTanh() {
        Tanh tanh = new Tanh();
        
        // Test positive input
        assertEquals(0.9999092042625951, tanh.activate(5.0), EPSILON);
        
        // Test negative input
        assertEquals(-0.9999092042625951, tanh.activate(-5.0), EPSILON);
        
        // Test zero input
        assertEquals(0.0, tanh.activate(0.0), EPSILON);
        
        // Test derivative
        assertEquals(0.00018158323094387872, tanh.derivative(5.0), EPSILON);
        assertEquals(0.00018158323094387872, tanh.derivative(-5.0), EPSILON);
    }

    @Test
    void testSoftmax() {
        Softmax softmax = new Softmax();
        
        // Test array input
        double[] input = {1.0, 2.0, 3.0};
        double[] output = softmax.activate(input);
        
        // Check sum equals 1
        double sum = 0.0;
        for (double value : output) {
            sum += value;
        }
        assertEquals(1.0, sum, EPSILON);
        
        // Check all values are positive
        for (double value : output) {
            assertTrue(value > 0);
        }
        
        // Check values are in correct order
        assertTrue(output[2] > output[1]);
        assertTrue(output[1] > output[0]);
    }

    @Test
    void testLeakyReLU() {
        LeakyReLU leakyRelu = new LeakyReLU();
        
        // Test positive input
        assertEquals(5.0, leakyRelu.activate(5.0), EPSILON);
        
        // Test negative input
        assertEquals(-0.5, leakyRelu.activate(-5.0), EPSILON);
        
        // Test zero input
        assertEquals(0.0, leakyRelu.activate(0.0), EPSILON);
        
        // Test derivative
        assertEquals(1.0, leakyRelu.derivative(5.0), EPSILON);
        assertEquals(0.1, leakyRelu.derivative(-5.0), EPSILON);
    }

    @Test
    void testELU() {
        ELU elu = new ELU();
        
        // Test positive input
        assertEquals(5.0, elu.activate(5.0), EPSILON);
        
        // Test negative input
        assertEquals(-0.9932620530009145, elu.activate(-5.0), EPSILON);
        
        // Test zero input
        assertEquals(0.0, elu.activate(0.0), EPSILON);
        
        // Test derivative
        assertEquals(1.0, elu.derivative(5.0), EPSILON);
        assertEquals(0.006737946999085467, elu.derivative(-5.0), EPSILON);
    }
} 