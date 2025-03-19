package com.neuralnet.core;

public interface Activation {
    double activate(double input);
    double derivative(double input);
} 