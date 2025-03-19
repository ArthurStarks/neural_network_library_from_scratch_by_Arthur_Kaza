package com.neuralnet.benchmarks;

import com.neuralnet.core.*;
import com.neuralnet.activations.*;
import com.neuralnet.optimizers.*;
import com.neuralnet.training.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
public class NetworkBenchmark {
    private Network network;
    private DataSet trainingData;
    private DataSet validationData;
    private Optimizer optimizer;
    private Trainer trainer;

    @Setup
    public void setup() {
        // Create network
        optimizer = new Adam();
        network = new Network(optimizer);
        
        // Add layers
        network.addLayer(new Layer(784, new ReLU(), optimizer));     // Input layer (MNIST)
        network.addLayer(new Layer(128, new ReLU(), optimizer));     // Hidden layer
        network.addLayer(new Layer(64, new ReLU(), optimizer));      // Hidden layer
        network.addLayer(new Layer(10, new Softmax(), optimizer));   // Output layer

        // Create synthetic MNIST-like data
        trainingData = new DataSet(784, 10);
        validationData = new DataSet(784, 10);
        
        // Generate 1000 training samples
        for (int i = 0; i < 1000; i++) {
            double[] input = new double[784];
            double[] target = new double[10];
            
            // Generate random input (simulating MNIST pixels)
            for (int j = 0; j < 784; j++) {
                input[j] = Math.random();
            }
            
            // Generate random target (one-hot encoded)
            int label = (int)(Math.random() * 10);
            target[label] = 1.0;
            
            trainingData.addSample(input, target);
        }
        
        // Generate 200 validation samples
        for (int i = 0; i < 200; i++) {
            double[] input = new double[784];
            double[] target = new double[10];
            
            for (int j = 0; j < 784; j++) {
                input[j] = Math.random();
            }
            
            int label = (int)(Math.random() * 10);
            target[label] = 1.0;
            
            validationData.addSample(input, target);
        }

        // Create trainer
        trainer = new Trainer(network, trainingData, validationData, 32, 1);
    }

    @Benchmark
    public void testForwardPass() {
        double[] input = new double[784];
        for (int i = 0; i < 784; i++) {
            input[i] = Math.random();
        }
        network.forward(input);
    }

    @Benchmark
    public void testBackwardPass() {
        double[] input = new double[784];
        double[] target = new double[10];
        for (int i = 0; i < 784; i++) {
            input[i] = Math.random();
        }
        int label = (int)(Math.random() * 10);
        target[label] = 1.0;
        
        network.forward(input);
        network.backward(target);
    }

    @Benchmark
    public void testMiniBatchTraining() {
        trainer.train();
    }

    @Test
    public void runBenchmarks() throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(NetworkBenchmark.class.getSimpleName())
                .forks(1)
                .warmupIterations(5)
                .measurementIterations(5)
                .build();

        new Runner(opt).run();
    }
} 