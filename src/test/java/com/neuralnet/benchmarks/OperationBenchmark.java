package com.neuralnet.benchmarks;

import com.neuralnet.core.*;
import com.neuralnet.activations.*;
import com.neuralnet.optimizers.*;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.junit.jupiter.api.Test;

import java.util.concurrent.TimeUnit;
import java.util.Random;

@BenchmarkMode({Mode.AverageTime, Mode.Throughput})
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
public class OperationBenchmark {
    private Network network;
    private DataSet dataset;
    private Optimizer optimizer;
    private Random random;
    private double[] input;
    private double[] target;

    @Setup
    public void setup() {
        optimizer = new Adam();
        random = new Random(42);

        // Créer un réseau avec plusieurs couches
        network = new Network(optimizer);
        network.addLayer(new Layer(100, new ReLU(), optimizer));
        network.addLayer(new Layer(50, new ReLU(), optimizer));
        network.addLayer(new Layer(10, new Softmax(), optimizer));

        // Créer un ensemble de données
        dataset = new DataSet(100, 10);
        for (int i = 0; i < 1000; i++) {
            double[] input = new double[100];
            double[] target = new double[10];
            for (int j = 0; j < 100; j++) {
                input[j] = random.nextDouble();
            }
            target[random.nextInt(10)] = 1.0;
            dataset.addSample(input, target);
        }

        // Créer des données de test
        input = new double[100];
        target = new double[10];
        for (int i = 0; i < 100; i++) {
            input[i] = random.nextDouble();
        }
        target[random.nextInt(10)] = 1.0;
    }

    @Benchmark
    public void testForwardPass() {
        network.forward(input);
    }

    @Benchmark
    public void testBackwardPass() {
        network.forward(input);
        network.backward(target);
    }

    @Benchmark
    public void testWeightUpdate() {
        network.forward(input);
        network.backward(target);
        network.updateWeights(0.01);
    }

    @Benchmark
    public void testMiniBatchProcessing() {
        List<double[]> batchInputs = dataset.getInputs().subList(0, 32);
        List<double[]> batchTargets = dataset.getTargets().subList(0, 32);
        
        for (int i = 0; i < batchInputs.size(); i++) {
            network.forward(batchInputs.get(i));
            network.backward(batchTargets.get(i));
        }
        network.updateWeights(0.01);
    }

    @Benchmark
    public void testActivationFunctions() {
        ReLU relu = new ReLU();
        Sigmoid sigmoid = new Sigmoid();
        Tanh tanh = new Tanh();
        Softmax softmax = new Softmax();

        for (int i = 0; i < 1000; i++) {
            double x = random.nextDouble() * 2 - 1;
            relu.activate(x);
            sigmoid.activate(x);
            tanh.activate(x);
        }
        softmax.activate(input);
    }

    @Benchmark
    public void testOptimizerUpdates() {
        Connection connection = new Connection(new Neuron(), new Neuron(), 1.0);
        Neuron neuron = new Neuron();
        
        optimizer.initialize(connection);
        optimizer.initialize(neuron);
        
        connection.setDeltaWeight(random.nextDouble());
        neuron.setDeltaBias(random.nextDouble());
        
        optimizer.updateWeights(connection, 0.01);
        optimizer.updateBias(neuron, 0.01);
    }

    @Benchmark
    public void testGradientComputation() {
        network.forward(input);
        network.backward(target);
        
        // Accéder aux gradients pour s'assurer qu'ils sont calculés
        for (Layer layer : network.getLayers()) {
            for (Neuron neuron : layer.getNeurons()) {
                neuron.getDeltaBias();
                for (Connection conn : neuron.getInputConnections()) {
                    conn.getDeltaWeight();
                }
            }
        }
    }

    @Benchmark
    public void testLayerConnections() {
        Layer layer1 = new Layer(100, new ReLU(), optimizer);
        Layer layer2 = new Layer(50, new ReLU(), optimizer);
        
        layer1.connectTo(layer2);
        
        // Accéder aux connexions pour s'assurer qu'elles sont créées
        for (Neuron neuron : layer1.getNeurons()) {
            neuron.getOutputConnections().size();
        }
        for (Neuron neuron : layer2.getNeurons()) {
            neuron.getInputConnections().size();
        }
    }

    @Test
    public void runBenchmarks() throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(OperationBenchmark.class.getSimpleName())
                .forks(1)
                .warmupIterations(5)
                .measurementIterations(5)
                .build();

        new Runner(opt).run();
    }
} 