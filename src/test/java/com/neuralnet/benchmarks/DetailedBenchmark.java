package com.neuralnet.benchmarks;

import com.neuralnet.core.*;
import com.neuralnet.activations.*;
import com.neuralnet.optimizers.*;
import com.neuralnet.training.*;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.junit.jupiter.api.Test;

import java.util.concurrent.TimeUnit;
import java.util.Random;

@BenchmarkMode({Mode.AverageTime, Mode.Throughput})
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
public class DetailedBenchmark {
    private Network smallNetwork;
    private Network mediumNetwork;
    private Network largeNetwork;
    private DataSet smallDataset;
    private DataSet mediumDataset;
    private DataSet largeDataset;
    private Optimizer optimizer;
    private Random random;

    @Setup
    public void setup() {
        optimizer = new Adam();
        random = new Random(42);

        // Create networks of different sizes
        smallNetwork = createNetwork(10, 5, 1);
        mediumNetwork = createNetwork(100, 50, 10);
        largeNetwork = createNetwork(1000, 500, 100);

        // Create datasets of different sizes
        smallDataset = createDataset(100, 10, 1);
        mediumDataset = createDataset(1000, 100, 10);
        largeDataset = createDataset(10000, 1000, 100);
    }

    private Network createNetwork(int inputSize, int hiddenSize, int outputSize) {
        Network network = new Network(optimizer);
        network.addLayer(new Layer(inputSize, new ReLU(), optimizer));
        network.addLayer(new Layer(hiddenSize, new ReLU(), optimizer));
        network.addLayer(new Layer(outputSize, new Softmax(), optimizer));
        return network;
    }

    private DataSet createDataset(int size, int inputSize, int outputSize) {
        DataSet dataset = new DataSet(inputSize, outputSize);
        for (int i = 0; i < size; i++) {
            double[] input = new double[inputSize];
            double[] target = new double[outputSize];
            for (int j = 0; j < inputSize; j++) {
                input[j] = random.nextDouble();
            }
            target[random.nextInt(outputSize)] = 1.0;
            dataset.addSample(input, target);
        }
        return dataset;
    }

    @Benchmark
    public void testSmallNetworkForward() {
        double[] input = new double[10];
        for (int i = 0; i < 10; i++) {
            input[i] = random.nextDouble();
        }
        smallNetwork.forward(input);
    }

    @Benchmark
    public void testMediumNetworkForward() {
        double[] input = new double[100];
        for (int i = 0; i < 100; i++) {
            input[i] = random.nextDouble();
        }
        mediumNetwork.forward(input);
    }

    @Benchmark
    public void testLargeNetworkForward() {
        double[] input = new double[1000];
        for (int i = 0; i < 1000; i++) {
            input[i] = random.nextDouble();
        }
        largeNetwork.forward(input);
    }

    @Benchmark
    public void testSmallNetworkBackward() {
        double[] input = smallDataset.getInputs().get(0);
        double[] target = smallDataset.getTargets().get(0);
        smallNetwork.forward(input);
        smallNetwork.backward(target);
    }

    @Benchmark
    public void testMediumNetworkBackward() {
        double[] input = mediumDataset.getInputs().get(0);
        double[] target = mediumDataset.getTargets().get(0);
        mediumNetwork.forward(input);
        mediumNetwork.backward(target);
    }

    @Benchmark
    public void testLargeNetworkBackward() {
        double[] input = largeDataset.getInputs().get(0);
        double[] target = largeDataset.getTargets().get(0);
        largeNetwork.forward(input);
        largeNetwork.backward(target);
    }

    @Benchmark
    public void testSmallBatchTraining() {
        Trainer trainer = new Trainer(smallNetwork, smallDataset, null, 10, 1);
        trainer.train();
    }

    @Benchmark
    public void testMediumBatchTraining() {
        Trainer trainer = new Trainer(mediumNetwork, mediumDataset, null, 32, 1);
        trainer.train();
    }

    @Benchmark
    public void testLargeBatchTraining() {
        Trainer trainer = new Trainer(largeNetwork, largeDataset, null, 64, 1);
        trainer.train();
    }

    @Benchmark
    public void testActivationFunctions() {
        ReLU relu = new ReLU();
        Sigmoid sigmoid = new Sigmoid();
        Tanh tanh = new Tanh();
        Softmax softmax = new Softmax();

        double[] input = new double[100];
        for (int i = 0; i < 100; i++) {
            input[i] = random.nextDouble() * 2 - 1;
        }

        for (int i = 0; i < 100; i++) {
            relu.activate(input[i]);
            sigmoid.activate(input[i]);
            tanh.activate(input[i]);
        }
        softmax.activate(input);
    }

    @Benchmark
    public void testOptimizerUpdates() {
        SGD sgd = new SGD();
        Adam adam = new Adam();
        RMSprop rmsprop = new RMSprop();

        Connection connection = new Connection(new Neuron(), new Neuron(), 1.0);
        Neuron neuron = new Neuron();

        sgd.initialize(connection);
        adam.initialize(connection);
        rmsprop.initialize(connection);

        sgd.initialize(neuron);
        adam.initialize(neuron);
        rmsprop.initialize(neuron);

        connection.setDeltaWeight(random.nextDouble());
        neuron.setDeltaBias(random.nextDouble());

        sgd.updateWeights(connection, 0.01);
        adam.updateWeights(connection, 0.01);
        rmsprop.updateWeights(connection, 0.01);

        sgd.updateBias(neuron, 0.01);
        adam.updateBias(neuron, 0.01);
        rmsprop.updateBias(neuron, 0.01);
    }

    @Test
    public void runBenchmarks() throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(DetailedBenchmark.class.getSimpleName())
                .forks(1)
                .warmupIterations(5)
                .measurementIterations(5)
                .build();

        new Runner(opt).run();
    }
} 