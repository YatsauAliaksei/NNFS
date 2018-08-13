package org.ml4bull.nn;

import lombok.Builder;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.nn.data.Data;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.data.Printer;
import org.ml4bull.nn.layer.InputNeuronLayer;
import org.ml4bull.nn.layer.NeuronLayer;
import org.ml4bull.nn.layer.OutputNeuronLayer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Semaphore;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import static org.ml4bull.util.MathUtils.log2;

@Slf4j
public class MultiLayerPerceptron implements SupervisedNeuralNetwork {

    private final NeuronLayer inputLayer;
    private final NeuronLayer outputLayer;
    private final List<NeuronLayer> perceptronLayers;
    private OptimizationAlgorithm optAlg;
    private Semaphore semaphore;

    @Builder
    public MultiLayerPerceptron(int input, int output, OptimizationAlgorithm optAlg, ActivationFunction outActFunc) {
        this.optAlg = optAlg;
        this.inputLayer = new InputNeuronLayer(input);
        this.outputLayer = new OutputNeuronLayer(output, outActFunc);
        this.perceptronLayers = new ArrayList<>();
        this.perceptronLayers.add(outputLayer);
        this.semaphore = new Semaphore(optAlg.getBatchSize());
    }

    @Override
    public MultiLayerPerceptron addHiddenLayer(NeuronLayer hiddenLayer) {
        perceptronLayers.add(perceptronLayers.size() - 1, hiddenLayer);
        return this;
    }

    @Override
    public double[][] classify(DataSet dataSet, boolean isParallel, Printer printer) {
        double[][] result = new double[dataSet.getInput().length][];
        IntStream is = IntStream.range(0, dataSet.getInput().length);

        if (isParallel)
            is = is.parallel();

        is.forEach(i -> {
                    result[i] = process(dataSet.getInput()[i]);
                    printer.print(i, result[i], dataSet.getOutput()[i]);
                }
        );
        return result;
    }

    public double[] classify(double[] data) {
        return process(data);
    }

    private double[] process(double[] data) {
        double[] v = inputLayer.forwardPropagation(data);
        for (NeuronLayer layer : perceptronLayers) {
            v = layer.forwardPropagation(v);
        }
        return v;
    }

    @Override
    public double train(DataSet dataSet, boolean isParallel) {
        return train(dataSet.getInput(), dataSet.getOutput(), isParallel);
    }

    public double train(List<Data> dataSet, boolean isParallel) {
        final int dataSize = dataSet.size();

        DoubleStream ds = dataSet.stream()
                .mapToDouble(data -> calculateAndGetItemError(data.getInput(), data.getOutput()));
        double error = run(isParallel, ds);

        return -error / dataSize;
    }

    private double train(double[][] data, double[][] expected, boolean isParallel) {
        final int dataSize = data.length;
        if (dataSize == 0) {
            log.error("DataSize couldn't be empty");
            throw new RuntimeException("DataSize couldn't be empty");
        }

        DoubleStream ds = IntStream.range(0, dataSize)
                .mapToDouble(i -> calculateAndGetItemError(data[i], expected[i]));
        double error = run(isParallel, ds);

        return -error / dataSize;
    }

    private double run(boolean isParallel, DoubleStream ds) {
        if (isParallel)
            ds = ds.parallel().unordered();

        return ds.sum();
    }

    @SneakyThrows(InterruptedException.class)
    private double calculateAndGetItemError(double[] input, double[] output) {
        semaphore.acquire();

        // predict
        double[] calcY = process(input);

        // Back propagation for hidden layers
        List<NeuronLayer> revList = new ArrayList<>(perceptronLayers);
        Collections.reverse(revList);

        double[] errorOut = output;
        for (NeuronLayer aRevList : revList) {
            errorOut = aRevList.backPropagation(errorOut);
        }
        tryUpdateWeights();

        return itemCostFunction(calcY, output);
    }

    private void tryUpdateWeights() {
        if (!optAlg.isLimitReached()) return;

        optimize();
        semaphore.release(optAlg.getBatchSize());
    }

    private void optimize() {
        perceptronLayers.forEach(l -> l.optimizeWeights(optAlg));
    }

    private double itemCostFunction(double[] calculated, double[] expected) {
        return IntStream.range(0, calculated.length)
                .mapToDouble(i ->
                        expected[i] * log2(calculated[i]) + (1 - expected[i]) * log2(1 - calculated[i])
                ).parallel().unordered().sum();
    }
}
