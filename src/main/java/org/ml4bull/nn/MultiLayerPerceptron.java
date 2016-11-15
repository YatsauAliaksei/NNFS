package org.ml4bull.nn;

import lombok.Builder;
import lombok.extern.slf4j.Slf4j;
import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.OptimizationAlgorithm;
import org.ml4bull.nn.data.Data;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.data.Printer;
import org.ml4bull.nn.layer.InputNeuronLayer;
import org.ml4bull.nn.layer.NeuronLayer;
import org.ml4bull.nn.layer.OutputNeuronLayer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

import static org.ml4bull.util.MathUtils.log2;

@Slf4j
public class MultiLayerPerceptron implements SupervisedNeuralNetwork {

    private final NeuronLayer inputLayer;
    private final NeuronLayer outputLayer;
    private final List<NeuronLayer> perceptronLayers;
    private OptimizationAlgorithm optAlg;

    @Builder
    public MultiLayerPerceptron(int input, int output, OptimizationAlgorithm optAlg, ActivationFunction outActFunc) {
        this.optAlg = optAlg;
        this.inputLayer = new InputNeuronLayer(input);
        this.outputLayer = new OutputNeuronLayer(output, outActFunc);
        this.perceptronLayers = new ArrayList<>();
        this.perceptronLayers.add(outputLayer);
    }

    @Override
    public MultiLayerPerceptron addHiddenLayer(NeuronLayer hiddenLayer) {
        perceptronLayers.add(perceptronLayers.size() - 1, hiddenLayer);
        return this;
    }

    @Override
    public double[][] classify(DataSet dataSet, Printer printer) {
        double[][] result = new double[dataSet.getInput().length][];
        IntStream.range(0, dataSet.getInput().length)
                .parallel()
                .forEach(i -> {
                            result[i] = process(dataSet.getInput()[i]);
                            printer.print(i, result[i], dataSet.getOutput()[i]);
                        }
                );
        return result;
    }

    private double[] process(double[] data) {
        double[] v = inputLayer.forwardPropagation(data);
        return perceptronLayers.stream()
                .reduce(v, (d, nl) -> nl.forwardPropagation(d), (v1, v2) -> v1);
    }

    @Override
    public double train(DataSet dataSet) {
        return train(dataSet.getInput(), dataSet.getOutput());
    }

    public double train(List<Data> dataSet) {
        final int dataSize = dataSet.size();

        double error = dataSet.stream()
                .mapToDouble(data -> calculateAndGetItemError(data.getInput(), data.getOutput()))
//                .parallel()
                .sum();

        updateWeights();

        return -error / dataSize;
    }

    private double train(double[][] data, double[][] expected) {
        final int dataSize = data.length;

        double error = IntStream.range(0, dataSize)
                .mapToDouble(i -> calculateAndGetItemError(data[i], expected[i]))
//                .parallel()
                .sum();

        updateWeights();

        return -error / dataSize;
    }

    private double calculateAndGetItemError(double[] input, double[] output) {
        // predict
        double[] calcY = process(input);

        // Back propagation for hidden layers
        List<NeuronLayer> revList = new ArrayList<>(perceptronLayers);
        Collections.reverse(revList);

        double[] errorOut = output;
        for (NeuronLayer aRevList : revList) {
            errorOut = aRevList.backPropagation(errorOut);
        }
        updateWeights();

        double error = itemCostFunction(calcY, output);
        return error;
    }

    private void updateWeights() {
        if (!optAlg.isLimitReached()) return;

        perceptronLayers.stream()
                .flatMap(l -> l.getNeurons().stream())
//                .parallel()
                .forEach(neuron -> {
                    double[] weights = neuron.getWeights();
                    double[] weightsError = neuron.getWeightsError();

                    optAlg.optimizeWeights(weights, weightsError);

                    neuron.resetErrorWeights();
                });
    }


    private double itemCostFunction(double[] calculated, double[] expected) {
        double error = .0;
        for (int i = 0; i < calculated.length; i++) {
            error += expected[i] * log2(calculated[i]) + (1 - expected[i]) * log2(1 - calculated[i]);
        }
        return error;
    }
}
