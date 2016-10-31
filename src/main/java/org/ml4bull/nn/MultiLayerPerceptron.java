package org.ml4bull.nn;

import org.ml4bull.algorithm.ActivationFunction;
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

public class MultiLayerPerceptron implements SupervisedNeuralNetwork {

    private final NeuronLayer inputLayer;
    private final NeuronLayer outputLayer;
    private final List<NeuronLayer> perceptronLayers;
    private double regularizationRate = 8e-2;
    private double learningRate = 12e-1;

    public MultiLayerPerceptron(int input, int output, ActivationFunction outActFunc) {
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
    public double[][] test(DataSet dataSet, Printer printer) {
        double[][] result = new double[dataSet.getInput().length][];
        IntStream.range(0, dataSet.getInput().length).forEach(i -> {
                    result[i] = process(dataSet.getInput()[i]);
                    printer.print(i, result[i], dataSet.getOutput()[i]);
                }
        );
        return result;
    }

    @Override
    public double[] process(double[] data) {
        double[] v = inputLayer.forwardPropagation(data);

        for (NeuronLayer perceptronLayer : perceptronLayers) {
            v = perceptronLayer.forwardPropagation(v);
        }
        return v;
    }

    @Override
    public double train(DataSet dataSet) {
        return train(dataSet.getInput(), dataSet.getOutput());
    }

    @Override
    public double train(double[][] data, double[][] expected) {
        double error = 0;
        final int dataSize = data.length;

        for (int i = 0; i < dataSize; i++) {
            // predict
            double[] calcY = process(data[i]);

            // Back propagation for hidden layers
            List<NeuronLayer> revList = new ArrayList<>(perceptronLayers);
            Collections.reverse(revList);

            double[] errorOut = expected[i];
            for (NeuronLayer aRevList : revList) {
                errorOut = aRevList.backPropagation(errorOut);
            }

            error = calculateCurrentItemError(calcY, expected[i], error);
        }

        weightsErrorProcessing(dataSize);

        return -error / dataSize;
    }

    private void weightsErrorProcessing(int dataSize) {
        perceptronLayers.forEach(l -> l.getNeurons().forEach(neuron -> {
            double[] weights = neuron.getWeights();
            double[] weightsError = neuron.getWeightsError();

            for (int w = 0; w < weightsError.length; w++) {
                // gradient descent
                double regularization = w == 0 ? 0 : regularizationRate * weights[w];
                weights[w] -= learningRate * (weightsError[w] + regularization) / dataSize;
            }

            neuron.resetErrorWeights();
        }));
    }

    private double calculateCurrentItemError(double[] calculated, double[] expected, double e) {
        for (int i = 0; i < calculated.length; i++) {
            e += expected[i] * log2(calculated[i]) + (1 - expected[i]) * log2(1 - calculated[i]);
        }
        return e;
    }

    public MultiLayerPerceptron setRegularizationRate(double regularizationRate) {
        this.regularizationRate = regularizationRate;
        return this;
    }

    public MultiLayerPerceptron setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }
}
