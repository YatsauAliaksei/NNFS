package org.ml4bull.nn;

import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.data.Printer;
import org.ml4bull.nn.layer.InputNeuronLayer;
import org.ml4bull.nn.layer.NeuronLayer;
import org.ml4bull.nn.layer.OutputNeuronLayer;
import org.ml4bull.util.Factory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class MultiLayerPerceptron implements SupervisedNeuralNetwork {

    private final NeuronLayer inputLayer;
    private final NeuronLayer outputLayer;
    private final List<NeuronLayer> perceptronLayers;
    private double[][] lastResults;

    public MultiLayerPerceptron(int input, int output, ActivationFunction outputLayerActivationFunction) {
        this.inputLayer = new InputNeuronLayer(input);
        this.outputLayer = new OutputNeuronLayer(output, outputLayerActivationFunction);
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
        for (int i = 0; i < dataSet.getInput().length; i++) {
            result[i] = process(dataSet.getInput()[i]);
            printer.print(i, result[i], dataSet.getOutput()[i]);
        }
        return result;
    }

    @Override
    public double[] process(double[] data) {
        lastResults = new double[perceptronLayers.size() + 1][];

        double[] v = inputLayer.forwardPropagation(data);
        lastResults[0] = v;

        for (int i = 0; i < perceptronLayers.size(); i++) {
            lastResults[i] = v;
            v = perceptronLayers.get(i).forwardPropagation(v);
        }
        return v;
    }

    @Override
    public double train(DataSet dataSet) {
        return train(dataSet.getInput(), dataSet.getOutput());
    }

    /**
     * @param data
     * @param expected
     * @return error.
     */
    @Override
    public double train(double[][] data, double[][] expected) {
        double error = 0;
        MatrixOperations mo = Factory.getMatrixOperations();

        double[][][] deltas = new double[perceptronLayers.size()][][]; // layer/neuron/theta

        for (int i = 0; i < data.length; i++) {
            // predict
            double[] calcY = process(data[i]);
            // calculate out error start point for back propagation.
            double[] errorOut = new double[expected[i].length];
            for (int j = 0; j < expected[i].length; j++) {
                errorOut[j] = calcY[j] - expected[i][j];
            }

            // Back propagation for hidden layers
            ArrayList<NeuronLayer> revList = new ArrayList<>(perceptronLayers);
            Collections.reverse(revList);

            for (int layer = 0; layer < revList.size(); layer++) {
                int index = revList.size() - 1 - layer;

                // compute current weight error
                double[][] dl = new double[errorOut.length][lastResults[index].length + 1]; // neuron/theta

                for (int l = 0; l < dl.length; l++) {
                    dl[l][0] = errorOut[l];
                    for (int e = 1; e < dl[l].length; e++) {
                        dl[l][e] = errorOut[l] * lastResults[index][e - 1];
                    }
                }

                deltas[index] = mo.sum(deltas[index], dl);

                if (index == 0) break;

                //********************************************************************************************
                // Compute next error
                errorOut = revList.get(layer).backPropagation(errorOut);
            }

            mo.roundMatrix(calcY, 0.5); // for error calculation. todo: cost function

            if (!Arrays.equals(calcY, expected[i]))
                error++; // Very naive implementation. Should be changed to cost function. todo
        }

        for (int l = 0; l < deltas.length; l++) {
            for (int n = 0; n < deltas[l].length; n++) {
                Neuron neuron = perceptronLayers.get(l).getNeurons().get(n);
                double[] weights = neuron.getWeights();

                for (int w = 0; w < deltas[l][n].length; w++) {
                    double regularization = w == 0 ? 0 : 0.01 * weights[w];
                    weights[w] -= 0.9 * (deltas[l][n][w] + regularization) / data.length;
                }
            }
        }

        return error / expected.length;
    }
}
