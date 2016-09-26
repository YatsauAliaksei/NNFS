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
import java.util.List;

public class StupidPerceptron implements Perceptron {

    private final NeuronLayer inputLayer;
    private final NeuronLayer outputLayer;
    private final List<NeuronLayer> hiddenLayers;
    private double[][] hlLastResult;

    public StupidPerceptron(int input, int output, ActivationFunction af) {
        this.inputLayer = new InputNeuronLayer(input);
        this.outputLayer = new OutputNeuronLayer(output, af);
        this.hiddenLayers = new ArrayList<>();
    }

    public StupidPerceptron addHiddenLayer(NeuronLayer hiddenLayer) {
        hiddenLayers.add(hiddenLayer);
        return this;
    }

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
        hlLastResult = new double[hiddenLayers.size()][];

        double[] v = inputLayer.forwardPropagation(data);
        for (int i = 0; i < hiddenLayers.size(); i++) {
            v = hiddenLayers.get(i).forwardPropagation(v);
            hlLastResult[i] = v;
        }
        v = outputLayer.forwardPropagation(v);
        return v;
    }

    public double train(DataSet dataSet) {
        return train(dataSet.getInput(), dataSet.getOutput());
    }

    /**
     * @param data
     * @param expected
     * @return error.
     */
    public double train(double[][] data, double[][] expected) {
        double error = 0;
        MatrixOperations mo = Factory.getMatrixOperations();

        double[][][] deltas = new double[hiddenLayers.size() + 1][][]; // layer/neuron/theta

        for (int i = 0; i < data.length; i++) {
            // predict
            double[] calcY = process(data[i]);
            // calculate out error i.e. back propagation.
            double[] errorOut = new double[expected[i].length];
            for (int j = 0; j < expected[i].length; j++) {
                errorOut[j] = calcY[j] - expected[i][j];
            }

            hiddenLayers.add(outputLayer);
            double[] errorH = errorOut;

            // Back propagation for hidden layers
            for (int layer = 0; layer < hiddenLayers.size(); layer++) {
                int index = hiddenLayers.size() - 1 - layer;
                NeuronLayer neuronLayer = hiddenLayers.get(index);
                List<Neuron> neurons = neuronLayer.getNeurons();


                double[] previousA;
                if (index == 0) {
                    previousA = data[i];
                } else {
                    previousA = hlLastResult[index - 1];
                }

                // compute current weight error
                double[] f = errorH;
                double[][] dl = new double[f.length][previousA.length + 1]; // neuron/theta

                for (int l = 0; l < dl.length; l++) {
                    dl[l][0] = f[l];
                    for (int e = 1; e < dl[l].length; e++) {
                        dl[l][e] = f[l] * previousA[e - 1];
                    }
                }

                deltas[index] = mo.sum(deltas[index], dl);

                if (index == 0) break;

                //********************************************************************************************
                // Compute next error
                // get weights
                double[][] theta = new double[neurons.size()][];
                for (int s = 0; s < neurons.size(); s++) {
                    double[] weights = neurons.get(s).getWeights();
                    theta[s] = Arrays.copyOfRange(weights, 1, weights.length);
                }

                // calculating next layer error
                double[][] thetaT = mo.transpose(theta);
                double[] e = mo.multiplySingleDim(thetaT, errorH);
                double[] a = new double[hlLastResult[index - 1].length];
                double[] currentError = new double[e.length];

                for (int s = 0; s < a.length; s++)
                    a[s] = (1 - hlLastResult[index - 1][s]) * hlLastResult[index - 1][s];

                for (int d = 0; d < currentError.length; d++)
                    currentError[d] = e[d] * a[d];

                errorH = currentError;
            }
            hiddenLayers.remove(outputLayer);
            mo.roundMatrix(calcY, 0.5);

            if (!Arrays.equals(calcY, expected[i]))
                error++;
        }

        hiddenLayers.add(outputLayer);

        for (int l = 0; l < deltas.length; l++) {
            for (int n = 0; n < deltas[l].length; n++) {
                Neuron neuron = hiddenLayers.get(l).getNeurons().get(n);
                double[] weights = neuron.getWeights();

                for (int w = 0; w < deltas[l][n].length; w++) {
                    double regularization = w == 0 ? 0 : 0.01 * weights[w];
                    weights[w] -= 0.9 * (deltas[l][n][w] + regularization) / data.length;
                }
            }
        }

        hiddenLayers.remove(outputLayer);

        return error / expected.length;
    }
}
