package org.ml4bull.nn;

import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.data.Printer;
import org.ml4bull.nn.layer.NeuronLayer;

public interface SupervisedNeuralNetwork {
    MultiLayerPerceptron addHiddenLayer(NeuronLayer hiddenLayer);

    double[][] test(DataSet dataSet, Printer printer);

    double[] process(double[] inValues);

    double train(DataSet dataSet);

    double train(double[][] data, double[][] expected);
}
