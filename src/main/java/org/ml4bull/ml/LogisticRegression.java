package org.ml4bull.ml;

import org.ml4bull.annotation.Untested;
import org.ml4bull.nn.Neuron;
import org.ml4bull.nn.data.Data;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.data.Printer;

import java.util.List;

@Untested
public class LogisticRegression implements SupervisedAlgorithm {
    final private Neuron neuron;

    public LogisticRegression(int fNumber) {
        neuron = new Neuron();
    }

    @Override
    public double[][] classify(DataSet dataSet, Printer printer) {
        return new double[0][];
    }

    @Override
    public double train(DataSet dataSet) {
        return 0;
    }

    @Override
    public double train(List<Data> dataSet) {
        return 0;
    }
}
