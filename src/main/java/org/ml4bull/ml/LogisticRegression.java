package org.ml4bull.ml;

import org.ml4bull.annotation.Untested;
import org.ml4bull.nn.Neuron;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.data.Printer;

@Untested
public class LogisticRegression implements SupervisedAlgorithm {
    final private Neuron neuron;

    public LogisticRegression(int fNumber) {
        neuron = new Neuron();
    }

    @Override
    public double[][] classify(DataSet dataSet, boolean classifyParallel, Printer printer) {
        return new double[0][];
    }

    @Override
    public double train(DataSet dataSet, boolean trainParallel) {
        return 0;
    }
}
