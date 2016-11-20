package org.ml4bull.ml;

import org.ml4bull.nn.data.Data;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.data.Printer;

import java.util.List;

public interface SupervisedAlgorithm {

    double[][] classify(DataSet dataSet, boolean classifyParallel, Printer printer);

    default double[][] classify(DataSet dataSet, boolean classifyParallel) {
        return classify(dataSet, classifyParallel, (i, calc, ideal) -> {
        });
    }

    double train(DataSet dataSet, boolean trainParallel);

    default double train(List<Data> dataSet, boolean trainParallel) {
        double[][] data = new double[dataSet.size()][];
        double[][] expected = new double[dataSet.size()][];
        for (int i = 0; i < dataSet.size(); i++) {
            data[i] = dataSet.get(i).getInput();
            expected[i] = dataSet.get(i).getOutput();
        }
        return train(new DataSet(data, expected), trainParallel);
    }

}
