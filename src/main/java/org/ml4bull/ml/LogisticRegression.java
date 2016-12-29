package org.ml4bull.ml;

import lombok.Builder;
import org.ml4bull.algorithm.SigmoidFunction;
import org.ml4bull.algorithm.optalg.GradientDescent;
import org.ml4bull.annotation.Untested;
import org.ml4bull.nn.MultiLayerPerceptron;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.data.Printer;

@Untested
public class LogisticRegression implements SupervisedAlgorithm {
    private final MultiLayerPerceptron mlp;

    @Builder
    public LogisticRegression(int inputLength, int batchSize) {
        mlp = MultiLayerPerceptron.builder()
                .input(inputLength)
                .output(1)
                .optAlg(GradientDescent.builder().batchSize(batchSize).build())
                .outActFunc(new SigmoidFunction()).build();
    }

    @Override
    public double[][] classify(DataSet dataSet, boolean classifyParallel, Printer printer) {
        return mlp.classify(dataSet, classifyParallel, printer);
    }

    @Override
    public double train(DataSet dataSet, boolean trainParallel) {
        return mlp.train(dataSet, trainParallel);
    }
}
