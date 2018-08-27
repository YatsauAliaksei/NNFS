package org.ml4bull.algorithm;

import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.util.Factory;
import org.ml4bull.util.MLUtils;

public class DropoutRegularization {

    private ActivationFunction stepFunction;

    public DropoutRegularization(double threshold) {
        stepFunction = new StepFunction(threshold);
    }

    public double[] dropout(double[] input) {
        double[] randomDropout = MLUtils.getRandomDropout(input.length);
        randomDropout = stepFunction.activate(randomDropout);
//        randomDropout[0] = 1; // omit bias

        MatrixOperations mo = Factory.getMatrixOperations();
        return mo.scalarMultiply(input, randomDropout);
    }
}
