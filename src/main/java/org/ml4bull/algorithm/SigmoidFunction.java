package org.ml4bull.algorithm;

import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.util.Factory;

public class SigmoidFunction implements ActivationFunction {

    @Override
    public double activate(double[] theta, double[] features) {
        MatrixOperations mo = Factory.getMatrixOperations();
        double power = mo.multiply(theta, features);

        return 1 / (1 + Math.exp(-power));
    }
}
