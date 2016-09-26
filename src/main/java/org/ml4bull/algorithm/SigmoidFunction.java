package org.ml4bull.algorithm;

public class SigmoidFunction implements ActivationFunction {

    @Override
    public double activate(double value) {
        return 1 / (1 + Math.exp(-value));
    }
}
