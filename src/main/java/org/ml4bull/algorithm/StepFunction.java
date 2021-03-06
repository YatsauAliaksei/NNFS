package org.ml4bull.algorithm;

public class StepFunction implements ActivationFunction {

    private double threshold = 0.5;

    public StepFunction() {

    }

    public StepFunction(double threshold) {
        this.threshold = threshold;
    }

    @Override
    public double activate(double value) {
        if (value >= threshold) {
            return 1;
        } else
            return 0;
    }

    @Override
    public double[] derivative(double[] lastInput) {
        throw new UnsupportedOperationException("Derivative for step function doesn't exist.");
    }
}
