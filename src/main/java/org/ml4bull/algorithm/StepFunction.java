package org.ml4bull.algorithm;

public class StepFunction implements ActivationFunction {

    private double threshold = 0.5;

    public StepFunction() {

    }

    public StepFunction(double treshold) {
        this.threshold = treshold;
    }

    @Override
    public double activate(double value) {
        if (value >= threshold) {
            return 1;
        } else
            return 0;
    }

    public double[] activate(double[] value) {
        double[] r = new double[value.length];
        for (int i = 0; i < value.length; i++) {
            if (value[i] >= threshold) {
                r[i] = 1;
            } else
                r[i] = 0;
        }
        return r;
    }
}
