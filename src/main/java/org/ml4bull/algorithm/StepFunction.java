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

    @Override
    public double[] activate(double[] layerOutput) {
        double[] r = new double[layerOutput.length];
        for (int i = 0; i < layerOutput.length; i++) {
            r[i] = activate(layerOutput[i]);
        }
        return r;
    }
}
