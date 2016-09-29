package org.ml4bull.algorithm;

public class SigmoidFunction implements ActivationFunction {

    @Override
    public double[] activate(double[] layerOutput) {
        double[] result = new double[layerOutput.length];
        for (int i = 0; i < layerOutput.length; i++) {
            result[i] = activate(layerOutput[i]);
        }
        return result;
    }

    @Override
    public double activate(double value) {
        return 1 / (1 + Math.exp(-value));
    }
}
