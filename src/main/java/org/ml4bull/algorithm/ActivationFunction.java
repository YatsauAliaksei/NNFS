package org.ml4bull.algorithm;

public interface ActivationFunction {

    double activate(double value);

    double[] derivative(double[] lastInput);

    default double[] activate(double[] layerOutput) {
        double[] result = new double[layerOutput.length];
        for (int i = 0; i < layerOutput.length; i++) {
            result[i] = activate(layerOutput[i]);
        }
        return result;
    }
}
