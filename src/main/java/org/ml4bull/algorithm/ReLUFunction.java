package org.ml4bull.algorithm;

public class ReLUFunction implements ActivationFunction {

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
        return value <= .0 ? .0 : value;
    }

    @Override
    public double[] derivative(double[] lastInput) {
        double[] a = new double[lastInput.length];
        for (int s = 0; s < a.length; s++)
            a[s] = lastInput[s] <= .0 ? 0 : 1.;
        return a;
    }
}
