package org.ml4bull.algorithm;

// (0, 1]
public class SoftmaxFunction implements ActivationFunction {

    @Override
    public double[] activate(double[] layerOutput) {
        double denominator = 0;
        for (double lO : layerOutput) {
            denominator += Math.exp(lO);
        }
        double[] result = new double[layerOutput.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = Math.exp(layerOutput[i]) / denominator;
        }
        return result;
    }

    @Override
    public double activate(double value) {
        throw new UnsupportedOperationException("For softmax we need full layout output.");
    }

    @Override
    public double[] derivative(double[] lastInput) {
        double[] a = new double[lastInput.length];
        for (int s = 0; s < a.length; s++)
            a[s] = (1 - lastInput[s]) * lastInput[s];
        return a;
    }
}
