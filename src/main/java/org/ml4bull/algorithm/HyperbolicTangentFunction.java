package org.ml4bull.algorithm;

// (-1, 1)
public class HyperbolicTangentFunction implements ActivationFunction {

    @Override
    public double activate(double value) {
        return Math.tanh(value);
    }

    @Override
    public double[] derivative(double[] lastInput) {
        double[] a = new double[lastInput.length];
        for (int s = 0; s < a.length; s++) {
            a[s] = 1 - Math.pow(lastInput[s], 2);
        }
        return a;
    }
}
