package org.ml4bull.algorithm;

// (0, 1)
public class SigmoidFunction implements ActivationFunction {

    @Override
    public double activate(double value) {
        return 1 / (1 + Math.exp(-value));
    }

    @Override
    public double[] derivative(double[] lastInput) {
        double[] a = new double[lastInput.length];
        for (int s = 0; s < a.length; s++)
            a[s] = (1 - lastInput[s]) * lastInput[s];
        return a;
    }
}
