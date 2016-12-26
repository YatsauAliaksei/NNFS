package org.ml4bull.algorithm;

public class LiniarFunction implements ActivationFunction {
    @Override
    public double[] activate(double[] layerOutput) {
        return layerOutput;
    }

    @Override
    public double activate(double value) {
        return value;
    }

    @Override
    public double[] derivative(double[] lastInput) {
        return lastInput;
    }
}
