package org.ml4bull.algorithm;

public interface ActivationFunction {

    double[] activate(double[] layerOutput);

    double activate(double value);
}
