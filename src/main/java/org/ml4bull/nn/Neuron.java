package org.ml4bull.nn;


import org.ml4bull.util.Factory;

import java.util.Random;

public class Neuron {
    private double[] features;
    private double[] weights;
    private double lastA;

    public void setFeatures(double[] features) {
        if (weights == null) {
            Random random = new Random();
            weights = new double[features.length];
            for (int i = 0; i < weights.length; i++) {
                weights[i] = random.nextDouble();
            }
        }
        this.features = features;
    }

    public double[] getWeights() {
        return weights;
    }

    public double calculate() {
        lastA = Factory.getMatrixOperations().multiply(weights, features);
        return lastA;
    }
}
