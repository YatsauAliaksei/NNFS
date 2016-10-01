package org.ml4bull.nn;


import org.ml4bull.util.Factory;

import java.util.Random;

public class Neuron {
    private double[] features;
    private double[] weights;
    private double[] weightsError;

    public void setFeatures(double[] features) {
        if (weights == null) {
            Random random = new Random();
            weights = random.doubles(features.length, -0.5, 0.5).toArray();
        }
        this.features = features;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public double calculate() {
        return Factory.getMatrixOperations().multiply(weights, features);
    }

    public double[] getWeightsError() {
        return weightsError;
    }

    public void addWeightsError(double[] we) {
        if (this.weightsError == null) {
            this.weightsError = we;
        } else {
            for (int i = 0; i < weightsError.length; i++) {
                weightsError[i] += we[i];
            }
        }
    }

    public void resetErrorWeights() {
        weightsError = null;
    }
}
