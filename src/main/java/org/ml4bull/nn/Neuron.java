package org.ml4bull.nn;


public class Neuron {
    private double[] features;
    private double[] weights;
    private boolean isBias;

    public double[] getFeatures() {
        return features;
    }

    public void setFeatures(double[] features) {
        this.features = features;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public boolean isBias() {
        return isBias;
    }
}
