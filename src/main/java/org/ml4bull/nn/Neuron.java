package org.ml4bull.nn;


import lombok.Getter;
import lombok.Setter;
import org.ml4bull.util.Factory;
import org.ml4bull.util.MLUtils;

public class Neuron {
    private double[] features;
    @Getter
    @Setter
    private double[] weights;
    @Getter
    private double[] weightsError;

    public void setFeatures(double[] features) {
        if (weights == null) {
            weights = MLUtils.getRandomWeights(features.length);
        }
        this.features = features;
    }

    public double calculate() {
        return Factory.getMatrixOperations().multiply(weights, features);
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
