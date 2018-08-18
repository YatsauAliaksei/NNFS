package org.ml4bull.nn;


import com.google.common.util.concurrent.AtomicDoubleArray;
import lombok.Getter;
import lombok.Setter;
import org.ml4bull.util.Factory;
import org.ml4bull.util.MLUtils;

import java.util.stream.IntStream;

public class Neuron {
    private ThreadLocal<double[]> features = new ThreadLocal<>();
    @Getter
    @Setter
    private double[] weights;
    @Getter
    private AtomicDoubleArray weightsError;

    public void setFeatures(double[] features) {
        if (weights == null) {
            synchronized (this) {
                if (weights == null) {
                    weightsError = new AtomicDoubleArray(features.length);
                    weights = MLUtils.getRandomWeights(features.length); // basic random weights.
                    weights[0] = 1; // bias initial
                }
            }
        }
        this.features.set(features);
    }

    public double calculate() {
        return Factory.getMatrixOperations().multiply(weights, features.get());
    }

    public void addWeightsError(double[] we) {
        for (int i = 0; i < weightsError.length(); i++) {
            weightsError.addAndGet(i, we[i]);
        }
    }

    public void resetErrorWeights() {
        weightsError = new AtomicDoubleArray(weights.length);
//        IntStream.range(0, weightsError.length()).forEach(i -> weightsError.set(i, 0));
    }
}
