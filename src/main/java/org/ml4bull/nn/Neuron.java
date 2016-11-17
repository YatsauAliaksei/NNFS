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
    volatile private double[] weights;
    @Getter
    private AtomicDoubleArray weightsError;

    public void setFeatures(double[] features) {
        if (weights == null) {
            synchronized (this) {
                if (weights != null) return;
                weightsError = new AtomicDoubleArray(features.length);
                weights = MLUtils.getRandomWeights(features.length);
            }
        }
        this.features.set(features);
    }

    public double calculate() {
        return Factory.getMatrixOperations().multiply(weights, features.get());
    }

    public synchronized void addWeightsError(double[] we) {
        for (int i = 0; i < weightsError.length(); i++) {
            weightsError.addAndGet(i, we[i]);
        }


/*        if (this.weightsError == null) {
            this.weightsError = we;
        } else {
            for (int i = 0; i < weightsError.length; i++) {
                weightsError[i] += we[i];
            }
        }*/
    }

    public void resetErrorWeights() {
        IntStream.range(0, weightsError.length()).forEach(i -> weightsError.set(i, 0));
    }
}
