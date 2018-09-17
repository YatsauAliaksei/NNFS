package org.ml4bull.nn;


import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.util.Factory;
import org.ml4bull.util.MLUtils;

import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

@Builder
public class Neuron {
    @Getter
    @Setter
    private double[] weights;
    @Setter
    @Getter
    private double bias;
    @Getter
    private final Queue<double[]> weightsErrorQueue = new ConcurrentLinkedQueue<>();
    @Getter
    private final Queue<Double> biasErrorQueue = new ConcurrentLinkedQueue<>();

    public void tryInit(int numberOfFeatures) {
        if (weights != null) return;

        synchronized (this) {
            if (weights == null)
                weights = MLUtils.getRandomWeights(numberOfFeatures); // basic random weights.
        }
    }

    public double calculate(double[] features) {
        return Factory.getMatrixOperations().multiply(weights, features) + bias;
    }

    public void addWeightsError(double[] we) {
        weightsErrorQueue.add(we);
    }

    public void addBiasError(double v) {
        biasErrorQueue.add(v);
    }

    public double[] getWeightsErrorPrimitive() {
        MatrixOperations mo = Factory.getMatrixOperations();
        return weightsErrorQueue.stream().reduce(mo::sum).orElseThrow(RuntimeException::new);
    }

    public double getBiasErrorSum() {
        return biasErrorQueue.stream().mapToDouble(Double::doubleValue).sum();
    }

    public void resetErrorWeights() {
        weightsErrorQueue.clear();
        biasErrorQueue.clear();
    }
}
