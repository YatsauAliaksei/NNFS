package org.ml4bull.algorithm.optalg;

import org.ml4bull.nn.Neuron;

public interface OptimizationAlgorithm {

    boolean isLimitReached();

//    void optimizeWeights(double[] weights, double[] weightsError);

    void optimizeWeights(Neuron neuron);

    int getBatchSize();

    default void optimizeBiases(double[] biases) {};
}
