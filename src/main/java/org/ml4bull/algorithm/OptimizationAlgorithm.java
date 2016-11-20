package org.ml4bull.algorithm;

public interface OptimizationAlgorithm {
    boolean isLimitReached();

    void optimizeWeights(double[] weights, double[] weightsError);

    boolean hasError();

    int getBatchSize();
}
