package org.ml4bull.algorithm.optalg;

public interface OptimizationAlgorithm {
    boolean isLimitReached();

    void optimizeWeights(double[] weights, double[] weightsError);

    int getBatchSize();
}
