package org.ml4bull.algorithm;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.stream.IntStream;

@Builder
public class GradientDescent implements OptimizationAlgorithm {

    @Getter
    private final int batchSize;
    @Getter
    @Setter
    private double regularizationRate = 8e-2;
    @Getter
    @Setter
    private double learningRate = 12e-1;
    private volatile int counter;

    @Override
    public boolean isLimitReached() {
        if (++counter == batchSize) {
            counter = 0;
            return true;
        }
        return false;
    }

    @Override
    public void optimizeWeights(double[] weights, double[] weightsError) {
        IntStream.range(0, weightsError.length)
                .forEach(w -> {
                    // omit bias regularization
                    double regularization = w == 0 ? 0 : regularizationRate * weights[w];
                    weights[w] -= learningRate * (weightsError[w] + regularization) / batchSize;
                });
    }

    public static class GradientDescentBuilder {
        private double regularizationRate = 8e-2;
        private double learningRate = 12e-1;
    }
}
