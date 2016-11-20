package org.ml4bull.algorithm;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

@Builder
public class GradientDescent implements OptimizationAlgorithm {

    @Getter
    private int batchSize;
    @Getter
    @Setter
    private double regularizationRate;
    @Getter
    @Setter
    private double learningRate;
    @Getter
    private AtomicInteger counter;

    @Override
    public boolean isLimitReached() {
        if (counter.incrementAndGet() == batchSize) {
            counter.addAndGet(-batchSize);
            return true;
        }
        return false;
    }

    public boolean hasError() {
        return counter.get() != 0;
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
        private AtomicInteger counter = new AtomicInteger();
    }
}
