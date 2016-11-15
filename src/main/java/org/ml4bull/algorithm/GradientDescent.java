package org.ml4bull.algorithm;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

@Builder
public class GradientDescent implements OptimizationAlgorithm {

    @Getter
    private final long batchSize;
    @Getter
    @Setter
    private double regularizationRate = 8e-2;
    @Getter
    @Setter
    private double learningRate = 12e-1;
    private final AtomicInteger counter = new AtomicInteger();

    @Override
    public boolean isLimitReached() {
        return counter.incrementAndGet() == batchSize;
    }

    @Override
    public void optimizeWeights(double[] weights, double[] weightsError) {
        IntStream.range(0, weightsError.length)
//                .parallel()
                .forEach(w -> {
                    // omit bias regularization
                    double regularization = w == 0 ? 0 : regularizationRate * weights[w];
                    weights[w] -= learningRate * (weightsError[w] + regularization) / batchSize;
                });
        counter.set(0);
    }
}
