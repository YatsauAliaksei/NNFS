package org.ml4bull.algorithm.optalg;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.concurrent.atomic.AtomicInteger;

@Builder
public class GradientDescent implements OptimizationAlgorithm {

    @Getter
    protected int batchSize;
    @Getter
    @Setter
    protected double regularizationRate = 8e-2;
    @Getter
    @Setter
    protected double learningRate = 12e-1;
    @Getter
    private AtomicInteger counter = new AtomicInteger();
    @Getter
    protected boolean withRegularization;

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
        for (int w = 0; w < weightsError.length; w++) {

            // omit bias regularization
            double regularization = w == 0 ? 0 : regularizationRate * weights[w];
            if (!withRegularization) regularization = 0;

            weights[w] -= learningRate * (weightsError[w] + regularization) / batchSize;
        }
    }

    public static class GradientDescentBuilder {
        private double regularizationRate = 8e-2;
        private double learningRate = 12e-1;
        private AtomicInteger counter = new AtomicInteger();
    }
}
