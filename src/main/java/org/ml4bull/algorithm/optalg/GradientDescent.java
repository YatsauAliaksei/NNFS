package org.ml4bull.algorithm.optalg;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Builder
public class GradientDescent implements OptimizationAlgorithm {

    @Getter
    protected int batchSize;
    @Getter
    @Setter
    protected double regularizationRate = 8e-6;
    @Getter
    @Setter
    protected double learningRate = 1e-2;
    @Getter
    private int counter;
    @Getter
    protected boolean withRegularization;

    @Override
    public synchronized boolean isLimitReached() {
        if (++counter == batchSize) {
            counter = 0;
            return true;
        }
        return false;
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
        private double regularizationRate = 8e-6;
        private double learningRate = 1e-2;
//        private AtomicInteger counter = new AtomicInteger();
    }
}
