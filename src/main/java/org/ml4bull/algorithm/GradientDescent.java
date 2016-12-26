package org.ml4bull.algorithm;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.concurrent.atomic.AtomicInteger;

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
    @Getter
    private boolean withRegularization;

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
//        rmsProp(weights, weightsError);

        adam(weights, weightsError);
    }


    private void adam(double[] weights, double[] weightsError) {
        double eps = 1e-8;
        double beta1 = 9e-1;
        double beta2 = 999e-3;
        double m = 0;
        double v = 0;

        // adam
        for (int w = 0; w < weightsError.length; w++) {
            m = beta1 * m + (1 - beta1) * weightsError[w];
            v = beta2 * v + (1 - beta2) * Math.pow(weightsError[w], 2);

            // omit bias regularization
            double regularization = w == 0 ? 0 : regularizationRate * weights[w];
            if (!withRegularization) regularization = 0;

            weights[w] -= learningRate * (m + regularization) / (Math.sqrt(v) + eps) / batchSize;
        }
    }

    private void rmsProp(double[] weights, double[] weightsError) {
        double cache = 0;
        double eps = 1e-4;
        double decay_rate = 9e-1;

        // RMSprop
        for (int w = 0; w < weightsError.length; w++) {
            cache = decay_rate * cache + (1 - decay_rate) * Math.pow(weightsError[w], 2);

            // omit bias regularization
            double regularization = w == 0 ? 0 : regularizationRate * weights[w];
            if (!withRegularization) regularization = 0;

            weights[w] -= learningRate * (weightsError[w] + regularization) / (Math.sqrt(cache) + eps) / batchSize;
        }
    }

    public static class GradientDescentBuilder {
        private double regularizationRate = 8e-2;
        private double learningRate = 12e-1;
        private AtomicInteger counter = new AtomicInteger();
    }
}
