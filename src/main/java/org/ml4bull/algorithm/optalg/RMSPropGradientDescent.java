package org.ml4bull.algorithm.optalg;

import lombok.Builder;

import java.util.concurrent.atomic.AtomicInteger;

public class RMSPropGradientDescent extends GradientDescent {

    @Builder(builderMethodName = "build")
    private RMSPropGradientDescent(int batchSize, double regularizationRate, double learningRate, boolean withRegularization) {
        super(batchSize, regularizationRate, learningRate, new AtomicInteger(), withRegularization);
    }

    @Override
    public void optimizeWeights(double[] weights, double[] weightsError) {
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
}
