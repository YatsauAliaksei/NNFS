package org.ml4bull.algorithm.optalg;

import lombok.Builder;

public class ADAMGradientDescent extends GradientDescent {

    @Builder(builderMethodName = "build")
    private ADAMGradientDescent(int batchSize, double regularizationRate, double learningRate, boolean withRegularization) {
        super(batchSize, regularizationRate, learningRate, 0, withRegularization);
    }

    @Override
    public void optimizeWeights(double[] weights, double[] weightsError) {
        double eps = 1e-8;
        double beta1 = 9e-1;
        double beta2 = 999e-3;
        double m = 0;
        double v = 0;

        for (int w = 0; w < weightsError.length; w++) {
            m = beta1 * m + (1 - beta1) * weightsError[w];
            v = beta2 * v + (1 - beta2) * Math.pow(weightsError[w], 2);

            // omit bias regularization
            double regularization = w == 0 ? 0 : regularizationRate * weights[w];
            if (!withRegularization) regularization = 0;

            weights[w] -= learningRate * (m + regularization) / (Math.sqrt(v) + eps) / batchSize;
        }

    }
}
