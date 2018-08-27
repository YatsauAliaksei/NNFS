package org.ml4bull.algorithm.optalg;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;
import org.ml4bull.nn.Neuron;

@Builder
@AllArgsConstructor
public class GradientDescent implements OptimizationAlgorithm {

    @Getter
    protected int batchSize;
    @Getter
    @Setter
    @Builder.Default
    protected double regularizationRate = 1e-6;
    @Getter
    @Setter
    @Builder.Default
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
    public void optimizeWeights(Neuron neuron) {
        double[] weights = neuron.getWeights();
        double[] weightsError = neuron.getWeightsErrorPrimitive();

        for (int w = 0; w < weightsError.length; w++) {
            double regularization = withRegularization ? regularizationRate * weights[w] : 0;

            weights[w] -= learningRate * (weightsError[w] + regularization) / batchSize;
        }
    }
}
