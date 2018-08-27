package org.ml4bull.algorithm.optalg;

import lombok.Builder;
import org.ml4bull.nn.Neuron;

import java.util.HashMap;
import java.util.Map;

public class RMSPropGradientDescent extends GradientDescent {

    private Map<Neuron, double[]> cacheMap = new HashMap<>();
    private Map<Neuron, Double> biasCache = new HashMap<>();
    private double decay_rate = 9e-1;
    private double eps = 1e-8;

    @Builder(builderMethodName = "buildRMS")
    private RMSPropGradientDescent(int batchSize, double regularizationRate, double learningRate, boolean withRegularization) {
        super(batchSize, regularizationRate, learningRate, 0, withRegularization);
    }

    @Override
    public void optimizeWeights(Neuron neuron) {

        // weights update
        double[] weights = neuron.getWeights();
        double[] cache = cacheMap.computeIfAbsent(neuron, k -> new double[weights.length]);
        double[] weightsError = neuron.getWeightsErrorPrimitive();

        // RMSprop
        for (int w = 0; w < weightsError.length; w++) {
            cache[w] = decay_rate * cache[w] + (1 - decay_rate) * Math.pow(weightsError[w], 2);

            double regularization = withRegularization ? regularizationRate * weights[w] : 0;

            weights[w] -= learningRate * (weightsError[w] + regularization) / (Math.sqrt(cache[w]) + eps) / batchSize;
        }

        // bias update
        double bCache = biasCache.computeIfAbsent(neuron, k -> 0d);

        double biasErrorSum = neuron.getBiasErrorSum();
        double bias = neuron.getBias();
        bCache = decay_rate * bCache + (1 - decay_rate) * Math.pow(biasErrorSum, 2);
        bias -= learningRate * (biasErrorSum) / (Math.sqrt(bCache) + eps) / batchSize;

        biasCache.put(neuron, bCache);
        neuron.setBias(bias);
    }
}
