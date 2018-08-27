package org.ml4bull.algorithm.optalg;

import lombok.Builder;
import org.ml4bull.nn.Neuron;

import java.util.HashMap;
import java.util.Map;
import java.util.Queue;
import java.util.function.Function;

public class ADAMGradientDescent extends GradientDescent {

    private double eps = 1e-8;
    private double beta1 = 9e-1;
    private double beta2 = 999e-3;
    private Map<Neuron, double[]> cacheMap = new HashMap<>();
    private Map<Neuron, double[]> velocityMap = new HashMap<>();
    private Map<Neuron, Double> biasCache = new HashMap<>();
    private Map<Neuron, Double> biasVelocityCache = new HashMap<>();

    @Builder(builderMethodName = "buildAdam")
    private ADAMGradientDescent(int batchSize, double regularizationRate, double learningRate, boolean withRegularization) {
        super(batchSize, regularizationRate, learningRate, 0, withRegularization);
    }

    @Override
    public void optimizeWeights(Neuron neuron) {
        double[] weights = neuron.getWeights();

        double[] cache = getParam(neuron, cacheMap, k -> new double[weights.length]);
        double[] velocity = getParam(neuron, velocityMap, k -> new double[weights.length]);

        // weights
        int time = 0;
        Queue<double[]> weQueue = neuron.getWeightsErrorQueue(); // TODO: empty every time
        while (!weQueue.isEmpty()) {
            double[] error = weQueue.poll();
            time++;

            for (int w = 0; w < error.length; w++) {
                velocity[w] = beta1 * velocity[w] + (1 - beta1) * error[w];
                cache[w] = beta2 * cache[w] + (1 - beta2) * Math.pow(error[w], 2);

                double velocityCorrected = velocity[w] / (1 - Math.pow(beta1, time));
                double cacheCorrected = cache[w] / (1 - Math.pow(beta2, time));

                double regularization = withRegularization ? regularizationRate * weights[w] : 0;

                weights[w] -= learningRate * (velocityCorrected + regularization) / (Math.sqrt(cacheCorrected) + eps) / batchSize;
            }
        }

        // bias
        time = 0;
        Queue<Double> biasErrorQueue = neuron.getBiasErrorQueue();
        double bCache = getParam(neuron, biasCache, k -> 0d);
        double bVelocity = getParam(neuron, biasVelocityCache, k -> 0d);
        double bias = neuron.getBias();
        while (!biasErrorQueue.isEmpty()) {
            double error = biasErrorQueue.poll();
            time++;

            bVelocity = beta1 * bVelocity + (1 - beta1) * error;
            bCache = beta2 * bCache + (1 - beta2) * Math.pow(error, 2);

            double velocityCorrected = bVelocity / (1 - Math.pow(beta1, time));
            double cacheCorrected = bCache / (1 - Math.pow(beta2, time));

            bias -= learningRate * velocityCorrected / (Math.sqrt(cacheCorrected) + eps) / batchSize;
        }

        neuron.setBias(bias);
    }

    private static <K, Out> Out getParam(K key, Map<K, Out> cacheMap, Function<K, Out> function) {
        return cacheMap.computeIfAbsent(key, function);
    }
}
