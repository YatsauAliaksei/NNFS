package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.nn.Neuron;

import java.util.List;

public class LinearNeuronLayer implements NeuronLayer {

    @Override
    public double[] forwardPropagation(double[] inValues) {
        return inValues;
    }

    @Override
    public double[] backPropagation(double[] previousError) {
        return previousError;
    }

    @Override
    public void optimizeWeights(OptimizationAlgorithm optAlg) {
        // stub
    }

    @Override
    public List<Neuron> getNeurons() {
        return List.of();
    }

    @Override
    public double[] calculateRawResult(double[] b) {
        return b;
    }

    @Override
    public double[] activate(double[] rawResults) {
        return rawResults;
    }

    @Override
    public double[] enrichFeatureWithBias(double[] f) {
        return new double[0];
    }
}
