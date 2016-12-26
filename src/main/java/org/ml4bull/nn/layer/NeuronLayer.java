package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.OptimizationAlgorithm;
import org.ml4bull.nn.Neuron;

import java.util.List;

public interface NeuronLayer {
    double[] forwardPropagation(double[] inValues);

    double[] backPropagation(double[] previousError);

    void optimizeWeights(OptimizationAlgorithm optAlg);

    List<Neuron> getNeurons();
}
