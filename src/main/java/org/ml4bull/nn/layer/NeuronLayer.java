package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.nn.Neuron;

import java.util.List;

public interface NeuronLayer {
    double[] forwardPropagation(double[] inValues);

    double[] backPropagation(double[] previousError);

    void optimizeWeights(OptimizationAlgorithm optAlg);

    List<Neuron> getNeurons();

    double[] calculateRawResult(double[] b);

    double[] activate(double[] rawResults);

    default double[] enrichFeatureWithBias(double[] f) {
        double[] b = new double[f.length + 1];
        b[0] = 1;
        System.arraycopy(f, 0, b, 1, f.length);
        return b;
    }
}
