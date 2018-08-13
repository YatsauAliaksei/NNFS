package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.DropoutRegularization;
import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.nn.Neuron;

import java.util.List;

public class InputNeuronLayer implements NeuronLayer {
    private int inputSize;
    private boolean isDropEnabled = false;
    private DropoutRegularization dropoutRegularization = new DropoutRegularization(0.005);

    public InputNeuronLayer(int input) {
        this.inputSize = input;
    }

    @Override
    public double[] forwardPropagation(double[] inValues) {
        double[] result = inValues;

        if (isDropEnabled)
            result = dropoutRegularization.dropout(result);

        return result;
    }

    @Override
    public double[] backPropagation(double[] previousError) {
        return new double[0];
    }

    @Override
    public void optimizeWeights(OptimizationAlgorithm unused) {
        // no-op
    }

    @Override
    public List<Neuron> getNeurons() {
        return null;
    }

    @Override
    public double[] calculateRawResult(double[] b) {
        return new double[0];
    }

    @Override
    public double[] activate(double[] rawResults) {
        return new double[0];
    }

    public int getInputSize() {
        return inputSize;
    }
}
