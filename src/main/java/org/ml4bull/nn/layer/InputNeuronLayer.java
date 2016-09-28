package org.ml4bull.nn.layer;

import org.ml4bull.nn.Neuron;

import java.util.List;

public class InputNeuronLayer implements NeuronLayer {
    private int inputSize;

    public InputNeuronLayer(int input) {
        this.inputSize = input;
    }

    @Override
    public double[] forwardPropagation(double[] inValues) {
        return inValues;
    }

    @Override
    public double[] backPropagation(double[] previousError) {
        return new double[0];
    }

    @Override
    public List<Neuron> getNeurons() {
        return null;
    }

    public int getInputSize() {
        return inputSize;
    }
}
