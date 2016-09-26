package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.ActivationFunction;

public class OutputNeuronLayer extends HiddenNeuronLayer {

    public OutputNeuronLayer(int outputSize, ActivationFunction activationFunction) {
        super(outputSize, activationFunction);
    }

    @Override
    public double[] forwardPropagation(double[] inValues) {
        return super.forwardPropagation(inValues);
    }
}
