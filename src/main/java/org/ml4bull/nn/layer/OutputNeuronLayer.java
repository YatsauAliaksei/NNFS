package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.ActivationFunction;

public class OutputNeuronLayer extends HiddenNeuronLayer {

    public OutputNeuronLayer(int outputSize, ActivationFunction activationFunction) {
        super(outputSize, activationFunction);
        isDropoutEnabled = false;
    }

    @Override
    public double[] backPropagation(double[] expected) { // In output layer error is simple expected value.
//         calculate out error start point for back propagation.
        double[] errorOut = new double[expected.length];
        double[] result = lastResult.get();
        for (int j = 0; j < expected.length; j++) {
            errorOut[j] = result[j] - expected[j];
        }

        return super.backPropagation(errorOut);
    }
}
