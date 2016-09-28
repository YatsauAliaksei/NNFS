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

/*    @Override
    public double[] backPropagation(double[] previousError) {
        // calculate out error i.e. back propagation.
        double[] errorOut = new double[expected[i].length];
        for (int j = 0; j < expected[i].length; j++) {
            errorOut[j] = calcY[j] - expected[i][j];
        }
        return super.backPropagation(previousError);
    }*/
}
