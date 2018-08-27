package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.util.Factory;
import org.ml4bull.util.MLUtils;

public class OutputNeuronLayer extends HiddenNeuronLayer {

    public OutputNeuronLayer(int outputSize, ActivationFunction activationFunction) {
        super(outputSize, activationFunction);
        isDropoutEnabled = false;
    }

    @Override
    public double[] backPropagation(double[] expected) { // In output layer error is simple expected value.
//         calculate out error start point for back propagation.
        MatrixOperations mo = Factory.getMatrixOperations();
        double[] result = mo.copy(lastResult.get());

        int target = expected.length - MLUtils.transformClassToInt(expected);
        result[target] -= 1;

        calculateAndSaveDeltaError(result);

        return gradientVector(result);
    }
}
