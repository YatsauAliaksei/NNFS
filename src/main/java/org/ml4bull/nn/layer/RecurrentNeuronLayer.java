package org.ml4bull.nn.layer;

import com.google.common.util.concurrent.AtomicDoubleArray;
import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.HyperbolicTangentFunction;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.nn.Neuron;
import org.ml4bull.util.Factory;

import java.util.ArrayList;
import java.util.List;

public class RecurrentNeuronLayer extends HiddenNeuronLayer {

    private List<double[]> memory = new ArrayList<>();

    /**
     * Simple recurrent neuron net layer.
     * @param neuronsCount - should be the same as for previous layer.
     * @param unused       - unused.
     */
    public RecurrentNeuronLayer(int neuronsCount, ActivationFunction unused) {
        super(neuronsCount, new HyperbolicTangentFunction());
        memory.add(new double[neuronsCount]);
    }

    @Override
    public double[] forwardPropagation(double[] inValues) {
        MatrixOperations mo = Factory.getMatrixOperations();

        double[] stepOut = super.forwardPropagation(getLast());
        double[] out = mo.sum(inValues, stepOut);

        out = activationFunction.activate(out);
        memory.add(out);

        return out;
    }

    private double[] getLast() {
        return memory.get(memory.size() - 1);
    }

    @Override
    public double[] backPropagation(double[] previousError) {
        for (int i = memory.size() - 2; i <= 0 && i > -1; i--) {
            lastInput.set(memory.get(i));
            previousError = super.backPropagation(previousError);
        }

        MatrixOperations mo = Factory.getMatrixOperations();
        mo.scalarMultiply(previousError, 1.0 / memory.size());

        // Take average of all error in neuron.
        for (Neuron neuron : neurons) {
            AtomicDoubleArray weightsError = neuron.getWeightsError();
            for (int i = 0; i < weightsError.length(); i++) {
                weightsError.set(i, weightsError.get(i) / memory.size());
            }
        }

        return previousError;
    }
}
