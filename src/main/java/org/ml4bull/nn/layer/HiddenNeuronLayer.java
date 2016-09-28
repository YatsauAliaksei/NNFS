package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.nn.Neuron;
import org.ml4bull.util.Factory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class HiddenNeuronLayer implements NeuronLayer {
    private List<Neuron> neurons;
    private ActivationFunction activationFunction;
    private double[] lastResult;
    private double[] lastInput;

    public HiddenNeuronLayer(int neuronsCount, ActivationFunction activationFunction) {
        this.neurons = new ArrayList<>((int) (neuronsCount * 1.75 + 1));
        for (int i = 0; i < neuronsCount; i++) {
            neurons.add(new Neuron());
        }
        this.activationFunction = activationFunction;
    }

    public double[] forwardPropagation(double[] f) {
        lastInput = f;
        double[] b = new double[lastInput.length + 1];
        b[0] = 1;
        System.arraycopy(lastInput, 0, b, 1, lastInput.length);

        lastResult = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            Neuron n = neurons.get(i);
            n.setFeatures(b);
            lastResult[i] = compute(n);
        }
        return lastResult;
    }

    @Override
    public double[] backPropagation(double[] previousError) {
        double[][] theta = new double[neurons.size()][];
        MatrixOperations mo = Factory.getMatrixOperations();
        for (int s = 0; s < neurons.size(); s++) {
            double[] weights = neurons.get(s).getWeights();
            theta[s] = Arrays.copyOfRange(weights, 1, weights.length);
        }

        // calculating next layer error
        double[][] thetaT = mo.transpose(theta);
        double[] e = mo.multiplySingleDim(thetaT, previousError);
        double[] a = new double[lastInput.length];
        double[] currentError = new double[e.length];

        for (int s = 0; s < a.length; s++)
            a[s] = (1 - lastInput[s]) * lastInput[s];

        for (int d = 0; d < currentError.length; d++)
            currentError[d] = e[d] * a[d];

        return currentError;
    }

    @Override
    public List<Neuron> getNeurons() {
        return neurons;
    }

    private double compute(Neuron neuron) {
        double value = neuron.calculate();
        return activationFunction.activate(value);
    }
}
