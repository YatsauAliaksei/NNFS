package org.ml4bull.nn;

import org.ml4bull.algorithm.ActivationFunction;

import java.util.List;

public class NeuronLayer {
    private List<Neuron> neurons;
    private ActivationFunction activationFunction;

    public NeuronLayer(List<Neuron> neurons, ActivationFunction activationFunction) {
        this.neurons = neurons;
        this.activationFunction = activationFunction;
    }

    public double[] compute() {
        double[] out = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            out[i] = compute(neurons.get(i));
        }
        return out;
    }

    private double compute(Neuron neuron) {
        if (neuron.isBias()) return 1;

        return activationFunction.activate(neuron.getWeights(), neuron.getFeatures());
    }

    public void input(double[] f) {
        neurons.forEach(neuron -> neuron.setFeatures(f));
    }
}
