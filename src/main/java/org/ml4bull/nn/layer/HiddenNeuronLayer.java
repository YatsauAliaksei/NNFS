package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.nn.Neuron;

import java.util.ArrayList;
import java.util.List;

public class HiddenNeuronLayer implements NeuronLayer {
    private List<Neuron> neurons;
    private ActivationFunction activationFunction;

    public HiddenNeuronLayer(int neuronsCount, ActivationFunction activationFunction) {
        this.neurons = new ArrayList<>((int) (neuronsCount * 1.75 + 1));
        for (int i = 0; i < neuronsCount; i++) {
            neurons.add(new Neuron());
        }
        this.activationFunction = activationFunction;
    }

    public double[] forwardPropagation(double[] f) {
        double[] b = new double[f.length + 1];
        b[0] = 1;
        System.arraycopy(f, 0, b, 1, f.length);

        double[] out = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            Neuron n = neurons.get(i);
            n.setFeatures(b);
            out[i] = compute(n);
        }
        return out;
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
