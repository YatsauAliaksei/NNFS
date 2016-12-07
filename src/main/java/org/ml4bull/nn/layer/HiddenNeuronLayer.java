package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.nn.Neuron;
import org.ml4bull.util.Factory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class HiddenNeuronLayer implements NeuronLayer {
    protected List<Neuron> neurons;
    protected ActivationFunction activationFunction;
    protected ThreadLocal<double[]> lastResult = new ThreadLocal<>();
    protected ThreadLocal<double[]> lastInput = new ThreadLocal<>();

    public HiddenNeuronLayer(int neuronsCount, ActivationFunction activationFunction) {
        neurons = IntStream.range(0, neuronsCount)
                .mapToObj(i -> new Neuron())
                .collect(Collectors.toList());
        this.activationFunction = activationFunction;
    }

    public double[] forwardPropagation(double[] f) {
        lastInput.set(f);
        double[] b = new double[f.length + 1];
        b[0] = 1;
        System.arraycopy(f, 0, b, 1, f.length);

        double[] rawResults = new double[neurons.size()];

        IntStream.range(0, neurons.size()).forEach(i -> {
            Neuron n = neurons.get(i);
            n.setFeatures(b);
            rawResults[i] = n.calculate();
        });

        lastResult.set(activationFunction.activate(rawResults));
        return lastResult.get();
    }

    @Override
    public double[] backPropagation(double[] previousError) {
        calculateWeightsError(previousError);

        // prepare
        // layer weights matrix
        double[][] theta = new double[neurons.size()][];

        IntStream.range(0, neurons.size()).forEach(i -> {
            double[] weights = neurons.get(i).getWeights();
            theta[i] = Arrays.copyOfRange(weights, 1, weights.length);
        });

        MatrixOperations mo = Factory.getMatrixOperations();
        // calculating layer error
        double[][] thetaT = mo.transpose(theta);
        double[] e = mo.multiplySingleDim(thetaT, previousError);
        double[] currentError = new double[e.length];

        double[] a = activationFunction.derivative(lastInput.get());

        IntStream.range(0, currentError.length).forEach(d -> currentError[d] = e[d] * a[d]);

        return currentError;
    }

    // Theta T x E. Multiply weights on previous layer error.
    protected void calculateWeightsError(double[] error) {
        IntStream.range(0, neurons.size()).forEach(i -> {
            Neuron neuron = neurons.get(i);
            double[] we = new double[neuron.getWeights().length];
            we[0] = error[i]; // bias error
            for (int t = 1; t < we.length; t++) { // omitting bias
                we[t] = error[i] * lastInput.get()[t - 1]; // calculate current layer error delta
            }
            neuron.addWeightsError(we);
        });
    }

    @Override
    public List<Neuron> getNeurons() {
        return neurons;
    }
}
