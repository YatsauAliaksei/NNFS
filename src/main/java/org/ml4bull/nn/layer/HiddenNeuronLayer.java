package org.ml4bull.nn.layer;

import com.google.common.util.concurrent.AtomicDoubleArray;
import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.DropoutRegularization;
import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.nn.Neuron;
import org.ml4bull.util.Factory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class HiddenNeuronLayer implements NeuronLayer {
    protected boolean isDropoutEnabled = true;
    protected List<Neuron> neurons;
    protected ActivationFunction activationFunction;
    protected ThreadLocal<double[]> lastResult = new ThreadLocal<>();
    protected ThreadLocal<double[]> lastInput = new ThreadLocal<>();
    private DropoutRegularization dropoutRegularization = new DropoutRegularization(0.5);

    public HiddenNeuronLayer(int neuronsCount, ActivationFunction activationFunction) {
        createNeurons(neuronsCount);
        this.activationFunction = activationFunction;
    }

    protected void createNeurons(int neuronsCount) {
        neurons = IntStream.range(0, neuronsCount)
                .mapToObj(i -> new Neuron())
                .collect(Collectors.toList());
    }

    public HiddenNeuronLayer(int neuronsCount, ActivationFunction activationFunction, boolean isDropoutEnabled) {
        this(neuronsCount, activationFunction);
        this.isDropoutEnabled = isDropoutEnabled;
    }

    public double[] forwardPropagation(double[] f) {
        lastInput.set(f);

        double[] b = enrichFeatureWithBias(f);
        double[] rawResults = calculateRawResult(b);

        return activate(rawResults);
    }

    public double[] activate(double[] rawResults) {
        double[] afterDropout = rawResults;
        if (isDropoutEnabled) {
            afterDropout = dropoutRegularization.dropout(rawResults);
        }
        lastResult.set(activationFunction.activate(afterDropout));
        return lastResult.get();
    }

    protected double[] enrichFeatureWithBias(double[] f) {
        double[] b = new double[f.length + 1];
        b[0] = 1;
        System.arraycopy(f, 0, b, 1, f.length);
        return b;
    }

    public double[] calculateRawResult(double[] b) {
        double[] rawResults = new double[neurons.size()];

        IntStream.range(0, neurons.size()).forEach(i -> {
            Neuron n = neurons.get(i);
            n.setFeatures(b);
            rawResults[i] = n.calculate();
        });
        return rawResults;
    }

    @Override
    public double[] backPropagation(double[] previousError) {
        calculateAndSaveDeltaError(previousError);

        return calculateLayerError(previousError);
    }

    protected double[] calculateLayerError(double[] previousError) {
        // layer weights matrix
        double[][] theta = createLayerWeightMatrix();

        MatrixOperations mo = Factory.getMatrixOperations();
        // calculating layer error
        double[][] thetaT = mo.transpose(theta);
        double[] e = mo.multiplySingleDim(thetaT, previousError);
        double[] layerError = new double[e.length];

        double[] d = activationFunction.derivative(lastInput.get());

        IntStream.range(0, layerError.length).forEach(err -> layerError[err] = e[err] * d[err]);

        return layerError;
    }

    private double[][] createLayerWeightMatrix() {
        double[][] theta = new double[neurons.size()][];

        IntStream.range(0, neurons.size()).forEach(i -> {
            double[] weights = neurons.get(i).getWeights();
            theta[i] = Arrays.copyOfRange(weights, 1, weights.length);
        });
        return theta;
    }

    // Theta T x E. Multiply weights on previous layer error.
    protected void calculateAndSaveDeltaError(double[] error) {
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
    public void optimizeWeights(OptimizationAlgorithm optAlg) {
        neurons.forEach(neuron -> {
            double[] weights = neuron.getWeights();
            // sum of gradient errors
            AtomicDoubleArray weightsError = neuron.getWeightsError();
            double[] we = IntStream.range(0, weightsError.length()).mapToDouble(weightsError::get).toArray();

            optAlg.optimizeWeights(weights, we);

            neuron.resetErrorWeights();
        });
    }

    @Override
    public List<Neuron> getNeurons() {
        return neurons;
    }
}
