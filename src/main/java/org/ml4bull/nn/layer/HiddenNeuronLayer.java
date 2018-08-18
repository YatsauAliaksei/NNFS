package org.ml4bull.nn.layer;

import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.AtomicDoubleArray;
import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.DropoutRegularization;
import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.nn.Neuron;
import org.ml4bull.util.Factory;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class HiddenNeuronLayer implements NeuronLayer {
    protected boolean isDropoutEnabled = true;
    protected List<Neuron> neurons;
    protected ActivationFunction activationFunction;
    protected ThreadLocal<double[]> lastResult = new ThreadLocal<>();
    protected ThreadLocal<double[]> lastInput = new ThreadLocal<>();
    private DropoutRegularization dropoutRegularization = new DropoutRegularization(0.005);

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
        if (isDropoutEnabled)
            afterDropout = dropoutRegularization.dropout(rawResults);

        double[] activated = activationFunction.activate(afterDropout);
        lastResult.set(activated);
        return activated;
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
        MatrixOperations mo = Factory.getMatrixOperations();

        double[] derivative = activationFunction.derivative(lastResult.get());
        double[] layerError = mo.scalarMultiply(previousError, derivative);

        calculateAndSaveDeltaError(layerError);
//        clip();

        return gradientVector(layerError);
    }

    private void clip() {
        for (Neuron neuron : neurons) {
            AtomicDoubleArray weightsError = neuron.getWeightsError();
            for (int i = 0; i < weightsError.length(); i++) {
                double dw = weightsError.get(i);
                if (dw > 5 || dw < -5) {
                    weightsError.set(i, dw > 5 ? 5 : -5);
                }
            }
        }
    }

    protected double[] gradientVector(double[] layerError) {
        // layer weights matrix
        double[][] theta = createLayerWeightMatrix();

        MatrixOperations mo = Factory.getMatrixOperations();
        // calculating gradient vector
        double[][] thetaT = mo.transpose(theta);
        return mo.multiplySingleDim(thetaT, layerError);
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
        Preconditions.checkArgument(error.length == neurons.size(), "Error length should be equal to neurons size.");

        IntStream.range(0, neurons.size()).forEach(i -> {
            Neuron neuron = neurons.get(i);
            double[] we = new double[neuron.getWeights().length];
            we[0] = error[i]; // bias error
            for (int t = 1; t < we.length; t++) { // omitting bias
                we[t] = error[i] * lastInput.get()[t - 1]; // the rate of change of the cost with respect to any weight in the network
            }
            neuron.addWeightsError(we);
//            gradientCheck(we);
        });
    }

    private boolean isGradientCheckEnable = false;

    private void gradientCheck(double[] gradientDerivative) {
        if (!isGradientCheckEnable) return;

        double e = 1e-4;
        double allowableError = 1e-4;

        neurons.forEach(neuron -> {
            double[] weights = neuron.getWeights();
            double[] derVal = new double[weights.length];

            for (int i = 0; i < weights.length; i++) {
                double tmp = weights[i];

                weights[i] += e;
                double resultPlusE = neuron.calculate();

                weights[i] = tmp - e;
                double resultMinusE = neuron.calculate();

                derVal[i] = (resultPlusE - resultMinusE) / 2 * e;
                // return old value
                weights[i] = tmp;
            }

            for (int i = 0; i < derVal.length; i++) {
                System.out.println("Gradient error. Expected: " + derVal[i] + " Actual: " + gradientDerivative[i]);
                if (derVal[i] - gradientDerivative[i] > allowableError) {
                    System.out.println("Gradient error. Expected: " + derVal[i] + " Actual: " + gradientDerivative[i]);
                    throw new RuntimeException("Gradient error. Expected: " + derVal[i] + " Actual: " + gradientDerivative[i]);
                }
            }
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
