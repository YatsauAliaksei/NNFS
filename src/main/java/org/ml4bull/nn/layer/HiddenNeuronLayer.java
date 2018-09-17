package org.ml4bull.nn.layer;

import com.google.common.base.Preconditions;
import lombok.Getter;
import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.DropoutRegularization;
import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.nn.Neuron;
import org.ml4bull.util.Factory;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class HiddenNeuronLayer implements NeuronLayer {
    protected boolean isDropoutEnabled = true;
    @Getter
    protected List<Neuron> neurons;
    protected ActivationFunction activationFunction;
    protected ThreadLocal<double[]> lastResult = new ThreadLocal<>();
    protected ThreadLocal<double[]> lastInput = new ThreadLocal<>();
    private DropoutRegularization dropoutRegularization = new DropoutRegularization(0.01);

    public HiddenNeuronLayer(int neuronsCount, ActivationFunction activationFunction) {
        this(neuronsCount, activationFunction, true);
    }

    public HiddenNeuronLayer(int neuronsCount, ActivationFunction activationFunction, boolean withBias) {
        createNeurons(neuronsCount, withBias);
        this.activationFunction = activationFunction;
    }

    public HiddenNeuronLayer(int neuronsCount, ActivationFunction activationFunction, boolean withBias, boolean isDropoutEnabled) {
        this(neuronsCount, activationFunction, withBias);
        this.isDropoutEnabled = isDropoutEnabled;
    }

    protected void createNeurons(int neuronsCount, boolean withBias) {
        neurons = IntStream.range(0, neuronsCount)
                .mapToObj(i -> Neuron.builder().bias(withBias ? 1 : 0).build())
                .collect(Collectors.toCollection(ArrayList::new));
    }

    public double[] forwardPropagation(double[] f) {
        lastInput.set(f);
        double[] rawResults = calculateRawResult(f);
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
            n.tryInit(b.length);
            rawResults[i] = n.calculate(b);
        });
        return rawResults;
    }

    @Override
    public double[] backPropagation(double[] previousError) {
        MatrixOperations mo = Factory.getMatrixOperations();

        double[] derivative = activationFunction.derivative(lastResult.get());
        double[] layerError = mo.scalarMultiply(previousError, derivative);

        calculateAndSaveDeltaError(layerError);

        return gradientVector(layerError);
    }

    protected double[] gradientVector(double[] layerError) {
        // layer weights matrix
        double[][] theta = createLayerWeightMatrix();

        MatrixOperations mo = Factory.getMatrixOperations();
        // calculating gradient vector
        double[][] thetaT = mo.transpose(theta);
        return mo.multiplySingleDim(thetaT, layerError);
    }

    protected double[][] createLayerWeightMatrix() {
        double[][] theta = new double[neurons.size()][];

        IntStream.range(0, neurons.size()).forEach(i ->
                theta[i] = neurons.get(i).getWeights()
        );
        return theta;
    }

    protected void addErrorsToNeurons(double[][] neuronsWeightsError) {
        IntStream.range(0, neurons.size()).forEach(i -> {
            Neuron neuron = neurons.get(i);
            neuron.addWeightsError(neuronsWeightsError[i]);
        });
    }

    protected void addBiasErrorsToNeurons(double[] neuronsBiasError) {
        IntStream.range(0, neurons.size()).forEach(i -> {
            Neuron neuron = neurons.get(i);
            neuron.addBiasError(neuronsBiasError[i]);
        });
    }

    protected double[][] calculateAndSaveDeltaError(double[] error) {
        double[][] layerNetMatrix = calculateDeltaError(error);
        addErrorsToNeurons(layerNetMatrix);

        return layerNetMatrix;
    }

    // Theta T x E. Multiply weights on previous layer error.
    protected double[][] calculateDeltaError(double[] error) {
        Preconditions.checkArgument(error.length == neurons.size(), "Error length should be equal to neurons size.");

        double[][] layerNetMatrix = new double[neurons.size()][];
        double[] lastIn = lastInput.get();
        int weightsSize = neurons.get(0).getWeights().length;
        for (int i = 0; i < layerNetMatrix.length; i++) {
            double[] we = new double[weightsSize];
            for (int t = 0; t < we.length; t++) {
                we[t] = error[i] * lastIn[t]; // the rate of change of the cost with respect to any weight in the network
            }
            layerNetMatrix[i] = we;
        }

//            gradientCheck(we);
        return layerNetMatrix;
    }

    protected double[] calculateBiasDeltaError(double[] error) {
        Preconditions.checkArgument(error.length == neurons.size(), "Error length should be equal to neurons size.");

        MatrixOperations mo = Factory.getMatrixOperations();
        return mo.copy(error);
    }

/*    private boolean isGradientCheckEnable = false;

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
    }*/

    @Override
    public void optimizeWeights(OptimizationAlgorithm optAlg) {
        neurons.forEach(neuron -> {
            optAlg.optimizeWeights(neuron);
            neuron.resetErrorWeights();
        });
    }
}
