package org.ml4bull.nn.layer;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang.time.StopWatch;
import org.ml4bull.algorithm.HyperbolicTangentFunction;
import org.ml4bull.algorithm.LinearFunction;
import org.ml4bull.algorithm.SoftmaxFunction;
import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.util.Factory;
import org.ml4bull.util.MLUtils;
import org.ml4bull.util.Memory;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.DoubleAdder;

@Slf4j
public class RecurrentNeuronLayer extends HiddenNeuronLayer {

    private ThreadLocal<Memory<double[]>> hiddenStateMemory = new ThreadLocal<>();
    private ThreadLocal<Memory<double[]>> featureMemory = new ThreadLocal<>();
    private ThreadLocal<Memory<double[]>> outMemory = new ThreadLocal<>();
    private ThreadLocal<Memory<double[]>> expectedMemory = new ThreadLocal<>();
    private HiddenNeuronLayer historyLayer;
    private HiddenNeuronLayer outLayer;
    private final int memorySize;

    private DoubleAdder loss = new DoubleAdder();
    private AtomicLong counter = new AtomicLong();

    private static final int LOSS_BATCH_SIZE = 2000;

    /**
     * Simple recurrent neuron net layer.
     */
    public RecurrentNeuronLayer(int hiddenLayerNeuronSize, int outLayerNeuronSize, int memorySize) {
        super(hiddenLayerNeuronSize, new HyperbolicTangentFunction(), false, true);

        this.memorySize = memorySize;
        historyLayer = new HiddenNeuronLayer(hiddenLayerNeuronSize, new LinearFunction(), true, false);
        outLayer = new HiddenNeuronLayer(outLayerNeuronSize, new SoftmaxFunction(), true, true);
    }

    private void initThreadLocals(int memorySize) {
        if (hiddenStateMemory.get() == null) {
            hiddenStateMemory.set(new Memory<>(memorySize));
        }
        if (featureMemory.get() == null) {
            featureMemory.set(new Memory<>(memorySize));
        }
        if (outMemory.get() == null) {
            outMemory.set(new Memory<>(memorySize));
        }
        if (expectedMemory.get() == null) {
            expectedMemory.set(new Memory<>(memorySize));
        }
    }

    @Override
    public double[] forwardPropagation(double[] inValues) {
        initThreadLocals(memorySize);

        featureMemory.get().add(inValues); // for bptt

        double[] rawResults = calculateRawResult(inValues);

        double[] rawResultHistory;
        double[] prevHiddenState = hiddenStateMemory.get().getLast();
        if (prevHiddenState == null) {
            rawResultHistory = new double[rawResults.length];
            historyLayer.calculateRawResult(rawResultHistory);
        } else {
            rawResultHistory = historyLayer.calculateRawResult(prevHiddenState);
        }

        MatrixOperations mo = Factory.getMatrixOperations();
        double[] hiddenStateRaw = mo.sum(rawResultHistory, rawResults);

        // hh - tanh
        double[] hiddenState = activationFunction.activate(hiddenStateRaw);
        hiddenStateMemory.get().add(hiddenState);

        // y - softmax
        double[] out = outLayer.forwardPropagation(hiddenState);
        outMemory.get().add(out);
        return out;
    }


    // Y - out Softmax layer
    // H - hidden layer
    @Override
    public double[] backPropagation(double[] previousError) {
        MatrixOperations mo = Factory.getMatrixOperations();

        expectedMemory.get().add(previousError);
        int target = previousError.length - MLUtils.transformClassToInt(previousError);
        calculateLoss(target);

        int hsNeuronSize = neurons.size();
        int hsNeuronWeightSize = neurons.get(0).getWeights().length;
        double[] dhNext = new double[hsNeuronSize];
        int timeSize = hiddenStateMemory.get().size();

        double[][] dWhy = new double[outLayer.neurons.size()][outLayer.neurons.get(0).getWeights().length]; // Y error weights accumulator
        double[][] dWhh = new double[hsNeuronSize][hsNeuronSize]; // H error weights accumulator
        double[][] dWxh = new double[hsNeuronSize][hsNeuronWeightSize]; // X error weights accumulator
        double[] dby = new double[outLayer.neurons.size()]; // Y bias accumulator
        double[] dbh = new double[historyLayer.neurons.size()]; // H bias accumulator

        for (int i = 0; i < timeSize; i++) {

            // y processing
            double[] yOut = outMemory.get().get(i);
            double[] expected = expectedMemory.get().get(i);
            double[] dY = mo.copy(yOut);

            target = expected.length - MLUtils.transformClassToInt(expected);
            dY[target] -= 1;

            double[] hiddenState = hiddenStateMemory.get().get(i);

            outLayer.lastInput.set(hiddenState); // set time state
            dWhy = mo.sum(dWhy, outLayer.calculateDeltaError(dY));
            dby = mo.sum(dby, outLayer.calculateBiasDeltaError(dY));

            double[] dh = mo.sum(dhNext, outLayer.gradientVector(dY));

            // current layer dW
            double[] dhRaw = mo.scalarMultiply(dh, activationFunction.derivative(hiddenState));

            this.lastInput.set(featureMemory.get().get(i));
            dWxh = mo.sum(dWxh, calculateDeltaError(dhRaw));

            // history layer update dW
            historyLayer.lastInput.set(i < timeSize - 1 ? hiddenStateMemory.get().get(i + 1) : new double[hsNeuronSize]); // set time state
            dWhh = mo.sum(dWhh, historyLayer.calculateDeltaError(dhRaw));
            dbh = mo.sum(dbh, historyLayer.calculateBiasDeltaError(dhRaw)); // accumulate h bias

            dhNext = historyLayer.gradientVector(dhRaw);
        }

        shrink(dWhh, timeSize);
        shrink(dWxh, timeSize);
        shrink(dWhy, timeSize);
        shrink(dby, timeSize);
        shrink(dbh, timeSize);

        // weights
        historyLayer.addErrorsToNeurons(dWhh);
        outLayer.addErrorsToNeurons(dWhy);
        addErrorsToNeurons(dWxh);

        // biases
        historyLayer.addBiasErrorsToNeurons(dbh);
        outLayer.addBiasErrorsToNeurons(dby);

        return dhNext;
    }

/*    @Override
    public double[] backPropagation(double[] previousError) {
        MatrixOperations mo = Factory.getMatrixOperations();

        expectedMemory.get().add(previousError);
        int target = previousError.length - MLUtils.transformClassToInt(previousError);
        calculateLoss(target);

        double[] layerError = new double[neurons.size()];
        double[] dby = new double[outLayer.neurons.size()]; // Y bias
        double[] dbh = new double[historyLayer.neurons.size()]; // H bias
        double[] dHt = new double[historyLayer.neurons.size()];
        double[][] dWhy = new double[outLayer.neurons.size()][outLayer.neurons.get(0).getWeights().length];
        double[][] dWhh = new double[historyLayer.neurons.size()][historyLayer.neurons.get(0).getWeights().length];
        double[][] dWxh = new double[neurons.size()][neurons.get(0).getWeights().length];

        int timeSize = hiddenStateMemory.get().size();
        for (int i = 0; i < timeSize; i++) {

            // y processing
            double[] yOut = outMemory.get().get(i);
            double[] expected = expectedMemory.get().get(i);
            double[] dY = mo.copy(yOut);

            target = expected.length - MLUtils.transformClassToInt(expected);
            dY[target] -= 1;

            outLayer.lastInput.set(hiddenStateMemory.get().get(i)); // set time state
            outLayer.lastResult.set(yOut);

            dWhy = mo.sum(dWhy, outLayer.calculateDeltaError(dY));
            dby = mo.sum(dby, outLayer.calculateBiasDeltaError(dY));

            double[] dH = outLayer.gradientVector(dY);

            // current layer dW
            double[] derivative = activationFunction.derivative(hiddenStateMemory.get().get(i));
            dHt = mo.sum(dHt, historyLayer.gradientVector(derivative));
            layerError = mo.sum(dH, dHt);

            lastInput.set(featureMemory.get().get(i));  // set time state
//            lastResult.set(hiddenStateMemory.get().get(i));
            dWxh = mo.sum(dWxh, calculateDeltaError(layerError));

            // history layer update dW
            historyLayer.lastInput.set(i < timeSize - 1 ? hiddenStateMemory.get().get(i + 1) : new double[neurons.size()]); // set time state
            dWhh = mo.sum(dWhh, historyLayer.calculateDeltaError(layerError));
            dbh = mo.sum(dbh, historyLayer.calculateBiasDeltaError(layerError));
        }

        // shrinking weights
        shrink(dWhh, timeSize);
        shrink(dWxh, timeSize);
        shrink(dWhy, timeSize);

        // shrinking biases
        shrink(dby, timeSize);
        shrink(dbh, timeSize);

        // setting weights
        historyLayer.addErrorsToNeurons(dWhh);
        outLayer.addErrorsToNeurons(dWhy);
        addErrorsToNeurons(dWxh);

        // setting biases
        historyLayer.addBiasErrorsToNeurons(dbh);
        outLayer.addBiasErrorsToNeurons(dby);

        return layerError;
    }*/

    private StopWatch stopwatch = new StopWatch(); // todo: to remove

    private void calculateLoss(int target) {
        loss.add(Math.log(outMemory.get().getLast()[target]));

        if (counter.incrementAndGet() % LOSS_BATCH_SIZE == 0) {
            log.info("Time between lost batch estimations: {}s", TimeUnit.MILLISECONDS.toSeconds(stopwatch.getTime()));
            System.out.println("=========== LOSS ==============");
            log.info("Loss: {}", -1 * loss.doubleValue() / counter.get());
            System.out.println("===============================");
            loss.reset();
            counter.set(0);
            stopwatch.reset();
            stopwatch.start();
        }
    }

    @Override
    public void optimizeWeights(OptimizationAlgorithm optAlg) {
        historyLayer.optimizeWeights(optAlg);
        outLayer.optimizeWeights(optAlg);
        super.optimizeWeights(optAlg);
    }

    private void shrink(double[][] layerWeights, int denominator) {
        for (double[] neuronWeights : layerWeights) {
            for (int i = 0; i < neuronWeights.length; i++) {
                neuronWeights[i] /= denominator;
            }
        }
    }

    private void shrink(double[] biases, int denominator) {
        for (int i = 0; i < biases.length; i++) {
            biases[i] /= denominator;
        }
    }
}
