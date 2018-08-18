package org.ml4bull.nn.layer;

import com.google.common.util.concurrent.AtomicDoubleArray;
import lombok.extern.slf4j.Slf4j;
import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.HyperbolicTangentFunction;
import org.ml4bull.algorithm.LinearFunction;
import org.ml4bull.algorithm.SoftmaxFunction;
import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.nn.Neuron;
import org.ml4bull.util.Factory;
import org.ml4bull.util.MLUtils;
import org.ml4bull.util.Memory;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.DoubleAdder;

@Slf4j
public class RecurrentNeuronLayer extends HiddenNeuronLayer {

    private ThreadLocal<Memory<double[]>> hiddenStateMemory = new ThreadLocal<>();
    private ThreadLocal<Memory<double[]>> memFeat = new ThreadLocal<>();
    private ThreadLocal<Memory<double[]>> outMemory = new ThreadLocal<>();
    private ThreadLocal<Memory<double[]>> expectedErrors = new ThreadLocal<>();
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
        super(hiddenLayerNeuronSize, new HyperbolicTangentFunction(), true);

        this.memorySize = memorySize;

//        isDropoutEnabled = false;

        historyLayer = new HiddenNeuronLayer(hiddenLayerNeuronSize, new LinearFunction(), false);
        outLayer = new HiddenNeuronLayer(outLayerNeuronSize, new SoftmaxFunction(), true);
    }

    private void initThreadLocals(int memorySize) {
        if (hiddenStateMemory.get() == null) {
            hiddenStateMemory.set(new Memory<>(memorySize));
        }
        if (memFeat.get() == null) {
            memFeat.set(new Memory<>(memorySize));
        }
        if (outMemory.get() == null) {
            outMemory.set(new Memory<>(memorySize));
        }
        if (expectedErrors.get() == null) {
            expectedErrors.set(new Memory<>(memorySize));
        }
    }

    @Override
    public double[] forwardPropagation(double[] inValues) {
        initThreadLocals(memorySize);

        memFeat.get().add(inValues); // for bptt

        double[] b = enrichFeatureWithBias(inValues);
        double[] rawResults = calculateRawResult(b);

        final double[] rawResultHistory;
        if (hiddenStateMemory.get().size() == 0) {
            rawResultHistory = historyLayer.calculateRawResult(new double[rawResults.length + 1]);
        } else {
            rawResultHistory = historyLayer.calculateRawResult(enrichFeatureWithBias(hiddenStateMemory.get().getLast()));
        }

        MatrixOperations mo = Factory.getMatrixOperations();
        double[] hiddenState = mo.sum(rawResultHistory, rawResults);

        // hh - tanh
        hiddenState = activationFunction.activate(hiddenState);
        hiddenStateMemory.get().add(hiddenState);

        // y - softmax
        double[] out = outLayer.forwardPropagation(hiddenState);
        outMemory.get().add(out);
        return out;
    }

    @Override
    public double[] backPropagation(double[] previousError) {
        MatrixOperations mo = Factory.getMatrixOperations();

        expectedErrors.get().add(previousError);
        int target = previousError.length - MLUtils.transformClassToInt(previousError);
        calculateLoss(target);

        double[] layerError = new double[neurons.size()];
        for (int i = 0; i < hiddenStateMemory.get().size(); i++) {

            // y processing
            double[] yOut = outMemory.get().get(i);
            double[] expected = expectedErrors.get().get(i);
            double[] dY = mo.copy(yOut);

            for (int j = 0; j < expected.length; j++) {
                dY[j] = yOut[j] - expected[j];
            }

            outLayer.lastInput.set(hiddenStateMemory.get().get(i)); // set time state
            outLayer.lastResult.set(yOut);
//            dY[target] -= 1;

            double[] dH = outLayer.backPropagation(dY);
            layerError = mo.sum(dH, layerError);

            // current layer dW
            lastInput.set(memFeat.get().get(i));  // set time state
            lastResult.set(hiddenStateMemory.get().get(i));
            double[] derivative = activationFunction.derivative(hiddenStateMemory.get().get(i));
            layerError = mo.scalarMultiply(layerError, derivative);
            calculateAndSaveDeltaError(layerError);

            // history layer update dW
            historyLayer.lastInput.set(i < hiddenStateMemory.get().size() - 1 ? hiddenStateMemory.get().get(i + 1) : new double[neurons.size()]); // set time state
            historyLayer.calculateAndSaveDeltaError(layerError);

            layerError = historyLayer.gradientVector(layerError);
        }

        CompletableFuture<Void> cfThis = CompletableFuture.runAsync(() -> shrinkWeightsError(neurons, hiddenStateMemory.get().size()));
        CompletableFuture<Void> cfHistory = CompletableFuture.runAsync(() -> shrinkWeightsError(historyLayer.neurons, hiddenStateMemory.get().size()));
        CompletableFuture<Void> cfOut = CompletableFuture.runAsync(() -> shrinkWeightsError(outLayer.neurons, hiddenStateMemory.get().size()));

        CompletableFuture.allOf(cfThis, cfHistory, cfOut);

        return layerError;
    }

    private void calculateLoss(int target) {
        loss.add(Math.log(outMemory.get().getLast()[target]));

        if (counter.incrementAndGet() % LOSS_BATCH_SIZE == 0) {
            System.out.println("=========== LOSS ==============");
            log.info("Loss: {}", -1 * loss.doubleValue() / counter.get());
            System.out.println("===============================");
            loss.reset();
            counter.set(0);
        }
    }

    private void shrinkWeightsError(List<Neuron> neurons, int denominator) {
        for (Neuron neuron : neurons) {
            AtomicDoubleArray weightsError = neuron.getWeightsError();
            for (int i = 0; i < weightsError.length(); i++) {
                weightsError.set(i, weightsError.get(i) / denominator);
            }
        }
    }

    @Override
    public void optimizeWeights(OptimizationAlgorithm optAlg) {
        historyLayer.optimizeWeights(optAlg);
        super.optimizeWeights(optAlg);
    }
}
