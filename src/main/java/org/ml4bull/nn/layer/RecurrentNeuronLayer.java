package org.ml4bull.nn.layer;

import lombok.extern.slf4j.Slf4j;
import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.LiniarFunction;
import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.util.Factory;
import org.ml4bull.util.Memory;

@Slf4j
public class RecurrentNeuronLayer extends HiddenNeuronLayer {

    private Memory<double[]> memory;
    private Memory<double[]> memFeat;
    private HiddenNeuronLayer featureLayer;

    /**
     * Simple recurrent neuron net layer.
     * @param neuronsCount - should be the same as for previous layer.
     */
    public RecurrentNeuronLayer(int neuronsCount, ActivationFunction activationFunction, int memorySize) {
        super(neuronsCount, activationFunction, false);

        featureLayer = new HiddenNeuronLayer(neuronsCount, new LiniarFunction(), false);
        isDropoutEnabled = false;
        memory = new Memory<>(memorySize);
        memFeat = new Memory<>(memorySize);
    }

    @Override
    public double[] forwardPropagation(double[] inValues) {
        double[] featLayerOut = featureLayer.forwardPropagation(inValues);
        memFeat.add(featLayerOut);

        double[] b = enrichFeatureWithBias(getLast());
        double[] rawResult = calculateRawResult(b);

        MatrixOperations mo = Factory.getMatrixOperations();
        double[] out = mo.sum(featLayerOut, rawResult);

        out = activationFunction.activate(out);
        memory.add(out);

        return out;
    }

    private double[] getLast() {
        if (memory.size() == 0) {
            return new double[neurons.size()];
        }
        return memory.getLast();
    }

    @Override
    public double[] backPropagation(double[] previousError) {

        // bptt
        for (int i = 1; i < memory.size(); i++) {
            featureLayer.lastInput.set(memFeat.get(i - 1));
            featureLayer.calculateAndSaveDeltaError(previousError);

            lastInput.set(memory.get(i));
            previousError = super.backPropagation(previousError);
        }

        // calculate first input element error gradient.
        featureLayer.lastInput.set(memFeat.get(memFeat.size() - 1));
        featureLayer.calculateAndSaveDeltaError(previousError);

        return previousError;
    }

    @Override
    public void optimizeWeights(OptimizationAlgorithm optAlg) {
        featureLayer.optimizeWeights(optAlg);
        super.optimizeWeights(optAlg);
    }
}
