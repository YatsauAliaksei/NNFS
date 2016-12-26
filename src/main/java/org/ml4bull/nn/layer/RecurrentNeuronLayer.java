package org.ml4bull.nn.layer;

import lombok.extern.slf4j.Slf4j;
import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.HyperbolicTangentFunction;
import org.ml4bull.algorithm.LiniarFunction;
import org.ml4bull.algorithm.OptimizationAlgorithm;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.util.Factory;
import org.ml4bull.util.Memory;

@Slf4j
public class RecurrentNeuronLayer extends HiddenNeuronLayer {

    private Memory<double[]> memory = new Memory<>(3);
    private Memory<double[]> memFeat = new Memory<>(3);
    private HiddenNeuronLayer featureLayer;

    /**
     * Simple recurrent neuron net layer.
     * @param neuronsCount - should be the same as for previous layer.
     * @param unused       - unused.
     */
    public RecurrentNeuronLayer(int neuronsCount, ActivationFunction unused) {
        super(neuronsCount, new HyperbolicTangentFunction());
        this.featureLayer = new HiddenNeuronLayer(neuronsCount, new LiniarFunction(), false);
        isDropoutEnabled = false;
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

/*        for (int i = 1; i < memory.size(); i++) {
            featureLayer.lastInput.set(memFeat.get(i - 1));
            featureLayer.calculateAndSaveDeltaError(previousError);

            lastInput.set(memory.get(i));
            previousError = super.backPropagation(previousError);

        }

        featureLayer.lastInput.set(memFeat.get(memFeat.size() - 1));
        featureLayer.calculateAndSaveDeltaError(previousError);*/
//        if (memFeat.size() > 1) {
//            lastInput.set(memory.get(memory.size() - 2));
//            previousError = super.backPropagation(previousError);
//        }

        return previousError;
    }

    @Override
    public void optimizeWeights(OptimizationAlgorithm optAlg) {
//        featureLayer.optimizeWeights(optAlg);
//        super.optimizeWeights(optAlg);
    }
}
