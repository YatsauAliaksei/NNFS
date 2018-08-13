package org.ml4bull.nn.layer;

import lombok.extern.slf4j.Slf4j;
import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.util.Factory;
import org.ml4bull.util.MLUtils;
import org.ml4bull.util.Memory;

@Slf4j
public class RecurrentNeuronLayer extends HiddenNeuronLayer {

    private Memory<double[]> memory;
    private Memory<double[]> memFeat;
//    private HiddenNeuronLayer featureLayer;

    /**
     * Simple recurrent neuron net layer.
     */
    public RecurrentNeuronLayer(ActivationFunction activationFunction, int neuronSize, int memorySize) {
//        super(0, activationFunction, false);
        super(neuronSize, activationFunction, false);

//        isDropoutEnabled = false;
        memory = new Memory<>(memorySize);
        memFeat = new Memory<>(memorySize);

//        featureLayer = new HiddenNeuronLayer(neuronSize, activationFunction, false);
//        super.createNeurons(neuronSize);
    }

/*    private void init(int featuresCount) {
        if (featureLayer != null) return;

        featureLayer = new HiddenNeuronLayer(featuresCount, new LiniarFunction(), false);
        super.createNeurons(featuresCount);
    }*/

    // TODO: remove
    private char vectorToChar(double[] v) {
        int maxIndex = 0;
        for (int i = 0; i < v.length; i++) {
            if (v[i] > v[maxIndex])
                maxIndex = i;
        }

        double[] predictedV = new double[27];
        predictedV[maxIndex] = 1;

        if (maxIndex == 0)
            return ' ';

        return (char) (MLUtils.transformClassToInt(predictedV) + 96);
    }

    @Override
    public double[] forwardPropagation(double[] inValues) {
//        init(inValues.length);

        if (log.isDebugEnabled()) {
            char arg = vectorToChar(inValues);
            if (arg == '{')
                System.out.println(" STOP ");
            System.out.print(arg + ",");
//            log.debug("Forward: {}", arg);
        }

        memFeat.add(inValues);
        lastInput.set(inValues);

        double[] b = enrichFeatureWithBias(inValues);
        double[] rawResults = calculateRawResult(b);

//        double[] bH = enrichFeatureWithBias(getLast());
        double[] rawResultHistory;// = calculateRawResult(memFeat.getLast());
        if (memFeat.size() == 0) {
            rawResultHistory = new double[inValues.length];
        } else {
            double[] bH = enrichFeatureWithBias(memFeat.getLast());
            rawResultHistory = calculateRawResult(bH);
        }

        MatrixOperations mo = Factory.getMatrixOperations();
        double[] out = mo.sum(rawResultHistory, rawResults);

        out = activationFunction.activate(out);
        memory.add(out);

        return out;
    }

/*    private double[] getLast() {
        if (memFeat.size() == 0) {
            return new double[size];
        }
        return memFeat.getLast();
    }*/

    @Override
    public double[] backPropagation(double[] previousError) {
        MatrixOperations mo = Factory.getMatrixOperations();
        double[] layerError = new double[previousError.length];
        for (int i = 0; i < memory.size(); i++) {
            // memory in/out
//            featureLayer.
            lastInput.set(memFeat.get(i));
//            featureLayer.
            lastResult.set(memory.get(i));

            double[] derivative =
//                    featureLayer.
                    activationFunction.derivative(
//                                    featureLayer.
                            lastResult.get());
            previousError = mo.scalarMultiply(previousError, derivative);

            layerError = mo.sum(layerError, previousError);
        }
        mo.scalarMultiply(layerError, 1 / memory.size());

//        featureLayer.
        calculateAndSaveDeltaError(layerError);
        return
//                featureLayer.
                calculateLayerError(layerError);
//        return previousError;

/*
        // bptt
        for (int i = 1; i < memory.size(); i++) {
            featureLayer.lastInput.set(memFeat.get(i - 1));
            featureLayer.calculateAndSaveDeltaError(layerError);

            lastInput.set(memory.get(i));
            double[] error = featureLayer.backPropagation(layerError);
        }

        // calculate first input element error gradient.
        featureLayer.lastInput.set(memFeat.get(memFeat.size() - 1));
        featureLayer.calculateAndSaveDeltaError(previousError);

        return featureLayer.calculateLayerError(layerError);
//        return previousError;*/
    }

//    @Override
//    public void optimizeWeights(OptimizationAlgorithm optAlg) {
//        featureLayer.optimizeWeights(optAlg);
//        super.optimizeWeights(optAlg);
//    }
}
