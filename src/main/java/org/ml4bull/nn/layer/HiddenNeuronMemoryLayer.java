package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.util.Factory;
import org.ml4bull.util.Memory;

public class HiddenNeuronMemoryLayer extends HiddenNeuronLayer {

    private MatrixOperations mo = Factory.getMatrixOperations();
    private Memory<double[][]> neuronWeightsMemory;

    public HiddenNeuronMemoryLayer(int neuronsCount, ActivationFunction activationFunction, int memorySize) {
        this(neuronsCount, activationFunction, true, memorySize);
    }

    public HiddenNeuronMemoryLayer(int neuronsCount, ActivationFunction activationFunction, boolean withBias, int memorySize) {
        super(neuronsCount, activationFunction, withBias);
        init(memorySize);
    }

    public HiddenNeuronMemoryLayer(int neuronsCount, ActivationFunction activationFunction, boolean withBias, int memorySize, boolean isDropoutEnabled) {
        super(neuronsCount, activationFunction, withBias, isDropoutEnabled);
        init(memorySize);
    }

    private void init(int memorySize) {
        neuronWeightsMemory = new Memory<>(memorySize);
    }

    protected double[] gradientVectorInTime(double[] layerError, int time) {
        double[][] theta;
        // layer weights matrix
//        double[][] theta = getLayerWeightMatrixInTime(time);
        try {
             theta = getLayerWeightMatrixInTime(time);
        } catch (NullPointerException e) {
            System.out.println("Here");
            throw e;
        }

        MatrixOperations mo = Factory.getMatrixOperations();
        // calculating gradient vector
        double[][] thetaT = mo.transpose(theta);
        return mo.multiplySingleDim(thetaT, layerError);
    }

    protected double[][] getLayerWeightMatrixInTime(int time) {
        if (time == 0 || neuronWeightsMemory.isEmpty()) {
            return createLayerWeightMatrix(); // time 0 means current. Memory doesn't contain current weights.
        }
        return neuronWeightsMemory.get(time - 1);
    }

    @Override
    public void optimizeWeights(OptimizationAlgorithm optAlg) {
        double[][] theta = super.createLayerWeightMatrix();
        for (int i = 0; i < theta.length; i++) {
            theta[i] = mo.copy(theta[i]);
        }
        neuronWeightsMemory.add(theta);

        super.optimizeWeights(optAlg);
    }
}
