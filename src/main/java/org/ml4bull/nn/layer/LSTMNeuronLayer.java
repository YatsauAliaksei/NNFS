package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.HyperbolicTangentFunction;
import org.ml4bull.algorithm.LinearFunction;
import org.ml4bull.algorithm.SigmoidFunction;
import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.util.Factory;
import org.ml4bull.util.Memory;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

public class LSTMNeuronLayer extends HiddenNeuronLayer {

    private HiddenNeuronLayer inputGateFeat;
    private HiddenNeuronLayer outputGateFeat;
    private HiddenNeuronLayer forgetGateFeat;
    private HiddenNeuronLayer candidateFeat;

    private HiddenNeuronLayer inputGate;
    private HiddenNeuronLayer outputGate;
    private HiddenNeuronLayer forgetGate;

    private Memory<double[]> memory;
    private Memory<double[]> featureMem;
    private double[] cellState;

    private Map<HiddenNeuronLayer, HiddenNeuronLayer> lstmLayerMap;

    public LSTMNeuronLayer(ActivationFunction activationFunction, int memorySize) {
        super(0, activationFunction, false);
        memory = new Memory<>(memorySize);
        featureMem = new Memory<>(memorySize);
    }

    private void init(int inputSize) {
        if (inputGate != null) return;

        createNeurons(inputSize);
        this.inputGate = new HiddenNeuronLayer(inputSize, new SigmoidFunction(), false);
        this.outputGate = new HiddenNeuronLayer(inputSize, new SigmoidFunction(), false);
        this.forgetGate = new HiddenNeuronLayer(inputSize, new SigmoidFunction(), false);

        this.inputGateFeat = new HiddenNeuronLayer(inputSize, new LinearFunction(), false);
        this.outputGateFeat = new HiddenNeuronLayer(inputSize, new LinearFunction(), false);
        this.forgetGateFeat = new HiddenNeuronLayer(inputSize, new LinearFunction(), false);
        this.candidateFeat = new HiddenNeuronLayer(inputSize, new LinearFunction(), false);
        this.cellState = new double[inputSize];

        lstmLayerMap = new LinkedHashMap<>();
        lstmLayerMap.put(forgetGateFeat, forgetGate);
        lstmLayerMap.put(inputGateFeat, inputGate);
        lstmLayerMap.put(outputGateFeat, outputGate);
    }

    @Override
    public double[] forwardPropagation(double[] f) {
        init(f.length);

        featureMem.add(f);

        // calculate forget layer
        double[] fLO = gateCalculate(forgetGateFeat, forgetGate, f);
        // calculate input layer
        double[] iLO = gateCalculate(inputGateFeat, inputGate, f);
        // calculate output layer
        double[] oLO = gateCalculate(outputGateFeat, outputGate, f);

        MatrixOperations mo = Factory.getMatrixOperations();
        cellState = mo.scalarMultiply(fLO, cellState);

        // calculate new cellState
        double[] cellStateCandidates = gateCalculate(candidateFeat, this, f);

        // approving new cellState
        double[] approvedCandidates = mo.scalarMultiply(iLO, cellStateCandidates);
        // save new cell state
        cellState = mo.sum(cellState, approvedCandidates);

        // truncating values to be between -1 / 1.
        HyperbolicTangentFunction hyperbolicTangentFunction = new HyperbolicTangentFunction();
//        SigmoidFunction hyperbolicTangentFunction = new SigmoidFunction();
        double[] truncated = hyperbolicTangentFunction.activate(cellState);
        double[] hT = mo.scalarMultiply(oLO, truncated);
        memory.add(hT);

        return hT;
    }

    @Override
    public double[] backPropagation(double[] previousError) {
        double[] err = previousError;
        for (Map.Entry<HiddenNeuronLayer, HiddenNeuronLayer> entry : lstmLayerMap.entrySet()) {
            err = previousError;
            for (int i = 1; i < memory.size(); i++) {
                err = calculateError(entry.getKey(), entry.getValue(), i, err);
            }
        }

        lstmLayerMap.forEach((k, v) -> {
            k.lastInput.set(featureMem.get(featureMem.size() - 1));
            k.calculateAndSaveDeltaError(previousError);
        });

        return err;
    }

    private double[] calculateError(HiddenNeuronLayer featureLayer, HiddenNeuronLayer gateLayer, int memoryIndex, double[] previousError) {
        double[] fm = featureMem.get(memoryIndex - 1);
        featureLayer.lastInput.set(fm);
        featureLayer.calculateAndSaveDeltaError(previousError);

        double[] m = memory.get(memoryIndex);
        gateLayer.lastInput.set(m);
        previousError = gateLayer.backPropagation(previousError);
        return previousError;
    }

    private double[] gateCalculate(NeuronLayer featLayer, NeuronLayer gateLayer, double[] input) {
        double[] fTF = featLayer.forwardPropagation(input);
        double[] fL = enrichFeatureWithBias(getLast());
        double[] rawResult = gateLayer.calculateRawResult(fL);

        MatrixOperations mo = Factory.getMatrixOperations();
        double[] out = mo.sum(fTF, rawResult);
        return gateLayer.activate(out);
    }

    private double[] getLast() {
        return Optional.ofNullable(memory.getLast())
                .orElse(new double[neurons.size()]);
    }

    @Override
    public void optimizeWeights(OptimizationAlgorithm optAlg) {
        lstmLayerMap.forEach((k, v) -> {
            k.optimizeWeights(optAlg);
            v.optimizeWeights(optAlg);
        });
    }
}
