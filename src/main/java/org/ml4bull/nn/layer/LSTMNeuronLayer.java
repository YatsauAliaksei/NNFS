package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.HyperbolicTangentFunction;
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
    private HiddenNeuronLayer candidateGate;

    private Memory<double[]> memory;
    private Memory<double[]> featureMem;
//    private Memory<double[]> historyMem;
    private Memory<double[]> cellStateMem;
    private Memory<double[]> outMem;
//    private double[] cellState;

    private Map<HiddenNeuronLayer, HiddenNeuronLayer> lstmLayerMap;

    public LSTMNeuronLayer(int neuronSize, ActivationFunction activationFunction, int memorySize) {
        super(neuronSize, activationFunction, false);
        memory = new Memory<>(memorySize);
        featureMem = new Memory<>(memorySize);
//        historyMem = new Memory<>(memorySize);
        cellStateMem = new Memory<>(memorySize);
        outMem = new Memory<>(memorySize);
        init(neuronSize);
    }

    private void init(int neuronSize) {
        if (inputGate != null) return;

//        createNeurons(neuronSize);
        this.inputGate = new HiddenNeuronLayer(neuronSize, new SigmoidFunction(), false);
        this.outputGate = new HiddenNeuronLayer(neuronSize, new SigmoidFunction(), false);
        this.forgetGate = new HiddenNeuronLayer(neuronSize, new SigmoidFunction(), false);
        this.candidateGate = new HiddenNeuronLayer(neuronSize, new HyperbolicTangentFunction(), false);

//        this.inputGateFeat = new HiddenNeuronLayer(neuronSize, new LinearFunction(), false);
//        this.outputGateFeat = new HiddenNeuronLayer(neuronSize, new LinearFunction(), false);
//        this.forgetGateFeat = new HiddenNeuronLayer(neuronSize, new LinearFunction(), false);
//        this.candidateFeat = new HiddenNeuronLayer(neuronSize, new LinearFunction(), false);

//        this.cellState = new double[neuronSize]; // todo <----

        lstmLayerMap = new LinkedHashMap<>();
        lstmLayerMap.put(forgetGateFeat, forgetGate);
        lstmLayerMap.put(inputGateFeat, inputGate);
        lstmLayerMap.put(outputGateFeat, outputGate);
    }

    private HyperbolicTangentFunction hyperbolicTangentFunction = new HyperbolicTangentFunction();

    @Override
    public double[] forwardPropagation(double[] f) {
        featureMem.add(f); // save
        MatrixOperations mo = Factory.getMatrixOperations();

        double[] prevOut = getLastMemory(outMem, f.length); // read previous out

        // -- first way
        double[] mergedInput = mo.concatenate(prevOut, f);
        // decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.”
        double[] forgetGateResult = forgetGate.forwardPropagation(mergedInput); // sigmoid layer called the “forget gate layer.". Output [0, 1], 0 - forget completely
        double[] lastCellState = getLastMemory(cellStateMem, f.length); // getting last Cell State
        double[] cellStateAfterForgetGate = mo.scalarMultiply(forgetGateResult, lastCellState); // what Cell State to forget decision

        // -- second way
        // decide what new information we’re going to store in the cell state
        double[] inputGateResult = inputGate.forwardPropagation(mergedInput); // sigmoid layer called the “input gate layer” decides which values we’ll update.
        double[] candidateGateResult = candidateGate.forwardPropagation(mergedInput); // tanh layer creates a vector of new candidate values, that could be added to the state.
        double[] approvedCandidates = mo.scalarMultiply(inputGateResult, candidateGateResult); // how much we decided to update each state value.

        double[] newCellState = mo.sum(approvedCandidates, cellStateAfterForgetGate);

        cellStateMem.add(newCellState); // update Cell State

        double[] outResult = outputGate.forwardPropagation(mergedInput); // sigmoid layer which decides what parts of the cell state we’re going to output.

        // we put the cell state through tanh (to push the values to be between −1 and 1)
        double[] activatedCellState = hyperbolicTangentFunction.activate(newCellState);

        outResult = mo.scalarMultiply(outResult, activatedCellState); // multiply it by the output of the outputGate, so that we only output the parts we decided to.
        outMem.add(outResult); // save last out

        return outResult;
    }

    private double[] getLastMemory(Memory<double[]> memory, int defaultSize) {
        double[] history;
        if ((history = memory.getLast()) == null) {
            history = new double[defaultSize];
        }
        return history;
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
