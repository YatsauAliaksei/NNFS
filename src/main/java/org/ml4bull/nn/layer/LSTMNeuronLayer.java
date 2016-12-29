package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.HyperbolicTangentFunction;
import org.ml4bull.algorithm.SigmoidFunction;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.util.Factory;
import org.ml4bull.util.Memory;

public class LSTMNeuronLayer extends HiddenNeuronLayer {

    private NeuronLayer inputGate;
    private NeuronLayer outputGate;
    private NeuronLayer forgetGate;

    private Memory<double[]> memory = new Memory<>(100);
    private Memory<double[]> featureMem = new Memory<>(100);
    private double[] candidates;

    public LSTMNeuronLayer(int neuronsCount, ActivationFunction ignored) {
        super(neuronsCount, new HyperbolicTangentFunction(), false);
        int concatSize = neuronsCount * 2;
        this.inputGate = new HiddenNeuronLayer(concatSize, new SigmoidFunction(), false);
        this.outputGate = new HiddenNeuronLayer(concatSize, new SigmoidFunction(), false);
        this.forgetGate = new HiddenNeuronLayer(concatSize, new SigmoidFunction(), false);
        this.candidates = new double[concatSize];
    }

    public LSTMNeuronLayer(int neuronsCount, ActivationFunction ignored1, boolean ignored2) {
        this(neuronsCount, null);
    }

    @Override
    public double[] forwardPropagation(double[] f) {
        featureMem.add(f);
        MatrixOperations mo = Factory.getMatrixOperations();

        double[] concatValues = mo.concatenate(memory.getLast(), f);
        // forget info
        double[] fT = forgetGate.forwardPropagation(concatValues);
        candidates = mo.scalarMultiply(fT, candidates);

        // calculate new candidates
        double[] newCandidates = super.forwardPropagation(concatValues);
        // approving new candidates
        double[] iT = inputGate.forwardPropagation(concatValues);
        double[] approvedCandidates = mo.scalarMultiply(iT, newCandidates);
        // save new candidate state
        candidates = mo.sum(candidates, approvedCandidates);

        // truncating values to be between -1 / 1.
        HyperbolicTangentFunction hyperbolicTangentFunction = new HyperbolicTangentFunction();
        double[] truncated = hyperbolicTangentFunction.activate(candidates);
        double[] oT = outputGate.forwardPropagation(concatValues);
        double[] hT = mo.scalarMultiply(oT, truncated);
        memory.add(hT);

        return hT;
    }

    @Override
    public double[] backPropagation(double[] previousError) {


        return super.backPropagation(previousError);
    }
}
