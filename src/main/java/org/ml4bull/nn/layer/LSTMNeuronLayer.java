package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.HyperbolicTangentFunction;
import org.ml4bull.algorithm.SigmoidFunction;
import org.ml4bull.algorithm.SoftmaxFunction;
import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.util.Factory;
import org.ml4bull.util.MLUtils;
import org.ml4bull.util.Memory;

import java.util.stream.Stream;

import static org.ml4bull.util.MatrixUtils.createMatrix;

public class LSTMNeuronLayer extends HiddenNeuronLayer {

    protected final MatrixOperations mo = Factory.getMatrixOperations();
    final HyperbolicTangentFunction hyperbolicTangentFunction = new HyperbolicTangentFunction();

    HiddenNeuronMemoryLayer inputGate, outputGate, forgetGate, candidateGate, probLayer;

    Memory<double[]> expectedMem, cMem, mergedMem;
    Memory<double[]> hcMem, hMem, hoMem, yMem, hfMem, hiMem;
    final int netSize;
    protected final int batchSize;
    public int counter;

    public LSTMNeuronLayer(int neuronSize, int outSize, ActivationFunction activationFunction, int memorySize, int batchSize) {
        super(0, activationFunction, false, false);
        this.netSize = neuronSize;
        this.batchSize = batchSize;
        int weightsMemorySize = memorySize < batchSize ? 1 : memorySize / batchSize;
        weightsMemorySize += memorySize % batchSize == 0 ? 0 : 1;

        initMemories(memorySize);
        initLayers(neuronSize, outSize, weightsMemorySize);
    }

    private void initMemories(int memorySize) {
        cMem = new Memory<>(memorySize);
        hMem = new Memory<>(memorySize);
        hcMem = new Memory<>(memorySize);
        hoMem = new Memory<>(memorySize);
        yMem = new Memory<>(memorySize);
        hfMem = new Memory<>(memorySize);
        hiMem = new Memory<>(memorySize);
        mergedMem = new Memory<>(memorySize);
        expectedMem = new Memory<>(memorySize);
    }

    private void initLayers(int neuronSize, int outSize, int memorySize) {
        this.inputGate = new HiddenNeuronMemoryLayer(neuronSize, new SigmoidFunction(), true, memorySize, false);
        this.outputGate = new HiddenNeuronMemoryLayer(neuronSize, new SigmoidFunction(), true, memorySize, false);
        this.forgetGate = new HiddenNeuronMemoryLayer(neuronSize, new SigmoidFunction(), true, memorySize, false);
        this.candidateGate = new HiddenNeuronMemoryLayer(neuronSize, new HyperbolicTangentFunction(), true, memorySize, false);
        this.probLayer = new HiddenNeuronMemoryLayer(outSize, new SoftmaxFunction(), true, memorySize, false);
    }

    @Override
    public double[] forwardPropagation(double[] f) {
        counter++;
        double[] prevOut = getLastMemory(hMem, netSize); // read previous out
        // concatenate prev out and current input
        double[] mergedInput = mo.concatenate(prevOut, f);
        mergedMem.add(mergedInput);

        // hf * c_old
        // decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.”
        double[] hf = forgetGate.forwardPropagation(mergedInput); // sigmoid layer called the “forget gate layer.". Output [0, 1], 0 - forget completely
        hfMem.add(hf);

        // hi * hc
        // decide what new information we’re going to store in the cell state
        double[] hi = inputGate.forwardPropagation(mergedInput); // sigmoid layer called the “input gate layer” decides which values we’ll update.
        hiMem.add(hi);
        // hc
        double[] hc = candidateGate.forwardPropagation(mergedInput); // tanh layer creates a vector of new candidate values, that could be added to the state.
        hcMem.add(hc);

        double[] c_old = getLastMemory(cMem, netSize); // getting last Cell State
        // c = hf * c_old + hi * hc
        double[] c = mo.sum(mo.scalarMultiply(hi, hc), mo.scalarMultiply(hf, c_old)); // new cell state
        cMem.add(c);

        double[] ho = outputGate.forwardPropagation(mergedInput); // sigmoid layer which decides what parts of the cell state we’re going to output.
        hoMem.add(ho);

        double[] h = mo.scalarMultiply(ho, hyperbolicTangentFunction.activate(c)); // multiply it by the output of the outputGate, so that we only output the parts we decided to.
        hMem.add(h);

        double[] y = probLayer.forwardPropagation(h);
        yMem.add(y);
        return y;
    }

    @Override
    public double[] backPropagation(double[] previousError) {
        expectedMem.add(previousError);

        int netWeightsSize = forgetGate.getNeurons().get(0).getWeights().length;

        double[][] dWf = createMatrix(netSize, netWeightsSize);
        double[][] dWi = createMatrix(netSize, netWeightsSize);
        double[][] dWo = createMatrix(netSize, netWeightsSize);
        double[][] dWc = createMatrix(netSize, netWeightsSize);
        double[][] dWy = createMatrix(probLayer.getNeurons().size(), netSize);

        double[] dBf = createMatrix(netSize);
        double[] dBi = createMatrix(netSize);
        double[] dBo = createMatrix(netSize);
        double[] dBc = createMatrix(netSize);
        double[] dBy = createMatrix(probLayer.getNeurons().size());

        double[] dhNext = createMatrix(hMem.getLast().length);
        double[] dcNext = createMatrix(cMem.getLast().length);

        final int time = yMem.size();
        for (int i = 0; i < time; i++) {
            // y processing
            double[] dY = mo.copy(yMem.get(i));

            double[] expected = expectedMem.get(i);
            int target = expected.length - MLUtils.transformClassToInt(expected);
            dY[target] -= 1;

            int wTime = (counter - i) > 0 ? 0 : 1 + (i - counter) / batchSize; // todo: check it
            double[] dh = mo.sum(probLayer.gradientVectorInTime(dY, wTime), dhNext);
//            dh = hyperbolicTangentFunction.activate(dh); // todo: tmp
            // Derivative: h = ho * tanh(c)
            double[] c = cMem.get(i);
            double[] dho = mo.scalarMultiply(dh, hyperbolicTangentFunction.activate(c)); // dh * tanh(c)
            double[] dc = mo.scalarMultiply(dh, mo.scalarMultiply(hoMem.get(i), hyperbolicTangentFunction.derivative(c))); // dh * ho * dtang(c)
            dcNext = mo.sum(dcNext, dc);
            dcNext = hyperbolicTangentFunction.activate(dcNext); // todo: tmp

            // Derivative: c = hi * hc + c_old * hf
            double[] c_old = i < time - 1 ? cMem.get(i + 1) : new double[c.length];

            double[] dhi = mo.scalarMultiply(dcNext, hcMem.get(i));
            double[] dhc = mo.scalarMultiply(dcNext, hiMem.get(i));
            double[] dhf = mo.scalarMultiply(dcNext, c_old);

            double[] dhcGrad = mo.scalarMultiply(dhc, candidateGate.activationFunction.derivative(hcMem.get(i)));
            double[] dhiGrad = mo.scalarMultiply(dhi, inputGate.activationFunction.derivative(hiMem.get(i)));
            double[] dhfGrad = mo.scalarMultiply(dhf, forgetGate.activationFunction.derivative(hfMem.get(i)));
            double[] dhoGrad = mo.scalarMultiply(dho, outputGate.activationFunction.derivative(hoMem.get(i)));

            double[] lastIn = mergedMem.get(i);
            candidateGate.lastInput.set(lastIn);
            inputGate.lastInput.set(lastIn);
            forgetGate.lastInput.set(lastIn);
            outputGate.lastInput.set(lastIn);
            probLayer.lastInput.set(hMem.get(i));

            dWc = mo.sum(candidateGate.calculateDeltaError(dhcGrad), dWc);
            dWi = mo.sum(inputGate.calculateDeltaError(dhiGrad), dWi);
            dWf = mo.sum(forgetGate.calculateDeltaError(dhfGrad), dWf);
            dWo = mo.sum(outputGate.calculateDeltaError(dhoGrad), dWo);
            dWy = mo.sum(probLayer.calculateDeltaError(dY), dWy);

            dBc = mo.sum(candidateGate.calculateBiasDeltaError(dhcGrad), dBc);
            dBi = mo.sum(inputGate.calculateBiasDeltaError(dhiGrad), dBi);
            dBf = mo.sum(forgetGate.calculateBiasDeltaError(dhfGrad), dBf);
            dBo = mo.sum(outputGate.calculateBiasDeltaError(dhoGrad), dBo);
            dBy = mo.sum(probLayer.calculateBiasDeltaError(dY), dBy);

            double[] dXc = candidateGate.gradientVectorInTime(dhcGrad, wTime);
            double[] dXi = inputGate.gradientVectorInTime(dhiGrad, wTime);
            double[] dXf = forgetGate.gradientVectorInTime(dhfGrad, wTime);
            double[] dXo = outputGate.gradientVectorInTime(dhoGrad, wTime);

            double[] dX = Stream.of(dXc, dXi, dXf, dXo).reduce(mo::sum).get();
            // dhNext update
            System.arraycopy(dX, 0, dhNext, 0, dhNext.length);
            dcNext = mo.scalarMultiply(dcNext, hfMem.get(i));

            MLUtils.shrink(4, dhNext);// hyperbolicTangentFunction.activate(dhNext); // todo: tmp
//            dcNext = hyperbolicTangentFunction.activate(dcNext); // todo: tmp
        }

        MLUtils.shrink(time, dWf, dWi, dWo, dWc, dWy);
        MLUtils.shrink(time, dBf, dBi, dBo, dBc, dBy);

        // weights
        forgetGate.addErrorsToNeurons(dWf);
        inputGate.addErrorsToNeurons(dWi);
        outputGate.addErrorsToNeurons(dWo);
        candidateGate.addErrorsToNeurons(dWc);
        probLayer.addErrorsToNeurons(dWy);

        // biases
        forgetGate.addBiasErrorsToNeurons(dBf);
        inputGate.addBiasErrorsToNeurons(dBi);
        outputGate.addBiasErrorsToNeurons(dBo);
        candidateGate.addBiasErrorsToNeurons(dBc);
        probLayer.addBiasErrorsToNeurons(dBy);

        return dhNext;
    }

    double[] getLastMemory(Memory<double[]> memory, int defaultSize) {
        double[] history;
        if ((history = memory.getLast()) == null) {
            history = new double[defaultSize];
        }
        return history;
    }

    @Override
    public void optimizeWeights(OptimizationAlgorithm optAlg) {
        forgetGate.optimizeWeights(optAlg);
        inputGate.optimizeWeights(optAlg);
        candidateGate.optimizeWeights(optAlg);
        outputGate.optimizeWeights(optAlg);
        probLayer.optimizeWeights(optAlg);
        counter = 0;
    }
}
