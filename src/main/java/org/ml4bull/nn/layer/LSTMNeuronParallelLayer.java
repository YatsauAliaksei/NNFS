package org.ml4bull.nn.layer;

import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.util.MLUtils;
import org.ml4bull.util.Memory;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Stream;

import static org.ml4bull.util.MatrixUtils.createMatrix;

public class LSTMNeuronParallelLayer extends LSTMNeuronLayer {

    private final ForkJoinPool fjp = new ForkJoinPool(Runtime.getRuntime().availableProcessors());

    public LSTMNeuronParallelLayer(int neuronSize, int outSize, ActivationFunction activationFunction, int memorySize, int batchSize) {
        super(neuronSize, outSize, activationFunction, memorySize, batchSize);
    }

    @Override
    public double[] forwardPropagation(double[] f) {
        counter++; // fixme: ugly
        double[] prevOut = getLastMemory(hMem, netSize); // read previous out
        // concatenate prev out and current input
        double[] mergedInput = mo.concatenate(prevOut, f);
        mergedMem.add(mergedInput);

        // hf * c_old
        CompletableFuture<double[]> cellStateAfterForgetGateCF = CompletableFuture.supplyAsync(() -> {
            // -- first way
            // decide what information we’re going to throw away from the cell state. This decision is made by a sigmoid layer called the “forget gate layer.”
            double[] hf = forgetGate.forwardPropagation(mergedInput); // sigmoid layer called the “forget gate layer.". Output [0, 1], 0 - forget completely
            hfMem.add(hf);
            double[] c_old = getLastMemory(cMem, netSize); // getting last Cell State
            return mo.scalarMultiply(hf, c_old); // what Cell State to forget decision
        }, fjp);

        // hi * hc
        CompletableFuture<double[]> activatedCellStateCF = CompletableFuture.supplyAsync(() -> {
            // -- second way
            // decide what new information we’re going to store in the cell state
            double[] hi = inputGate.forwardPropagation(mergedInput); // sigmoid layer called the “input gate layer” decides which values we’ll update.
            hiMem.add(hi);

            // hc
            double[] hc = candidateGate.forwardPropagation(mergedInput); // tanh layer creates a vector of new candidate values, that could be added to the state.
            hcMem.add(hc);
            return mo.scalarMultiply(hi, hc); // how much we decided to update each state value.
        }, fjp)
                // concat ways
                .thenCombine(cellStateAfterForgetGateCF, (approvedCandidates, cellStateAfterForgetGate) -> {
                    // c = hf * c_old + hi * hc
                    double[] c = mo.sum(approvedCandidates, cellStateAfterForgetGate); // new cell state
                    cMem.add(c);
                    return c;
                })
                // we put the cell state through tanh (to push the values to be between −1 and 1)
                .thenApply(hyperbolicTangentFunction::activate);

        return CompletableFuture.supplyAsync(() -> {
            double[] ho = outputGate.forwardPropagation(mergedInput); // sigmoid layer which decides what parts of the cell state we’re going to output.
            hoMem.add(ho);
            return ho;
        }, fjp)
                .thenCombine(activatedCellStateCF, (ho, activatedCellState) -> {
                    double[] h = mo.scalarMultiply(ho, activatedCellState); // multiply it by the output of the outputGate, so that we only output the parts we decided to.
                    hMem.add(h);
                    return h;
                })
                .thenApply(h -> {
                    double[] y = probLayer.forwardPropagation(h);
                    yMem.add(y);
                    return y;
                }).join();
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

        counter = counter > batchSize ? counter - counter / batchSize * batchSize : counter; // normalize if 'check' pass happened.
        final int time = yMem.size();
        for (int i = 0; i < time; i++) {
            // y processing
            double[] dY = mo.copy(yMem.get(i));

            double[] expected = expectedMem.get(i);
            int target = expected.length - MLUtils.transformClassToInt(expected);
            dY[target] -= 1;

            int wTime = (counter - i) > 0 ? 0 : 1 + (i - counter) / batchSize; // todo: check it
            double[] dh = mo.sum(probLayer.gradientVectorInTime(dY, wTime), dhNext);
            dh = hyperbolicTangentFunction.activate(dh); // todo: tmp

            // Derivative: h = ho * tanh(c)
            double[] c = cMem.get(i);
            double[] dc = mo.scalarMultiply(dh, mo.scalarMultiply(hoMem.get(i), hyperbolicTangentFunction.derivative(c))); // dh * ho * dtang(c)
            dcNext = mo.sum(dcNext, dc);
            dcNext = hyperbolicTangentFunction.activate(dcNext); // todo: tmp

            // Derivative: c = hi * hc + c_old * hf
            Object[] tmpBoxC = new Object[2]; // dirty...
            Object[] tmpBoxI = new Object[2];
            Object[] tmpBoxF = new Object[2];
            Object[] tmpBoxO = new Object[2];
            Object[] tmpBoxY = new Object[2];

            double[] lastIn = mergedMem.get(i);
            double[] c_old = i < time - 1 ? cMem.get(i + 1) : new double[c.length];

            CompletableFuture<double[]> cfC = processGate(dWc, dBc, wTime, lastIn, i, dcNext, tmpBoxC, hiMem.get(i), candidateGate, hcMem);
            CompletableFuture<double[]> cfI = processGate(dWi, dBi, wTime, lastIn, i, dcNext, tmpBoxI, hcMem.get(i), inputGate, hiMem);
            CompletableFuture<double[]> cfF = processGate(dWf, dBf, wTime, lastIn, i, dcNext, tmpBoxF, c_old, forgetGate, hfMem);
            // dh * tanh(c)
            CompletableFuture<double[]> cfO = processGate(dWo, dBo, wTime, lastIn, i, dcNext, tmpBoxO, hyperbolicTangentFunction.activate(c), outputGate, hoMem);

            // combine
            CompletableFuture.allOf(cfC, cfI, cfF, cfO).join();

            double[] dX = Stream.of(cfC, cfI, cfF, cfO).map(CompletableFuture::join).reduce(mo::sum).get();
            // dhNext update

            CompletableFuture<double[][]> cfDhDc = processNextDhDc(dhNext, dcNext, i, dX);

            dWf = (double[][]) tmpBoxF[0];
            dBf = (double[]) tmpBoxF[1];

            dWc = (double[][]) tmpBoxC[0];
            dBc = (double[]) tmpBoxC[1];

            dWi = (double[][]) tmpBoxI[0];
            dBi = (double[]) tmpBoxI[1];

            dWo = (double[][]) tmpBoxO[0];
            dBo = (double[]) tmpBoxO[1];

            processOutGate(dWy, dBy, dY, tmpBoxY, i).join();
            dWy = (double[][]) tmpBoxY[0];
            dBy = (double[]) tmpBoxY[1];

            double[][] dhDc = cfDhDc.join();
            dhNext = dhDc[0];
            dcNext = dhDc[1];
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

    private CompletableFuture<double[][]> processNextDhDc(double[] dh, double[] dc, int i, double[] dX) {
        return CompletableFuture.supplyAsync(() -> {
            double[][] tmpBoxDcDh = new double[2][];
            System.arraycopy(dX, 0, dh, 0, dh.length);
            double[] dcNext = mo.scalarMultiply(dc, hfMem.get(i));

            tmpBoxDcDh[0] = hyperbolicTangentFunction.activate(dh); // todo: tmp
            tmpBoxDcDh[1] = hyperbolicTangentFunction.activate(dcNext); // todo: tmp
            return tmpBoxDcDh;
        }, fjp);
    }

    private CompletableFuture<Void> processOutGate(double[][] dWy, double[] dBy, double[] dY, Object[] tmpBoxY, int t) {
        return CompletableFuture.runAsync(() -> {
            probLayer.lastInput.set(hMem.get(t));
            tmpBoxY[0] = mo.sum(probLayer.calculateDeltaError(dY), dWy);
            tmpBoxY[1] = mo.sum(probLayer.calculateBiasDeltaError(dY), dBy);
        });
    }

    private CompletableFuture<double[]> processGate(double[][] dW, double[] dB, int wTime, double[] lastIn, int t,
                                                    double[] finalDcNext, Object[] tmpBox, double[] hiMem,
                                                    HiddenNeuronMemoryLayer candidateGate, Memory<double[]> hcMem) {

        return CompletableFuture.supplyAsync(() -> {
            double[] dhc = mo.scalarMultiply(finalDcNext, hiMem);
            double[] dhcGrad = mo.scalarMultiply(dhc, candidateGate.activationFunction.derivative(hcMem.get(t)));
            candidateGate.lastInput.set(lastIn);
            tmpBox[0] = mo.sum(candidateGate.calculateDeltaError(dhcGrad), dW);
            tmpBox[1] = mo.sum(candidateGate.calculateBiasDeltaError(dhcGrad), dB);
            return candidateGate.gradientVectorInTime(dhcGrad, wTime);
        }, fjp);
    }

    @Override
    public void optimizeWeights(OptimizationAlgorithm optAlg) {
        CompletableFuture.allOf(
                CompletableFuture.runAsync(() -> forgetGate.optimizeWeights(optAlg), fjp),
                CompletableFuture.runAsync(() -> inputGate.optimizeWeights(optAlg), fjp),
                CompletableFuture.runAsync(() -> candidateGate.optimizeWeights(optAlg), fjp),
                CompletableFuture.runAsync(() -> outputGate.optimizeWeights(optAlg), fjp),
                CompletableFuture.runAsync(() -> probLayer.optimizeWeights(optAlg), fjp)
        ).join();
        counter = 0;
    }
}
