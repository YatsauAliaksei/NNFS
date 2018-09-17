package org.ml4bull.nn;

import com.google.common.base.Stopwatch;
import lombok.Builder;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang.time.StopWatch;
import org.ml4bull.algorithm.ActivationFunction;
import org.ml4bull.algorithm.StepFunction;
import org.ml4bull.algorithm.optalg.OptimizationAlgorithm;
import org.ml4bull.nn.data.Data;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.data.Printer;
import org.ml4bull.nn.layer.InputNeuronLayer;
import org.ml4bull.nn.layer.NeuronLayer;
import org.ml4bull.nn.layer.OutputNeuronLayer;
import org.ml4bull.util.MLUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import static org.ml4bull.util.MathUtils.log2;

@Slf4j
public class MultiLayerPerceptron implements SupervisedNeuralNetwork {

    private final NeuronLayer inputLayer;
    private final NeuronLayer outputLayer;
    private final List<NeuronLayer> perceptronLayers;
    private final Semaphore semaphore;
    private final OptimizationAlgorithm optAlg;
    private AtomicInteger successCounter = new AtomicInteger();
    private StepFunction sf = new StepFunction();

    @Builder
    public MultiLayerPerceptron(int input, int output, NeuronLayer outputLayer, OptimizationAlgorithm optAlg, ActivationFunction outActFunc) {
        this.optAlg = optAlg;
        this.inputLayer = new InputNeuronLayer(input);

        if (outputLayer == null) {
            this.outputLayer = new OutputNeuronLayer(output, outActFunc);
        } else
            this.outputLayer = outputLayer;

        this.semaphore = new Semaphore(optAlg.getBatchSize());
        this.perceptronLayers = new ArrayList<>();
        this.perceptronLayers.add(this.outputLayer);
    }

    @Override
    public MultiLayerPerceptron addHiddenLayer(NeuronLayer hiddenLayer) {
        perceptronLayers.add(perceptronLayers.size() - 1, hiddenLayer);
        return this;
    }

    @Override
    public double[][] classify(DataSet dataSet, boolean isParallel, Printer printer) {
        double[][] result = new double[dataSet.getInput().length][];
        IntStream is = IntStream.range(0, dataSet.getInput().length);

        if (isParallel)
            is = is.parallel();

        is.forEach(i -> {
                    result[i] = process(dataSet.getInput()[i]);
                    printer.print(i, result[i], dataSet.getOutput()[i]);
                }
        );
        return result;
    }

    public void training(DataSet trainDS, double errorGoal, Consumer<Double> consumer) {
        double error;
        int epoch = 0;
        Stopwatch stopwatch = Stopwatch.createUnstarted();
        do {
            stopwatch.start();
            error = train(trainDS, false);
            log.info("Epoch: {} | Error: {}", ++epoch, error);

            stopwatch.stop();
            System.out.println("Time: " + stopwatch.toString());
            stopwatch.reset();
            consumer.accept(error);
        } while (error > errorGoal);
    }

    public double[] classify(double[] data) {
        return process(data);
    }

    private double[] process(double[] data) {
        double[] v = inputLayer.forwardPropagation(data);
        for (NeuronLayer layer : perceptronLayers) {
            v = layer.forwardPropagation(v);
        }
        return v;
    }

    @Override
    public double train(DataSet dataSet, boolean isParallel) {
        return train(dataSet.getInput(), dataSet.getOutput(), isParallel);
    }

    public double train(List<Data> dataSet, boolean isParallel) {
        final int dataSize = dataSet.size();

        DoubleStream ds = dataSet.stream()
                .mapToDouble(data -> calculateAndGetItemError(data.getInput(), data.getOutput()));
        double error = run(isParallel, ds);

        log.info("Success rate: {}/{}", successCounter.get(), dataSize);
        successCounter.set(0);

        return -error / dataSize;
    }

    private double train(double[][] data, double[][] expected, boolean isParallel) {
        final int dataSize = data.length;

        if (dataSize == 0) {
            log.error("DataSize couldn't be empty");
            throw new RuntimeException("DataSize couldn't be empty");
        }

        DoubleStream ds = IntStream.range(0, dataSize)
                .mapToDouble(i -> calculateAndGetItemError(data[i], expected[i]));
        double error = run(isParallel, ds);

        log.info("Success rate: {}/{}", successCounter.get(), dataSize);
        successCounter.set(0);
        return -error / dataSize;
    }

    private double run(boolean isParallel, DoubleStream ds) {
        if (isParallel)
            ds = ds.parallel().unordered();

        return ds.sum();
    }

    @SneakyThrows(InterruptedException.class)
    private double calculateAndGetItemError(double[] input, double[] output) {
        StopWatch watch = null;
        if (log.isDebugEnabled()) {
            watch = new StopWatch();
        }

        semaphore.acquire();

        // predict
        double[] calcY = process(input);

        // Back propagation for hidden layers
        List<NeuronLayer> revList = new ArrayList<>(perceptronLayers);
        Collections.reverse(revList);

        double[] errorOut = output;
        for (NeuronLayer aRevList : revList) {
            errorOut = aRevList.backPropagation(errorOut);
        }
        tryUpdateWeights();

        calcSuccess(output, calcY);

        double v = itemCostFunction(calcY, output);

        if (watch != null) {
            watch.stop();
            log.debug("Total time: {}", watch.toString());
        }

        return v;
    }

    private void calcSuccess(double[] output, double[] calcY) {
        double[] predicted = sf.activate(calcY);
        int p = MLUtils.transformClassToInt(predicted);
        int o = MLUtils.transformClassToInt(output);
        if (o == p)
            successCounter.incrementAndGet();
    }

    private Lock lock = new ReentrantLock();

    private void tryUpdateWeights() {
        if (!optAlg.isLimitReached()) return;

        if (lock.tryLock()) {
            optimize();
            semaphore.release(optAlg.getBatchSize());

            lock.unlock();
        } else {
            log.info("=============  SHOULD BE IMPOSSIBLE !!!!!!!!!!! ================== 3");
//            throw new RuntimeException();
        }
    }

    private void optimize() {
        perceptronLayers.forEach(l -> l.optimizeWeights(optAlg));
    }

    // cross-entropy
    private double itemCostFunction(double[] calculated, double[] expected) {
        return IntStream.range(0, calculated.length)
                .mapToDouble(i ->
                        expected[i] * log2(calculated[i]) + (1 - expected[i]) * log2(1 - calculated[i])
                ).parallel().unordered().sum();
    }
}
