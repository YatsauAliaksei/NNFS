package org.ml4bull.quiz;

import com.google.common.base.Stopwatch;
import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.ml4bull.algorithm.SigmoidFunction;
import org.ml4bull.algorithm.SoftmaxFunction;
import org.ml4bull.algorithm.StepFunction;
import org.ml4bull.algorithm.optalg.GradientDescent;
import org.ml4bull.algorithm.optalg.RMSPropGradientDescent;
import org.ml4bull.nn.MultiLayerPerceptron;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.layer.HiddenNeuronLayer;

import java.util.Arrays;

@Slf4j
public class FizzBuzzNN {

    @Test
    public void main() {
        Stopwatch stopwatch = Stopwatch.createStarted();
        FizzBuzzNN fb = new FizzBuzzNN();
        DataSet trainSet = fb.getTrainSet();

        GradientDescent optAlg = RMSPropGradientDescent.builder()
                .learningRate(0.5)
                .batchSize(80)
                .build();

        MultiLayerPerceptron sp = MultiLayerPerceptron.builder()
                .input(2)
                .output(4)
                .outActFunc(new SoftmaxFunction())
                .optAlg(optAlg)
                .build();

        sp.addHiddenLayer(new HiddenNeuronLayer(20, new SigmoidFunction()));

        double error;
        int epoch = 0;
        do {
            error = sp.train(trainSet, true);
            log.info("Epoch: {} | Error: {}", ++epoch, +error);
        } while (error > 1e-1);

        DataSet testSet = fb.getTestSet();
        sp.classify(testSet, false, (i, calc, ideal) -> System.out.println(backConvert(calc, i)));

        log.info("Time overall {}", stopwatch.stop().toString());
        System.exit(0);
    }

    public DataSet getTestSet() {
        return getTrainSet(1, 100);
    }

    public DataSet getTrainSet() {
        return getTrainSet(100, 10240);
    }

    public DataSet getTrainSet(int start, int end) {
        double[][] input = new double[end - start][];
        double[][] output = new double[end - start][];
        for (int i = start; i < end; i++) {
            input[i - start] = new double[]{i % 3.0, i % 5.0};
            output[i - start] = result(i);
        }
        return new DataSet(input, output);
    }

    private String backConvert(double[] d, int number) {
        StepFunction af = new StepFunction();
        d = af.activate(d);
        if (Arrays.equals(d, on3)) {
            return "fizz";
        } else if (Arrays.equals(d, on5)) {
            return "buzz";
        } else if (Arrays.equals(d, on15)) {
            return "fizzbuzz";
        } else {
            return String.valueOf(++number);
        }
    }

    private double[] result(int i) {
        if (i == 0) {
            return self;
        }
        if (i % 15 == 0) {
            return on15;
        } else if (i % 3 == 0) {
            return on3;
        } else if (i % 5 == 0) {
            return on5;
        } else {
            return self;
        }
    }

    private static final double[] on3 = {0, 0, 0, 1};
    private static final double[] on5 = {0, 0, 1, 0};
    private static final double[] on15 = {0, 1, 0, 0};
    private static final double[] self = {1, 0, 0, 0};
}
