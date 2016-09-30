package org.ml4bull.quiz;

import org.junit.Test;
import org.ml4bull.algorithm.SigmoidFunction;
import org.ml4bull.algorithm.StepFunction;
import org.ml4bull.nn.MultiLayerPerceptron;
import org.ml4bull.nn.data.DataSet;
import org.ml4bull.nn.layer.HiddenNeuronLayer;

import java.util.Arrays;

public class FizzBuzzNN {

    @Test
    public void main() {
        FizzBuzzNN fb = new FizzBuzzNN();
        MultiLayerPerceptron sp = new MultiLayerPerceptron(2, 4, new SigmoidFunction());
        sp.addHiddenLayer(new HiddenNeuronLayer(2, new SigmoidFunction()));

        DataSet trainSet = fb.getTrainSet();
        double error;
        int epoch = 0;
        do {
            error = sp.train(trainSet);
            System.out.println("Epoch: " + ++epoch + " | Error: " + error);
        } while (error > 0);

        DataSet testSet = fb.getTestSet();
        sp.test(testSet, (i, calc, ideal) -> System.out.println(backConvert(calc, i)));
        System.exit(0);
    }

    public DataSet getTestSet() {
        return getTrainSet(1, 100);
    }

    public DataSet getTrainSet() {
        return getTrainSet(100, 1024);
    }

    public DataSet getTrainSet(int start, int end) {
        double[][] input = new double[end - start][];
        double[][] output = new double[end - start][];
        for (int i = start; i < end; i++) {
            double[][] data = generateData(i);
//            System.out.print(i + ": ");
//            System.out.println(Arrays.deepToString(data));
            input[i - start] = new double[]{i % 3.0, i % 5.0};
            output[i - start] = data[1];
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


    private double[][] generateData(int i) {
        double[] data = binaryData(i);
        double[] result = result(i);

        return new double[][]{data, result};
    }

    private double[] binaryData(int i) {
        String binary = Integer.toBinaryString(i);
        char[] chars = binary.toCharArray();

        double[] data = new double[10];

        for (int j = 0; j < chars.length; j++) {
            data[9 - j] = chars[chars.length - j - 1] == '1' ? 1 : 0;
        }
        return data;
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
