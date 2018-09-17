package org.ml4bull.util;

import com.google.common.base.Preconditions;
import lombok.val;
import org.ml4bull.nn.data.Data;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;
import java.util.stream.IntStream;

public class MLUtils {

    /**
     * Transforms array representation of class to int
     * @param arr - array representation of class. Expected only one '1' value.
     * @return int representation of class.
     */
    public static int transformClassToInt(double[] arr) {
        int i = 0;
        for (; i < arr.length; i++) {
            if (arr[i] == 1) break;
        }
        return arr.length - i;
    }

    public static double[] transformIntToClass(int value, int arrLength) {
        double[] c = new double[arrLength];
        c[arrLength - value] = 1;
        return c;
    }

    public static double[] getRandomWeights(int size) {
        ThreadLocalRandom tlr = ThreadLocalRandom.current();
        return tlr.doubles(size, -0.3, 0.3).map(v -> {
            if (v == 0) {
                v = ThreadLocalRandom.current().nextDouble(1e-10, .3); // yes, let it be positive
            }
            return v;
        }).toArray();
    }

    public static double[] getRandomDropout(int size) {
        ThreadLocalRandom tlr = ThreadLocalRandom.current();
        return tlr.doubles(size, 0, 1).toArray();
    }

    public static List<Data> normalize(List<Data> data) {
        Preconditions.checkArgument(data != null && !data.isEmpty(), "Not null and not empty");
        val normalizedData = List.copyOf(data);

        IntStream.range(0, normalizedData.get(0).getInput().length)
                .parallel()
                .forEach(k -> normalize(k, normalizedData));

        return normalizedData;
    }

    public static double errorRate(List<Data> dataSet, Function<Data, double[]> classifyFunction) {
        int errorCounter = 0;
        for (Data data : dataSet) {
            double[] predicted = classifyFunction.apply(data);
            if (MLUtils.transformClassToInt(predicted) != MLUtils.transformClassToInt(data.getOutput()))
                errorCounter++;
        }
        return errorCounter / dataSet.size();
    }

    private static void normalize(int featureNumber, List<Data> data) {
        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;

        for (Data item : data) {
            double featureValue = item.getInput()[featureNumber];
            if (min > featureValue)
                min = featureValue;

            if (max < featureValue)
                max = featureValue;
        }

        double valuesRangeDelta = max - min;
        for (Data item : data) {
            double[] vector = item.getInput();
            double featureValue = vector[featureNumber];
            vector[featureNumber] = (featureValue - min) / valuesRangeDelta;
        }
    }

    // Clip' Shrink' section
    public static void shrink(int denominator, double[][]... toShrink) {
        List<CompletableFuture<Void>> futures = new ArrayList<>();
        for (double[][] doubles : toShrink) {
            CompletableFuture<Void> cf = CompletableFuture.runAsync(() ->
                    shrink(doubles, denominator)
            );
            futures.add(cf);
        }
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
    }

    public static void shrink(int denominator, double[]... toShrink) {
        for (double[] doubles : toShrink) {
            shrink(doubles, denominator);
        }
    }

    public static void shrink(double[][] layerWeights, int denominator) {
        for (double[] neuronWeights : layerWeights) {
            for (int i = 0; i < neuronWeights.length; i++) {
                neuronWeights[i] /= denominator;
            }
        }
    }

    public static void shrink(double[] biases, int denominator) {
        for (int i = 0; i < biases.length; i++) {
            biases[i] /= denominator;
        }
    }

    public static void clip(double[] data, double min, double max) {
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] > max ? max : data[i] < min ? min : data[i];
        }
    }

    public static void clip(double[][] toClip, double min, double max) {
        for (double[] dW : toClip) {
            MLUtils.clip(dW, min, max);
        }
    }

    public static void clip(double[][]... toClip) {
        for (double[][] doubles : toClip) {
            MLUtils.clip(doubles, -5, 5);
        }
    }

    public static void clip(double[]... toClip) {
        for (double[] doubles : toClip) {
            MLUtils.clip(doubles, -5, 5);
        }
    }
}
