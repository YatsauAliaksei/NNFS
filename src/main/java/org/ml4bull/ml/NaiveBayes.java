package org.ml4bull.ml;

import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.nn.data.Data;
import org.ml4bull.util.Factory;
import org.ml4bull.util.MLUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.IntStream;

public class NaiveBayes {
    private Map<Integer, Double> classProb; // The probability of class in data set.
    private Map<Integer, double[]> probabilities; // Key: class label, Value: word probability.

    // Assumes that Data output array has length equals to number of classes.
    public void train(List<Data> dataSet) {

        double[] classesCounter = summarizeDoubles(dataSet, Data::getOutput, $ -> true);

        probabilities = new HashMap<>(classesCounter.length, 1f);
        classProb = new HashMap<>(classesCounter.length, 1f);

        IntStream.rangeClosed(1, classesCounter.length).parallel().forEach(i -> {
                    classProb.put(i, classesCounter[i - 1] / dataSet.size());
                    double[] wp = summarizeDoubles(dataSet, Data::getInput,
                            classLabel -> MLUtils.transformClassToInt(classLabel.getOutput()) == i);
                    double totalWordAmount = Arrays.stream(wp).sum();
                    wp = Arrays.stream(wp).map(w -> {
                                double p = w / totalWordAmount;
                                return Math.log(p == 0 ? 1 : p);
                            }).toArray();
                    probabilities.put(i, wp);
                });
    }

    public double[] classify(Data data) {
        MatrixOperations mo = Factory.getMatrixOperations();
        int classLabelsSize = classProb.size();

        Map<Integer, Double> probMap = new HashMap<>(classLabelsSize, 1f);
        IntStream.rangeClosed(1, classLabelsSize).forEach(i -> {
            double[] cwp = probabilities.get(i);
            double v = mo.multiply(data.getInput(), cwp) + Math.log(classProb.get(i));
            probMap.put(i, v);
        });

        int classifiedLabel = 0;
        double maxVote = -1;
        for (Map.Entry<Integer, Double> entry : probMap.entrySet()) {
            if (entry.getValue() > maxVote) {
                maxVote = entry.getValue();
                classifiedLabel = entry.getKey();
            }
        }
        return MLUtils.transformIntToClass(classifiedLabel, classLabelsSize);
    }

    private double[] summarizeDoubles(List<Data> dataSet, Function<Data, double[]> function, Predicate<Data> filter) {
        return dataSet.stream()
                .filter(filter)
                .map(function)
                .parallel().reduce(Factory.getMatrixOperations()::sum).orElseThrow(RuntimeException::new);
    }
}
