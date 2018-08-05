package org.ml4bull.ml.tree;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.ml4bull.annotation.Untested;
import org.ml4bull.nn.data.Data;
import org.ml4bull.util.MLUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;

@Untested
// regression
// Millions of improvements could be done. Very basic implementation.
public class CARTDecisionTree {

    private Node root;
    private final int dataSetSizeThreshold;
    private final int acceptableError;
    @Getter
    private final double errorRate;
    private final Node stub = leaf(0);

    public CARTDecisionTree(List<Data> trainingData, List<Data> testSet, int dataSetSizeThreshold, int acceptableError) {
        this.dataSetSizeThreshold = dataSetSizeThreshold;
        this.acceptableError = acceptableError;

        createTree(new ArrayList<>(trainingData));
        errorRate = MLUtils.errorRate(testSet, this::classify);
    }

    public double[] classify(Data data) {
        Node curNode = root;
        while (curNode.fIndex != -1) {
            if (data.getInput()[curNode.fIndex] > curNode.fValue)
                curNode = curNode.right;
            else
                curNode = curNode.left;
        }

        return new double[]{curNode.fValue};
    }

    private void createTree(List<Data> dataSet) {
        root = createNode(dataSet, null);
    }

    private Node createNode(List<Data> dataSet, Node parent) {
        Node node = chooseBestSplit(dataSet);
        if (node.fIndex == -1)
            return node;

        if (parent == null)
            parent = node;

        Map<Boolean, List<Data>> splittedDataSet = binarySplit(dataSet, d -> d.getInput()[node.fIndex] > node.fValue);
        parent.right = createNode(splittedDataSet.get(true), parent);
        parent.left = createNode(splittedDataSet.get(false), parent);
        return parent;
    }

    private Node chooseBestSplit(List<Data> dataSet) {
        if (dataSet.isEmpty())
            return stub;

        if (dataSet.size() < dataSetSizeThreshold) {
            double average = dataSet.stream()
                    .mapToDouble(d -> d.getOutput()[0])
                    .average().getAsDouble();

            return leaf(average);
        }

        long numberOfLeftDiffValues = dataSet.stream()
                .map(d -> d.getOutput()[0])
                .distinct().count();

        if (numberOfLeftDiffValues == 1)
            return leaf(dataSet.get(0).getOutput()[0]);


        double mse = meanSquaredError(dataSet); // calculate current MSE
        double bestError = Double.MAX_VALUE;
        int fLength = dataSet.get(0).getInput().length;
        Node splitNode = new Node();
        for (int i = 0; i < fLength; i++) { // iterate throw ALL features in Data set
            final int k = i;
            double[] values = dataSet.stream()
                    .mapToDouble(d -> d.getInput()[k])
                    .distinct()
                    .toArray();

            for (double value : values) { // for EACH value split Data set with simple predicate
                Map<Boolean, List<Data>> splitSuggestion = binarySplit(dataSet, d -> d.getInput()[k] > value);
                List<Data> greater = splitSuggestion.get(true);
                List<Data> less = splitSuggestion.get(false);
                if (greater.size() < dataSetSizeThreshold || less.size() < dataSetSizeThreshold)
                    continue;

                double newError = meanSquaredError(greater) + meanSquaredError(less); // calculate new MSE for 2 sub-data sets.
                if (newError > bestError)
                    continue;

                splitNode.fIndex = i;
                splitNode.fValue = value;
                bestError = newError;
            }
        }

        if (mse - bestError < acceptableError)
            splitNode.fIndex = -1;

        return splitNode;
    }

    private Map<Boolean, List<Data>> binarySplit(List<Data> data, Predicate<Data> predicate) {
        return data.stream()
                .collect(Collectors.partitioningBy(predicate));
    }

    private double meanSquaredError(List<Data> data) {
        double mean = mean(data, d -> d.getOutput()[0]);
        double sum = data.stream()
                .mapToDouble(d ->
                        Math.pow(d.getOutput()[0] - mean, 2)
                ).sum();

        double populationVariance = sum / data.size();
        return populationVariance / data.size();
    }

    private double mean(List<Data> data, ToDoubleFunction<Data> function) {
        return data.stream()
                .mapToDouble(function)
                .average().getAsDouble();
    }

    private Node leaf(double fValue) {
        return new Node(null, null, -1, fValue);
    }

    @AllArgsConstructor
    @NoArgsConstructor
    private class Node {
        Node left, right;
        int fIndex = -1; // by default means final/leaf, no further split needed.
        double fValue;
    }
}
