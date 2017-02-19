package org.ml4bull.ml;


import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.AtomicDoubleArray;
import org.ml4bull.nn.data.Data;
import org.ml4bull.util.MLUtils;
import org.ml4bull.util.MathUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DecisionTree {

    private Node root;
    private int classLength;

    public void createTree(List<Data> dataSet) {
        Preconditions.checkState(root == null, "Please, create new tree object.");
        root = new Node();
        classLength = dataSet.get(0).getOutput().length;
        createNode(dataSet, root);
    }

    public double[] classify(Data data) {

        Node node = root;
        do {
            double v = data.getInput()[node.fIndex];
            node = node.findByValue(v);
        } while (node.nodes != null);

        return MLUtils.transformIntToClass((int) node.fValue, classLength);
    }

    private Node createNode(List<Data> dataSet, Node parent) {
        if (dataSet.size() == 1) {
            return getTerminalNode(dataSet);
        }

        long classCount = dataSet.stream()
                .map(Data::getOutput)
                .mapToInt(MLUtils::transformClassToInt)
                .distinct().count();

        if (classCount == 1) {
            return getTerminalNode(dataSet);
        }

        int fIndex = chooseBestFeatureToSplit(dataSet);
        Set<Double> fl = getFeatureList(dataSet, fIndex);
        parent.fIndex = fIndex;
        parent.nodes = new ArrayList<>();
        fl.forEach(f -> {
            List<Data> subDs = splitDataSet(dataSet, fIndex, f, 0);
            Node child = new Node();
            child.fValue = f;
            parent.nodes.add(createNode(subDs, child));
        });

        return parent;
    }

    private Node getTerminalNode(List<Data> dataSet) {
        Node n = new Node();
        double[] output = dataSet.get(0).getOutput();
        n.fValue = MLUtils.transformClassToInt(output);
        return n;
    }

    private double shannonEntropy(List<Data> dataSet) {
        AtomicInteger[] counter = new AtomicInteger[dataSet.get(0).getOutput().length];

        dataSet.stream().parallel().forEach(data -> {
            double[] ca = data.getOutput();
            int c = MLUtils.transformClassToInt(ca);
            counter[c].incrementAndGet();
        });

        double shannonEnt = .0;
        for (AtomicInteger i : counter) {
            double prob = (double) i.get() / dataSet.size();
            shannonEnt -= prob * MathUtils.log2(prob);
        }

        return shannonEnt;
    }

    private List<Data> splitDataSet(List<Data> dataSet, int fIndex, double fValue, double range) {
        return dataSet.stream()
                .filter(data -> data.getInput()[fIndex] - range / 2 > fValue ||
                        data.getInput()[fIndex] + range / 2 < fValue
                ).collect(Collectors.toList());
    }

    private int chooseBestFeatureToSplit(List<Data> dataSet) {
        final double baseEntropy = shannonEntropy(dataSet);
        int numF = dataSet.get(0).getInput().length;
        DecisionTreeHelper dth = new DecisionTreeHelper();

        IntStream.range(0, numF).parallel().forEach(i -> {

            Set<Double> featureList = getFeatureList(dataSet, i);

            double newEntropy = featureList.stream().mapToDouble(f -> {
                List<Data> subDs = splitDataSet(dataSet, i, f, 0);
                double prob = (double) subDs.size() / dataSet.size();
                return prob * shannonEntropy(subDs);
            }).sum();

            double infoGain = baseEntropy - newEntropy;
            if (infoGain > dth.getBestEntropy()) {
                dth.setBestEntropy(infoGain);
                dth.setBestFeature(i);
            }
        });

        return dth.getBestFeature();
    }

    private Set<Double> getFeatureList(List<Data> dataSet, int fIndex) {
        return dataSet.stream()
                .mapToDouble(data -> data.getInput()[fIndex])
                .boxed().collect(Collectors.toSet());
    }

    private class DecisionTreeHelper {
        AtomicDoubleArray ada = new AtomicDoubleArray(2) {{ // 0 - BestEntropy, 1 - BestFeature index.
            set(1, -1);
        }};

        void setBestEntropy(double bestEntropy) {
            ada.set(0, bestEntropy);
        }

        void setBestFeature(int bestFeature) {
            ada.set(1, bestFeature);
        }

        double getBestEntropy() {
            return ada.get(0);
        }

        int getBestFeature() {
            return (int) ada.get(1);
        }
    }

    private class Node {
        List<Node> nodes;
        int fIndex;
        double fValue;

        Node findByValue(double fValue) {
            return nodes.stream().filter(n -> n.fValue == fValue).findFirst().get();
        }
    }
}
