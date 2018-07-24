package org.ml4bull.ml;

import com.google.common.base.Preconditions;
import lombok.Getter;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.nn.data.Data;
import org.ml4bull.util.Factory;
import org.ml4bull.util.MLUtils;
import org.ml4bull.util.MathUtils;

import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;

@Getter
public class KNearestNeighbors {
    private List<Data> map;
    private int k;
    private final double errorRate; // initial error rate. Any classified value will be added to map. Error rate won't be updated in that case.

    public KNearestNeighbors(List<Data> map, int k) {
        Preconditions.checkArgument(k % 2 != 0, "Better 'k' to be odd number");
        Preconditions.checkArgument(map != null);

        this.map = map;
        this.k = k;

        int errorCounter = 0;
        for (Data data : map) {
            double[] predicted = classify(data);
            if (MLUtils.transformClassToInt(predicted) != MLUtils.transformClassToInt(data.getOutput()))
                errorCounter++;
        }
        errorRate = errorCounter / map.size();
    }

    /**
     * In case equals voting will return all class markers.
     * Expects normalized data [-1, 1],[0, 1].
     * F.i. [0, 1, 0, 0, 1] in case 2nd and last class had equals number of votes.
     * {@param input} will be added to data map and will be processed in next classification round.
     */
    public double[] classify(Data input) {
        LowestQueue lq = new LowestQueue();

        map.stream().parallel().forEach(i -> {
            if (i != input) {
                double ed = MathUtils.euclidianDistanceLazy(i.getInput(), input.getInput());
                lq.insert(ed, i.getOutput());
            }
        });

        if (!map.contains(input))
            map.add(input);

        return voting(lq);
    }

    private double[] voting(LowestQueue lq) {
        MatrixOperations mo = Factory.getMatrixOperations();

        double[] counter = lq.priorityQueue.stream()
                .map(i -> i.classValue)
                .reduce(mo::sum).orElseThrow();

        return transformToClass(counter);
    }

    private double[] transformToClass(double[] counter) {
        double max = Arrays.stream(counter).max().orElseThrow();

        for (int i = 0; i < counter.length; i++) {
            if (counter[i] != max) {
                counter[i] = 0;
            } else
                counter[i] = 1;
        }

        return counter;
    }

    private class LowestQueue {
        PriorityQueue<Item> priorityQueue = new PriorityQueue<>(k, (v1, v2) ->
                Double.compare(v2.euclidianDistance, v1.euclidianDistance) // reverse order
        );

        volatile double highestValue = Double.MAX_VALUE;

        private void insert(double value, double[] classValue) {
            if (highestValue < value) return;

            synchronized (this) {
                if (highestValue < value) return;

                Item candidate = new Item(value, classValue);
                if (priorityQueue.size() < 3) {
                    priorityQueue.add(candidate);
                    return;
                }

                Item largest = priorityQueue.peek();
                if (candidate.euclidianDistance >= largest.euclidianDistance)
                    return;

                priorityQueue.remove();
                priorityQueue.add(candidate);
                highestValue = priorityQueue.peek().euclidianDistance;
            }
        }

        private class Item {
            double euclidianDistance;
            double[] classValue;

            private Item(double ed, double[] classValue) {
                this.euclidianDistance = ed;
                this.classValue = classValue;
            }
        }
    }

}
