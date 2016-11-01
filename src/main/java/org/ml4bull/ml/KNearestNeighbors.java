package org.ml4bull.ml;

import com.google.common.base.Preconditions;
import org.ml4bull.matrix.MatrixOperations;
import org.ml4bull.nn.data.Data;
import org.ml4bull.util.Factory;
import org.ml4bull.util.MathUtils;

import java.util.List;

// Untested.
public class KNearestNeighbors {
    private List<Data> map;
    private int k;

    public KNearestNeighbors(List<Data> map, int k) {
        Preconditions.checkArgument(k % 2 != 0, "Better 'k' to be odd number");
        Preconditions.checkArgument(map != null);

        this.map = map;
        this.k = k;
    }

    public double[] classify(Data input) {
        LowestQueue lq = new LowestQueue();

        map.stream().peek(i -> {
            double ed = MathUtils.euclidianDistance(i.getInput(), input.getInput());
            lq.insert(ed, i.getOutput());
        }).parallel();

        return getPredictedClass(lq);
    }

    private double[] getPredictedClass(LowestQueue lq) {
        double[] counter = new double[lq.queue[0].classValue.length];

        MatrixOperations mo = Factory.getMatrixOperations();
        for (int i = 0; i < lq.queue.length; i++) {
            counter = mo.sum(counter, lq.queue[i].classValue);
        }

        return transformToClass(counter);
    }

    private double[] transformToClass(double[] counter) {
        double max = -1;
        for (double i : counter) {
            if (i > max) {
                max = i;
            }
        }

        for (int i = 0; i < counter.length; i++) {
            if (counter[i] != max) {
                counter[i] = 0;
            } else
                counter[i] = 1;
        }

        return counter;
    }

    public int getK() {
        return k;
    }

    private class LowestQueue {
        Item[] queue;
        volatile double highestValue = -1;

        private LowestQueue() {
            queue = new Item[k];
            for (int i = 0; i < queue.length; i++) {
                queue[i].ed = Integer.MAX_VALUE;
            }
        }

        private void insert(double value, double[] classValue) {
            if (highestValue < value) return;

            synchronized (this) {
                if (highestValue < value) return;

                Item item = new Item(value, classValue);

                for (int i = 0; i < queue.length; i++) {
                    if (queue[i].ed > item.ed) {
                        Item tmp = queue[i];
                        queue[i] = item;
                        item = tmp;
                    }
                }

                for (int i = queue.length; i >= 0; i--) {
                    if (queue[i].ed != -1) {
                        highestValue = queue[i].ed;
                    }
                }
            }
        }

        private class Item {
            double ed;
            double[] classValue;

            private Item(double ed, double[] classValue) {
                this.ed = ed;
                this.classValue = classValue;
            }
        }
    }

}
