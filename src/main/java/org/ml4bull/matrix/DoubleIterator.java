package org.ml4bull.matrix;

/**
 * Created by AYatsev.
 */
public interface DoubleIterator {

    default void iterate(double[][] t) {
        for (int l = 0; l < t.length; l++) {
            for (int e = 0; e < t[l].length; e++) {
                doIt(l, e);
            }
        }
    }

    void doIt(int l, int e);
}
