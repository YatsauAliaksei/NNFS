package org.ml4bull.matrix;

public interface DoubleIterator {

    default void iterate(Object[][] t) {
        for (int l = 0; l < t.length; l++) {
            for (int e = 0; e < t[l].length; e++) {
                doIt(l, e);
            }
        }
    }

    default void iterate(double[][] t) {
        for (int l = 0; l < t.length; l++) {
            for (int e = 0; e < t[l].length; e++) {
                doIt(l, e);
            }
        }
    }

    void doIt(int l, int e);
}
