package org.ml4bull.util;

import com.google.common.base.Preconditions;

import java.util.stream.IntStream;

public class MathUtils {

    public static double log2(double n) {
        return Math.log(n) / Math.log(2);
    }

    public static double euclidianDistance(double[] item1, double[] item2) {
        return Math.sqrt(euclidianDistanceLazy(item1,  item2));
    }

    /**
     * Without sqrt operation
     */
    public static double euclidianDistanceLazy(double[] item1, double[] item2) {
        Preconditions.checkArgument(item1.length == item2.length);

        return IntStream.range(0, item1.length).mapToDouble(i ->
                Math.pow(item1[i] - item2[i], 2)
        ).sum();
    }
}
