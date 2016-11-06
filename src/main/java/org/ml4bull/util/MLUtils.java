package org.ml4bull.util;

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
}
