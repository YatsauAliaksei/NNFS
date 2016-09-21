package org.ml4bull.matrix;

import org.junit.Test;

import java.util.Arrays;

import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;

public class MatrixOperationsSimpleTest {

    MatrixOperations mo = new MatrixOperationsSimple();

    @Test
    public void testTranspose() throws Exception {
        double[][] m = {
                {1, 2, 3},
                {4, 5, 6},
        };

        mo.printMatrix(m);
        double[][] transpose = mo.transpose(m);

        System.out.println();
        mo.printMatrix(transpose);

        double[][] t = {
                {1, 4},
                {2, 5},
                {3, 6},
        };
        assertThat(Arrays.deepEquals(transpose, t)).isTrue();
    }

    @Test
    public void testMultiply() {
        double[][] m = {
                {1, 2, 3},
                {4, 5, 6},
        };

        double[][] t = {
                {1, 4},
                {2, 5},
                {3, 6},
        };

        double[][] multiply = mo.multiply(m, t);
        mo.printMatrix(multiply);
        assertThat(Arrays.deepEquals(multiply, new double[][]{{17, 32}, {32, 77}}));
    }
}