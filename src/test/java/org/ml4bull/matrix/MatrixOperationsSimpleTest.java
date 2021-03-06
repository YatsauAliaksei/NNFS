package org.ml4bull.matrix;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.Test;

import java.util.Arrays;

import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;

@Slf4j
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
        System.out.println();

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
        System.out.println();
        mo.printMatrix(multiply);
        System.out.println();
        assertThat(Arrays.deepEquals(multiply, new double[][]{{17, 32}, {32, 77}}));
    }

    @Test
    public void testMultiplyVector() {
        double[] v = {1, 2, 3};
        mo.printMatrix(v);
        val transpose = mo.transpose(v);
        System.out.println();
        mo.printMatrix(transpose);
        System.out.println();
    }

    @Test
    public void testConcatenate() {
        double[] m1 = {1, 2, 3};
        double[] m2 = {4, 5, 6};
        double[] concatenate = mo.concatenate(m1, m2);

        assertThat(concatenate.length).isEqualTo(6);
        assertThat(concatenate[2]).isEqualTo(3);
        assertThat(concatenate[5]).isEqualTo(6);
        mo.printMatrix(concatenate);
    }

    @Test
    public void scalarMultiply() {
        double[] m1 = {1, 2, 3};
        double[] m2 = {4, 5, 6};

        double[] result = mo.scalarMultiply(m1, m2);
        mo.printMatrix(result);
        assertThat(result).isEqualTo(new double[] {4, 10, 18});
    }
}