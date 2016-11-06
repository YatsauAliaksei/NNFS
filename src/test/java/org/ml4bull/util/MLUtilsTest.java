package org.ml4bull.util;

import org.junit.Test;
import org.ml4bull.matrix.MatrixOperations;

import static org.assertj.core.api.Assertions.assertThat;


public class MLUtilsTest {

    @Test
    public void transformClassToInt() throws Exception {
        double[] v = {0, 0, 0, 0, 1, 0, 0};
        int i = MLUtils.transformClassToInt(v);
        assertThat(i).isEqualTo(3);
    }

    @Test
    public void transformIntToClass() throws Exception {
        double[] v = {0, 0, 0, 0, 1, 0, 0};
        double[] c = MLUtils.transformIntToClass(3, 7);
        MatrixOperations mo = Factory.getMatrixOperations();
        assertThat(mo.same(c, v)).isTrue();
    }
}