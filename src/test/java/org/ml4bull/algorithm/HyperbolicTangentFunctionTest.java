package org.ml4bull.algorithm;

import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;


public class HyperbolicTangentFunctionTest {

    private HyperbolicTangentFunction htF = new HyperbolicTangentFunction();

    @Test
    public void activate() throws Exception {
        double a = htF.activate(-2e-1);
        assertThat(a).isBetween(-.198, -.197);

        a = htF.activate(4e-1);
        assertThat(a).isBetween(.379, .38);
    }

    @Test
    public void derivative() throws Exception {
        double[] v = {.2, .3, .5, .9};
        double[] derivative = htF.derivative(v);

        assertThat(derivative[0]).isEqualTo(.96);
        assertThat(derivative[1]).isEqualTo(.91);
        assertThat(derivative[2]).isEqualTo(.75);
        assertThat(derivative[3]).isBetween(.189, .19);
    }

}