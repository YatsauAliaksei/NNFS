package org.ml4bull.nn.data;

public class DataSet {
    double[][] input;
    double[][] output;

    public DataSet(double[][] input, double[][] output) {
        this.input = input;
        this.output = output;
    }

    public double[][] getInput() {
        return input;
    }

    public void setInput(double[][] input) {
        this.input = input;
    }

    public double[][] getOutput() {
        return output;
    }

    public void setOutput(double[][] output) {
        this.output = output;
    }
}
