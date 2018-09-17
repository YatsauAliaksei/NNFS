package org.ml4bull.util;

import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;


public class MemoryTest {


    @Test
    public void add() {
        Memory<Integer> memory = new Memory<>(3);
        memory.add(1);
        memory.add(2);
        memory.add(3);

        assertThat(memory.get(0).intValue()).isEqualTo(3);
        assertThat(memory.get(1).intValue()).isEqualTo(2);
        assertThat(memory.get(2).intValue()).isEqualTo(1);

        assertThat(memory.getLast().intValue()).isEqualTo(3);
        assertThat(memory.get(2).intValue()).isEqualTo(1);
        assertThat(memory.get(1).intValue()).isEqualTo(2);
        assertThat(memory.get(0).intValue()).isEqualTo(3);
        assertThat(memory.size()).isEqualTo(3);

        memory.add(4);
        assertThat(memory.getLast().intValue()).isEqualTo(4);
        assertThat(memory.get(2).intValue()).isEqualTo(2);
        assertThat(memory.get(1).intValue()).isEqualTo(3);
        assertThat(memory.get(0).intValue()).isEqualTo(4);
        assertThat(memory.size()).isEqualTo(3);
    }

    @Test
    public void removeLast() {
        Memory<Integer> memory = new Memory<>(3);
        memory.add(1);
        memory.add(2);
        memory.add(3);

        assertThat(memory.removeLast().intValue()).isEqualTo(3);
        assertThat(memory.removeLast().intValue()).isEqualTo(2);
        assertThat(memory.removeLast().intValue()).isEqualTo(1);

        assertThat(memory.getLast()).isEqualTo(null);
        assertThat(memory.size()).isEqualTo(0);
    }
}