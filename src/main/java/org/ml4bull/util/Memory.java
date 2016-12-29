package org.ml4bull.util;

import com.google.common.base.Preconditions;

public class Memory<T> {
    private Node latest;
    private int maxSize;
    private int size;

    public Memory() {
        this.maxSize = Integer.MAX_VALUE;
    }

    public Memory(int maxSize) {
        this.maxSize = maxSize;
    }

    public void add(T t) {
        Node node = new Node();
        node.value = t;
        node.next = latest;
        latest = node;
        size++;
        truncateSizeIfNeeded();
    }

    private void truncateSizeIfNeeded() {
        if (size > maxSize) {
            Node node = latest;
            for (int i = 1; i <= maxSize; i++) {
                if (i == maxSize) {
                    node.next = null;
                }
                node = node.next;
            }
            size = maxSize;
        }
    }

    public T removeLast() {
        Preconditions.checkState(latest != null, "No values");

        Node tmp = latest;
        latest = latest.next;
        size--;
        return tmp.value;
    }

    public T get(int i) {
        Node node = latest;
        for (int j = 0; j < i; j++) {
            node = latest.next;
        }
        return node.value;
    }

    public T getLast() {
        return latest.value;
    }

    private class Node {
        Node next;
        T value;
    }

    public int size() {
        return size;
    }
}
