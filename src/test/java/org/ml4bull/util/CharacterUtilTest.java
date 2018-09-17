package org.ml4bull.util;

import org.junit.Test;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static org.assertj.core.api.Assertions.assertThat;

public class CharacterUtilTest {

    @Test
    public void russianChars() {
        CharUtils.Language language = CharUtils.Language.RUSSIAN;
        System.out.println((int) 'а' + "-" + (int) 'я');
        AtomicInteger k = new AtomicInteger();
        IntStream.rangeClosed(1072, 1103)
                .peek(i -> k.incrementAndGet())
                .forEach(i -> System.out.println((char) i + " - " + (i - language.getLeading())));

        System.out.println("Total:" + k);

        double[] doubles = CharUtils.charToVector('а', language); // russian A
        char a = CharUtils.vectorToChar(doubles, language);

        assertThat(a).isEqualTo('а');

        doubles = CharUtils.charToVector(' ', language);
        char space = CharUtils.vectorToChar(doubles, language);

        assertThat(space).isEqualTo(' ');
    }

    @Test
    public void charsTest() {
        CharUtils.Language language = CharUtils.Language.US;
        System.out.println("Space: " + (int) ' ');

        for (int i = 'a'; i <= 'z'; i++)
            System.out.println(((char) i) + " - " + (i - 96));

        System.out.println("AxA12 -$ /{1fas!fR}".replaceAll("[^а-я ]", ""));

        double[] doubles = CharUtils.charToVector('a', language);
        char a = CharUtils.vectorToChar(doubles, language);

        assertThat(a).isEqualTo('a');

        doubles = CharUtils.charToVector(' ', language);
        char space = CharUtils.vectorToChar(doubles, language);

        assertThat(space).isEqualTo(' ');
    }
}
