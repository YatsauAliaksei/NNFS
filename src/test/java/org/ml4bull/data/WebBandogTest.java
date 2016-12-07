package org.ml4bull.data;

import org.junit.Test;

import java.io.IOException;
import java.util.Set;

public class WebBandogTest {

    @Test
    public void startDig() throws IOException {
        //        String pageURL = "https://en.wikipedia.org/wiki/Portal:Contents/History_and_events";
//        String pageURL = "https://en.wikipedia.org/wiki/Portal:Contents";
//        String pageURL = "http://www.bbc.com/news";

        String pageURL = "https://www.oxforddictionaries.com";

        WebBandog wb = new WebBandog();
        Set<String> words = wb.findWords(pageURL, 10000, 10);
        System.out.println(words);
    }


}
