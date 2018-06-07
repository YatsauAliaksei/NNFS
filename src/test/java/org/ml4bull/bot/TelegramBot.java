package org.ml4bull.bot;

import org.telegram.telegrambots.ApiContextInitializer;
import org.telegram.telegrambots.TelegramBotsApi;
import org.telegram.telegrambots.api.methods.send.SendMessage;
import org.telegram.telegrambots.api.objects.Message;
import org.telegram.telegrambots.api.objects.Update;
import org.telegram.telegrambots.bots.DefaultBotOptions;
import org.telegram.telegrambots.bots.TelegramLongPollingBot;
import org.telegram.telegrambots.exceptions.TelegramApiException;
import org.telegram.telegrambots.exceptions.TelegramApiRequestException;

public class TelegramBot extends TelegramLongPollingBot {

    private TelegramBot() {
    }

    private TelegramBot(DefaultBotOptions options) {
        super(options);
    }

    @Override
    public void onUpdateReceived(Update update) {

    }

    @Override
    public String getBotUsername() {
        return "";
    }

    @Override
    public String getBotToken() {
        return "";
    }

    public Message sendMessageToChat(long chatId, String message) throws TelegramApiException {
        return execute(new SendMessage(chatId, message)); // -286755964L
    }

    @Override
    public void clearWebhook() throws TelegramApiRequestException {}

    private void registerBot() throws TelegramApiRequestException {
        ApiContextInitializer.init();
        TelegramBotsApi telegramBotsApi = new TelegramBotsApi();
        telegramBotsApi.registerBot(this);
        System.out.println("Registered");
    }

    private static TelegramBot instance;

    public static TelegramBot takeMe() {
        if (instance == null) { // no synchronization. Simplest version.
            instance = new TelegramBot();
            try {
                instance.registerBot();
            } catch (TelegramApiRequestException e) {
                throw new RuntimeException(e);
            }
        }
        return instance;
    }
}
