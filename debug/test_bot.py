HISTORY_NAME = 'Training_mipod_20_png'
import telebot

bot = telebot.TeleBot('1292996210:AAFVuU6mo6-rS2Tv6Xuy3ZucfdJzTaBN9PY')
CHAT_ID = 337882617

hist_csv_file = 'E:\Mary\history\Training_mipod_40_png_png_CONTINUATION_31_07_2021_17_07_29\history.csv'
auc_path = 'E:\Mary\history\Training_mipod_40_png_png_CONTINUATION_31_07_2021_17_07_29/auc.png'
loss_path = 'E:\Mary\history\Training_mipod_40_png_png_CONTINUATION_31_07_2021_17_07_29/loss.png'
with open(hist_csv_file) as hist:
    bot.send_document(chat_id=CHAT_ID, data=hist)
with open(auc_path, 'rb') as auc:
    bot.send_photo(chat_id=CHAT_ID, photo=auc)
with open(loss_path, 'rb') as loss:
    bot.send_photo(chat_id=CHAT_ID, photo=loss)