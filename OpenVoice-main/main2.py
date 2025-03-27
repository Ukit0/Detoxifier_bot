import speech_recognition as sr 
import os
from datetime import datetime

import torch 
import se_extractor 
from api import ToneColorConverter 
import pyttsx3 

from transformers import T5ForConditionalGeneration, AutoTokenizer

from telebot import TeleBot
from telebot import types

import sqlite3





def sql_request(sql):
    try:
        # Обращаемся к БД, если её нет, то она автоматически создается
        connection = sqlite3.connect("database/test.db")

        # Чтобы выполнить запрос к таблице в базе данных, нам нужно использовать объект курсора
        cursor = connection.cursor()

        # Теперь можем выполнить запрос к БД
        cursor.execute(sql)
        result = cursor.fetchall()
        # Фиксируем изменения в БД
        connection.commit()
        
        # Не забываем закрыть соединение
        connection.close()
        return result
    except Exception as e:
        print(e)




def voice_translation(file_name):
    
    print('Translation to text')
 
    r = sr.Recognizer()

    with sr.AudioFile(f'voice/{file_name}.flac') as voice_file:
        r.adjust_for_ambient_noise(voice_file) 
        audio = r.record(voice_file)
        words = r.recognize_google(audio, language = 'ru')

    return words #




base_model_name = 'sberbank-ai/ruT5-base'
model_name = 'SkolkovoInstitute/ruT5-base-detox'

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def paraphrase(text, model, tokenizer, n=None, max_length="auto", beams=3):
    print('Taraphrase start!')

    texts = [text] if isinstance(text, str) else text
    inputs = tokenizer(texts, return_tensors="pt", padding=True)["input_ids"].to(
        model.device
    )
    if max_length == "auto":
        max_length = inputs.shape[1] + 10

    result = model.generate(
        inputs,
        num_return_sequences=n or 1,
        do_sample=True,
        temperature=1.0,
        repetition_penalty=10.0,
        max_length=max_length,
        min_length=int(0.5 * max_length),
        num_beams=beams
    )
    texts = [tokenizer.decode(r, skip_special_tokens=True) for r in result]

    print('Taraphrase end!')

    if not n and isinstance(text, str):
        return texts[0]
    return texts




def ai_voice(text, file_name):

    print('Start')
    ckpt_converter = '../checkpoints_v2\converter'
    device="cuda:0" if torch.cuda.is_available() else "cpu" 
    output_dir = 'voice' 
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device) 
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth') 
    os.makedirs(output_dir, exist_ok=True) 
    
    print(1) 
    base_speaker = f'{output_dir}/{file_name}.wav' #Файл для тона 
    reference_speaker = f'{output_dir}/{file_name}.mp3' #Файл для обучения 
    
    print(2)
    source_se, audio_name = se_extractor.get_se(base_speaker, tone_color_converter, vad=True)
    print(3) 
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True) 
    
    print(4) 
    src_path = f'{output_dir}/tmp_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.wav' 

    print(5)
    tts = pyttsx3.init()

    voices = tts.getProperty('voices')

    tts.setProperty('rate', 120) 
    tts.setProperty('volume', 1)

    for voice in voices: 
        if voice.name == 'Aleksandr': 
            print('Aleksandr')
            tts.setProperty('voice', voice.id) 
            tts.save_to_file(text, src_path)
            tts.runAndWait() 

    print(6)
    save_path = f'{output_dir}/output_crosslingual_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.wav' 

    print(7)
    # Run the tone color converter 
    encode_message = "@MyShell" 
    tone_color_converter.convert( 
        audio_src_path=src_path,  
        src_se=source_se,  
        tgt_se=target_se,  
        output_path=save_path, 
        message=encode_message)
    
    print("Good")

    return save_path, src_path




def main():
    print('Super start')

    TOKEN_BOT = ''
    bot = TeleBot(TOKEN_BOT)

    sql_request("CREATE TABLE table_name "\
                "(`id` INTEGER PRIMARY KEY AUTOINCREMENT,"\
                " `users_id` text,"\
                " `type_message` text);")

    comand = ('Исправь аудио запись','Исправь текст')

    @bot.message_handler(commands=['start']) 
    def start_processing(message):
        
        if not(sql_request(f"SELECT users_id FROM table_name WHERE users_id = '{message.chat.id}';")):

            sql_request("INSERT INTO table_name (`users_id`, `type_message`) "\
                        f"VALUES ('{message.chat.id}','h');")

            keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=False)
            keyboard.add(types.KeyboardButton('Исправь текст'), types.KeyboardButton('Исправь аудио запись'))

            bot.send_message(message.chat.id, f'Привет {message.from_user.first_name}!', reply_markup=keyboard)


    @bot.message_handler(content_types=['voice']) 
    def voice_processing(message):
        print('Voice')

        if sql_request(f"SELECT type_message FROM table_name WHERE users_id = '{message.chat.id}';")[0][0] == 'voice':

            print('Pipao1')
            object_ = f'{message.chat.id}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
            filename = f'sound{object_}'
            format_ = ('ogg','flac','wav','mp3')

            file_info = bot.get_file(message.voice.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            with open(f'voice/{filename}.ogg', 'wb') as new_file: 
                new_file.write(downloaded_file)
            print('Pipao2')

            for formatt in format_[1:]:
                #ffmpeg.input(f'voice/{filename}.ogg').output(f'voice/{filename}.{formatt}').run()
                os.system(f'ffmpeg.exe -i voice/{filename}.ogg voice/{filename}.{formatt}')
            print('VOICE2')
            text = voice_translation(filename) 

            clean_text = paraphrase(text, model, tokenizer) 

            print('Not')
            voice_ai_file_path, tmp_path = ai_voice(clean_text,filename)
            print('Yes')

            with open(voice_ai_file_path, 'rb') as audio_ai:
                bot.send_voice(message.chat.id, audio_ai)

            os.remove(tmp_path)
            for formatt in format_:
                os.remove(f'voice/{filename}.{formatt}')
            os.remove(voice_ai_file_path)
    

    @bot.message_handler(content_types=['text']) 
    def text_processing(message):

        if bool(sql_request(f"SELECT users_id FROM table_name WHERE users_id = '{message.chat.id}';")):
            
            if message.text == comand[1]:
                bot.send_message(message.chat.id, 'Готов работать с твоим текстом!')

                sql_request(f"UPDATE table_name SET type_message = 'text' WHERE users_id = '{message.chat.id}'")

                print('Okey')

            elif message.text == comand[0]:
                bot.send_message(message.chat.id, 'Готов слушать тебя и исправлять!')

                sql_request(f"UPDATE table_name SET type_message = 'voice' WHERE users_id = '{message.chat.id}'")
                
                print('Lets do it')
            
            elif sql_request(f"SELECT type_message FROM table_name WHERE users_id = '{message.chat.id}';")[0][0] == 'text' and message.text != comand[0] and message.text != comand[1]:
                print('Pipao')
                clean_text = paraphrase(message.text, model, tokenizer)
                bot.send_message(message.chat.id, clean_text)


    bot.polling(none_stop=True, interval=0)



main()

#conda activate openvoice 
#cd Desktop\PyAi\OpenVoice-main
#python main2.py
