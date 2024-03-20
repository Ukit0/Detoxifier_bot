import speech_recognition as sr 
import os
from datetime import datetime

import torch 
import se_extractor 
from api import ToneColorConverter 
import pyttsx3 

from transformers import T5ForConditionalGeneration, AutoTokenizer

from telebot import TeleBot




number_of_calls = 0

def voice_translation(file_input):
    global number_of_calls
    
    print('Translation to text')
 
    r = sr.Recognizer()

    with sr.AudioFile(f'voice/{file_input}.flac') as voice_file:
        r.adjust_for_ambient_noise(voice_file) 
        audio = r.record(voice_file)
        words = r.recognize_google(audio, language = 'ru')

    number_of_calls += 1
    print(f'Номер вызова: {number_of_calls}')

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

    ckpt_converter = '../checkpoints/converter'
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

    return save_path




def main():
    print('Super start')

    TOKEN_BOT = ''
    bot = TeleBot(TOKEN_BOT)

    @bot.message_handler(commands=['start']) 
    def voice_processing(message):
        bot.send_message(message.chat.id, 'Hello')


    @bot.message_handler(content_types=['voice']) 
    def voice_processing(message):
        print('Voice')

        object_ = f'{message.chat.id}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
        filename = f'sound{object_}'
        format_ = ['ogg','flac','wav','mp3']

        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open(f'voice/{filename}.ogg', 'wb') as new_file: 
            new_file.write(downloaded_file)
        
        for formatt in format_[1:]:
            os.system(f'ffmpeg.exe -i voice/{filename}.ogg voice/{filename}.{formatt}')
        print('VOICE2')
        text = voice_translation(filename) 

        clean_text = paraphrase(text, model, tokenizer) 

        print('Not')
        voice_ai_file_path = ai_voice(clean_text,filename)
        print('Yes')

        with open(voice_ai_file_path, 'rb') as audio_ai:
            bot.send_voice(message.chat.id, audio_ai)

        for formatt in format_:
            os.remove(f'voice/{filename}.{formatt}')
        #os.remove(voice_ai_file_path)
    

    @bot.message_handler(content_types=['text']) 
    def voice_processing(message):
        text = input()
        clean_text = paraphrase(text, model, tokenizer)
        bot.send_message(message.chat.id, clean_text)

    bot.polling(none_stop=True, interval=0)

    #choice = input('select what you want to filter - text / voice:\n')


main()
#conda activate openvoice 
#cd Desktop\PyAi\OpenVoice-main
#python main2.py