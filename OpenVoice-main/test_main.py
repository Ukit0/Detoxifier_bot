import os
from datetime import datetime

import torch 
import se_extractor 
from api import ToneColorConverter 
import pyttsx3



def ai_voice(text, src_path = ''):
    print('Start')

    ckpt_converter = '../checkpoints_v2\converter'
    device="cuda:0" if torch.cuda.is_available() else "cpu" 
    output_dir = 'voice' 
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device) 
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth') 
    os.makedirs(output_dir, exist_ok=True) 
    
    print(1) 
    base_speaker = f'{output_dir}/Record.wav' #Файл для тона 
    reference_speaker = f'{output_dir}/Piece.mp3' #Файл для обучения 
    
    print(2)
    source_se, audio_name = se_extractor.get_se(base_speaker, tone_color_converter, vad=True)
    print(3) 
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True) 
    

    print(4)
    if len(src_path) == 0:
        src_path = f'{output_dir}/tmp_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.wav' 

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

    print(5)
    save_path = f'{output_dir}/output_crosslingual_{datetime.now().strftime("%M_%S_%f")}.wav' 

    print(6)
    # Run the tone color converter 
    encode_message = "@MyShell" 
    tone_color_converter.convert( 
        audio_src_path=src_path,  
        src_se=source_se,  
        tgt_se=target_se,  
        output_path=save_path, 
        message=encode_message)
    
    print("Good")

    os.remove(src_path)


    #choice = input('select what you want to filter - text / voice:\n')


ai_voice('Программа как же ты меня достала надеюсь что когда ты заработаешь ты поймешь что ты плохой человек когда же все это наконец произойдёт')
#conda activate openvoice 
#cd Desktop\PyAi\OpenVoice-main
#python test_main.py