import os
import json
import requests
from datetime import datetime
import pygame
from dotenv import load_dotenv
import base64
from google.cloud import texttospeech
from google.oauth2 import service_account

# 加載環境變量
load_dotenv(os.path.join(os.path.dirname(__file__), '../config/.env'))

class ResponseSpeaker:
    def __init__(self):
        # 加載服務賬號認證
        credentials_path = os.path.join(os.path.dirname(__file__), '../config/google_cloud_credentials.json')
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        
        # 初始化客戶端
        self.client = texttospeech.TextToSpeechClient(credentials=credentials)
        
        # 設置語音參數
        self.voice = texttospeech.VoiceSelectionParams(
            language_code='cmn-Hant-TW',
            name='cmn-TW-Standard-A',
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        # 設置音訊參數
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )
        
        # 初始化 pygame 用於播放音訊
        pygame.mixer.init()
        
        # 創建音訊文件存儲目錄
        self.audio_dir = os.path.join(os.path.dirname(__file__), '../../data/audio_output')
        os.makedirs(self.audio_dir, exist_ok=True)

    def text_to_speech(self, text):
        """將文本轉換為語音並保存為文件"""
        try:
            # 構建合成請求
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # 發送請求
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=self.voice,
                audio_config=self.audio_config
            )
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = os.path.join(self.audio_dir, f'speech_{timestamp}.mp3')
            
            # 保存音訊文件
            with open(audio_file, 'wb') as out:
                out.write(response.audio_content)
                
            return audio_file
            
        except Exception as e:
            print(f"轉換語音時出錯: {str(e)}")
            return None

    def play_audio(self, audio_file):
        """播放音訊文件"""
        if audio_file and os.path.exists(audio_file):
            try:
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
            except Exception as e:
                print(f"播放音訊時出錯: {str(e)}")
        else:
            print("音訊文件不存在或生成失敗")

    def process_history_file(self, file_path):
        """處理歷史記錄文件並播放對應的回應"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 根據文件類型提取要播放的文本
            if 'movement_plan' in data:  # movement_history
                text = "\n".join(data['movement_plan']['說明'])
            elif 'response' in data:  # chat_history 或 query_history
                text = data['response']
            else:
                raise ValueError("不支援的文件格式")
            
            print(f"正在轉換文本為語音：\n{text}\n")
            
            # 轉換並播放
            audio_file = self.text_to_speech(text)
            if audio_file:
                print(f"開始播放語音...")
                self.play_audio(audio_file)
                print(f"語音播放完成！\n")
            
        except Exception as e:
            print(f"處理文件時出錯: {str(e)}")

def main():
    speaker = ResponseSpeaker()
    
    # 測試不同類型的歷史記錄文件
    test_files = [
        '../../data/chat_history/chat_20250410_104002.json',
        '../../data/movement_history/movement_20250410_142131.json',
        '../../data/query_history/query_20250410_104646.json'
    ]
    
    for file_path in test_files:
        abs_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(abs_path):
            print(f"\n處理文件: {file_path}")
            speaker.process_history_file(abs_path)
        else:
            print(f"文件不存在: {file_path}")

if __name__ == "__main__":
    main() 