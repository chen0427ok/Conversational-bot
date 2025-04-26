import os
import json
import requests
from dotenv import load_dotenv
import base64
from datetime import datetime
import sys
import boto3
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.command_classifier import CommandClassifier

endpoint_name = "whisper-base-145428-Endpoint-20250421-150459"
region = "us-east-1"

runtime = boto3.client("sagemaker-runtime", region_name=region)


# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), '../config/.env'))

class SpeechToText:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_API_KEY')
        self.language_code = "zh-TW"  # 设置为繁体中文
        self.sample_rate = int(os.getenv('SAMPLE_RATE', 16000))
        self.transcript_dir = os.path.join(os.path.dirname(__file__), '../../data/transcripts')
        self.classifier = CommandClassifier()
        
        # 确保目录存在
        os.makedirs(self.transcript_dir, exist_ok=True)

    def save_transcript(self, transcripts, audio_file_path):
        """保存转写结果"""
        # 生成转写文件名（基于音频文件名）
        audio_filename = os.path.basename(audio_file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_filename = os.path.join(
            self.transcript_dir,
            f"transcript_{timestamp}.json"
        )
        
        # 准备保存的数据
        data = {
            'audio_file': audio_filename,
            'timestamp': timestamp,
            'transcripts': transcripts
        }
        
        # 保存为JSON文件
        with open(transcript_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"转写文本已保存至: {transcript_filename}")
        return transcript_filename

    def transcribe_file(self, audio_file_path):
        """将音频文件转换为文字并分类"""
        print("开始转换语音为文字...")

        # 读取音频文件
        with open(audio_file_path, "rb") as audio_file:
            content = base64.b64encode(audio_file.read()).decode('utf-8')

        # 准备API请求
        url = f"https://speech.googleapis.com/v1/speech:recognize?key={self.api_key}"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            'config': {
                'encoding': 'LINEAR16',
                'sampleRateHertz': self.sample_rate,
                'languageCode': self.language_code,
                'enableAutomaticPunctuation': True
            },
            'audio': {
                'content': content
            }
        }

        # 执行转换
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # 检查是否有错误
            
            results = response.json().get('results', [])
            
            # 提取转换结果并分类
            transcripts = []
            for result in results:
                alternatives = result.get('alternatives', [])
                if alternatives:
                    transcript = alternatives[0].get('transcript', '')
                    confidence = alternatives[0].get('confidence', 0)
                    
                    # 对文本进行命令分类
                    command_type = self.classifier.classify_command(transcript)
                    
                    transcripts.append({
                        'text': transcript,
                        'confidence': confidence,
                        'command_type': command_type
                    })
            
            print("语音转换和分类完成!")
            
            # 保存转写结果
            if transcripts:
                self.save_transcript(transcripts, audio_file_path)
            
            return transcripts

        except Exception as e:
            print(f"转换过程中出现错误: {str(e)}")
            return None 