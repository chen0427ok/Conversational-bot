import os
import json
import sys
import boto3
from dotenv import load_dotenv
from datetime import datetime

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), '../config/.env'))

class SpeechToText:
    def __init__(self):
        # Whisper 模型配置
        self.endpoint_name = os.getenv('SAGEMAKER_ENDPOINT_NAME', 'jumpstart-dft-hf-asr-whisper-large-20250426-025518')
        self.region = os.getenv('AWS_REGION', 'us-west-2')
        
        # 創建 SageMaker 客戶端
        self.runtime = boto3.client(
            "sagemaker-runtime", 
            region_name=self.region,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # 設置轉錄存儲目錄
        self.transcript_dir = os.path.join(os.path.dirname(__file__), '../../data/transcripts')
        
        # 确保目录存在
        os.makedirs(self.transcript_dir, exist_ok=True)

    def save_transcript(self, transcript_text, audio_file_path, confidence=0.9):
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
            'transcript': transcript_text,
            'confidence': confidence
        }
        
        # 保存为JSON文件
        with open(transcript_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"转写文本已保存至: {transcript_filename}")
        return transcript_filename

    def transcribe_file(self, audio_file_path):
        """将音频文件转换为文字"""
        print("开始转换语音为文字...")

        try:
            # 读取音频文件内容（不需要Base64编码）
            with open(audio_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            # 调用SageMaker端点进行语音识别
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="audio/wav",  # 直接使用wav格式
                Body=audio_bytes
            )
            
            # 解析响应
            response_body = response["Body"].read().decode("utf-8")
            result = json.loads(response_body)
            
            # 从结果中提取文本
            transcript_text = ""
            confidence = 0.9  # 默认置信度
            
            if "text" in result and isinstance(result["text"], list) and len(result["text"]) > 0:
                transcript_text = result["text"][0]
                confidence = result.get("confidence", 0.9) if "confidence" in result else 0.9
                print(f"識別結果: {transcript_text}")
            
            print("语音转换完成!")
            
            # 保存转写结果
            if transcript_text:
                self.save_transcript(transcript_text, audio_file_path, confidence)
            
            return transcript_text

        except Exception as e:
            print(f"转换过程中出现错误: {str(e)}")
            return None 

def main():
    # 測試音頻文件路徑
    test_audio_path = "/Users/brian/hackathon/data/audio/recording_20250426_010008.wav"
    
    print(f"開始測試語音識別，使用音頻文件：{test_audio_path}")
    
    # 檢查文件是否存在
    if not os.path.exists(test_audio_path):
        print(f"錯誤：音頻文件不存在於路徑 {test_audio_path}")
        return
    
    # 創建語音識別實例
    speech_to_text = SpeechToText()
    
    # 執行語音識別
    transcript_text = speech_to_text.transcribe_file(test_audio_path)
    
    # 打印識別結果
    if transcript_text:
        print("\n識別結果摘要:")
        print(f"文本: {transcript_text}")
    else:
        print("未能獲取有效的識別結果")

if __name__ == "__main__":
    main() 