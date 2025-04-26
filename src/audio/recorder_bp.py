import os
import wave
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from datetime import datetime

# 加载环境变量
load_dotenv(os.path.join(os.path.dirname(__file__), '../config/.env'))

class AudioRecorder:
    def __init__(self):
        self.sample_rate = int(os.getenv('SAMPLE_RATE', 16000))
        self.channels = int(os.getenv('CHANNELS', 1))
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 1024))
        self.record_seconds = int(os.getenv('RECORD_SECONDS', 5))
        self.audio_dir = os.path.join(os.path.dirname(__file__), '../../data/audio')
        
        # 确保目录存在
        os.makedirs(self.audio_dir, exist_ok=True)

    def record(self):
        """录制音频并保存为WAV文件"""
        print("开始录音...")
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(self.audio_dir, f"recording_{timestamp}.wav")
        
        # 录制音频
        recording = sd.rec(
            int(self.sample_rate * self.record_seconds),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.int16
        )
        
        # 等待录音完成
        sd.wait()
        print("录音完成!")

        # 保存为WAV文件
        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.sample_rate)
            wf.writeframes(recording.tobytes())
        
        print(f"音频已保存至: {output_filename}")
        return output_filename

    def is_silent(self, audio_data, silence_threshold=500):
        """检查音频片段是否为静音"""
        return np.mean(np.abs(audio_data)) < silence_threshold

    def wait_for_speech(self):
        """等待检测到语音"""
        print("等待语音输入...")
        while True:
            audio_data = sd.rec(
                int(self.sample_rate * 0.1),  # 每0.1秒偵測一次是否有聲音
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.int16
            )
            sd.wait()
            
            if not self.is_silent(audio_data):
                print("检测到语音!")
                return True 