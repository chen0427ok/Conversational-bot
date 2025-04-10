from audio.recorder import AudioRecorder
from audio.speech_to_text import SpeechToText
from utils.command_classifier import CommandClassifier
from utils.text_to_speech import ResponseSpeaker
import os
import json
from datetime import datetime

def main():
    # 初始化所有組件
    recorder = AudioRecorder()
    transcriber = SpeechToText()
    classifier = CommandClassifier()
    speaker = ResponseSpeaker()
    
    try:
        while True:
            print("\n等待語音輸入...")
            # 等待語音輸入
            if recorder.wait_for_speech():
                # 錄製音頻
                audio_file = recorder.record()
                
                # 轉換為文字並分類
                results = transcriber.transcribe_file(audio_file)
                
                if results:
                    print("\n識別結果：")
                    print("-" * 50)
                    
                    for result in results:
                        text = result['text']
                        command_type = result['command_type']
                        confidence = result['confidence']
                        
                        print(f"文本: {text}")
                        print(f"置信度: {confidence:.2%}")
                        print(f"命令類型: {command_type}")
                        print("-" * 50)
                        
                        # 根據命令類型處理
                        if command_type == '聊天':
                            response = classifier.chat_with_gemini(text)
                            classifier.save_chat_history(text, response, command_type)
                            print(f"\n聊天回應：\n{response}")
                            
                        elif command_type == '查詢':
                            response = classifier.handle_query(text)
                            classifier.save_query_history(text, response, command_type)
                            print(f"\n查詢結果：\n{response}")
                            
                        elif command_type == '行動':
                            response = classifier.handle_movement(text)
                            classifier.save_movement_history(text, response, command_type)
                            print("\n行動計劃：")
                            print(json.dumps(response, ensure_ascii=False, indent=2))
                        
                        # 獲取最新的歷史記錄文件
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        history_file = None
                        
                        if command_type == '聊天':
                            history_dir = os.path.join(os.path.dirname(__file__), '../data/chat_history')
                            history_file = os.path.join(history_dir, f"chat_{timestamp}.json")
                        elif command_type == '查詢':
                            history_dir = os.path.join(os.path.dirname(__file__), '../data/query_history')
                            history_file = os.path.join(history_dir, f"query_{timestamp}.json")
                        elif command_type == '行動':
                            history_dir = os.path.join(os.path.dirname(__file__), '../data/movement_history')
                            history_file = os.path.join(history_dir, f"movement_{timestamp}.json")
                        
                        # 播放語音回應
                        if history_file and os.path.exists(history_file):
                            print("\n正在生成語音回應...")
                            speaker.process_history_file(history_file)
                
                # 詢問是否繼續
                response = input("\n是否繼續錄音? (y/n): ")
                if response.lower() != 'y':
                    break
    
    except KeyboardInterrupt:
        print("\n程序已終止")
    except Exception as e:
        print(f"發生錯誤: {str(e)}")

if __name__ == "__main__":
    main() 