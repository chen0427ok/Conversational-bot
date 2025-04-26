from audio.recorder import AudioRecorder
from audio.speech_to_text_test import SpeechToText
from utils.command_classifier_claude import CommandClassifier
from utils.text_to_speech_test import ResponseSpeaker
import os
import json
from datetime import datetime

def main():
    # 初始化所有組件
    recorder = AudioRecorder()
    transcriber = SpeechToText()
    classifier = CommandClassifier()
    speaker = ResponseSpeaker()
    
    # 確保共用同一個speaker_db文件路徑
    classifier.speaker_data_file = recorder.speaker_data_file
    
    try:
        while True:
            print("\n等待語音輸入...")
            # 等待語音輸入
            if recorder.wait_for_speech():
                # 錄製音頻並進行語者辨識
                audio_file, speaker_id, similarity = recorder.record()
                print(f"🎤 語者辨識 → ID: {speaker_id}，相似度: {similarity:.4f}")
                
                # 將recorder的speaker_db同步到classifier
                classifier.speaker_db = recorder.speaker_db
                
                # 顯示識別結果和相似度
                if similarity < recorder.similarity_threshold:
                    print(f"⚠️ 相似度低於閾值 ({similarity:.2f} < {recorder.similarity_threshold})")
                    print(f"✅ 已自動註冊為新用戶: {speaker_id}")
                else:
                    print(f"✅ 已識別為已知說話者: {speaker_id} (相似度 {similarity:.2f})")
                
                # 轉換為文字
                transcript_text = transcriber.transcribe_file(audio_file)
                
                if transcript_text:
                    print("\n識別結果：")
                    print("-" * 50)
                    print(f"文本: {transcript_text}")
                    
                    # 分類命令
                    command_type = classifier.classify_command(transcript_text)
                    print(f"命令類型: {command_type}")
                    print("-" * 50)
                    
                    # 根據命令類型處理，並傳入語者ID以使用對話歷史
                    if command_type == '聊天':
                        # 傳入speaker_id以使用歷史對話
                        response = classifier.chat_with_gemini(transcript_text, speaker_id)
                        classifier.save_chat_history(transcript_text, response, command_type)
                        print(f"\n聊天回應：\n{response}")
                        
                    elif command_type == '查詢':
                        # 傳入speaker_id以使用歷史對話
                        response = classifier.handle_query(transcript_text, speaker_id)
                        classifier.save_query_history(transcript_text, response, command_type)
                        print(f"\n查詢結果：\n{response}")
                        
                    elif command_type == '行動':
                        # 傳入speaker_id以使用歷史對話
                        response = classifier.handle_movement(transcript_text, speaker_id)
                        classifier.save_movement_history(transcript_text, response, command_type)
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