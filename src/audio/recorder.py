import os
import wave
import uuid
from datetime import datetime

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from resemblyzer import VoiceEncoder, preprocess_wav
import soundfile as sf
import json
from sklearn.metrics.pairwise import cosine_similarity

# 加載環境變數
load_dotenv(os.path.join(os.path.dirname(__file__), '../config/.env'))


class AudioRecorder:
    """使用 Resemblyzer 進行聲紋提取與語者識別的錄音器。"""

    def __init__(self):
        # ─── 基本參數 ──────────────────────────────────────────────────────────
        self.sample_rate: int = int(os.getenv("SAMPLE_RATE", 16000))
        self.channels: int = int(os.getenv("CHANNELS", 1))
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", 1024))
        self.record_seconds: int = int(os.getenv("RECORD_SECONDS", 3))  # 即時性更佳

        # ─── 路徑設定 ──────────────────────────────────────────────────────────
        self.audio_dir = os.path.join(os.path.dirname(__file__), "../../data/audio")
        self.speaker_db_path = os.path.join(os.path.dirname(__file__), "../../data/speaker_db")
        self.speaker_data_file = os.path.join(self.speaker_db_path, "speaker_data.json")
        #self.old_pickle_file = os.path.join(self.speaker_db_path, "speaker_data.pkl")  # 舊的pickle文件路徑

        # ─── 語者識別參數 ───────────────────────────────────────────────────────
        self.similarity_threshold: float = float(os.getenv("SIM_THRESHOLD", 0.75))  # 提高閾值
        self.encoder = VoiceEncoder()  # Resemblyzer 語者嵌入模型

        # 確保資料夾存在
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.speaker_db_path, exist_ok=True)
        
        # # 檢查是否需要從pickle轉換到json
        # if not os.path.exists(self.speaker_data_file) and os.path.exists(self.old_pickle_file):
        #     print("[Info] 發現舊格式數據文件，正在轉換為JSON格式...")
        #     try:
        #         # 嘗試加載pickle文件
        #         import pickle
        #         with open(self.old_pickle_file, "rb") as f:
        #             old_data = pickle.load(f)
                
        #         # 將數據轉換為新格式並保存
        #         self.speaker_db = old_data
        #         self._save_speaker_db()
                
        #         # 刪除舊的pickle文件
        #         os.remove(self.old_pickle_file)
        #         print("[Info] 成功將數據從pickle轉換為JSON格式，並刪除了舊文件。")
        #     except Exception as e:
        #         print(f"[Warning] 轉換失敗: {e}，將創建新的數據庫。")
        #         self.speaker_db = {"speakers": {}}
        # else:
            # 載入／建立語者資料庫
        self.speaker_db = self._load_speaker_db()
        
        # 重置聲紋數據庫選項（需要時設為True）
        reset_db = False
        if reset_db and os.path.exists(self.speaker_data_file):
            os.remove(self.speaker_data_file)
            print("[Info] 已重置語者資料庫")
            self.speaker_db = {"speakers": {}}
        
        # 清理資料庫（可選）
        # self.clean_speaker_database()

    # ╭─────────────────────────────── 私有方法 ─────────────────────────────╮
    def _load_speaker_db(self):
        """從磁碟載入語者資料庫，若無則回傳空白資料結構"""
        if os.path.exists(self.speaker_data_file):
            try:
                with open(self.speaker_data_file, "r", encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Warning] 載入語者資料庫失敗: {e}，將重新建立。")
        return {"speakers": {}}

    def _save_speaker_db(self):
        """將語者資料庫儲存到磁碟"""
        speakers_copy = {"speakers": {}}
        for spk_id, data in self.speaker_db["speakers"].items():
            speakers_copy["speakers"][spk_id] = {
                "created_at": data.get("created_at", datetime.now().isoformat()),
                "conversations": data.get("conversations", [])
            }
            
            if "embeddings" in data:
                embeddings_list = []
                for emb in data["embeddings"]:
                    if hasattr(emb, 'tolist'):  # 如果是numpy陣列
                        embeddings_list.append(emb.tolist())
                    else:  # 已經是列表
                        embeddings_list.append(emb)
                speakers_copy["speakers"][spk_id]["embeddings"] = embeddings_list
                
        with open(self.speaker_data_file, "w", encoding='utf-8') as f:
            json.dump(speakers_copy, f, ensure_ascii=False, indent=2)
        print(f"[Info] 語者資料庫已更新，目前共有 {len(self.speaker_db['speakers'])} 位說話者。")
        
    def clean_speaker_database(self):
        """清理語者資料庫，移除不一致的嵌入"""
        print("[Info] 開始清理語者資料庫...")
        for speaker_id, data in list(self.speaker_db['speakers'].items()):
            # 確保所有嵌入向量都是numpy陣列
            embeddings = []
            for emb in data["embeddings"]:
                if isinstance(emb, list):
                    embeddings.append(np.array(emb))
                else:
                    embeddings.append(emb)
            
            # 如果只有一個嵌入向量，保留
            if len(embeddings) <= 1:
                continue
                
            # 計算每個嵌入向量與其他向量的平均相似度
            to_keep = []
            for i, embed1 in enumerate(embeddings):
                avg_similarity = 0
                count = 0
                
                for j, embed2 in enumerate(embeddings):
                    if i != j:
                        similarity = cosine_similarity([embed1], [embed2])[0][0]
                        avg_similarity += similarity
                        count += 1
                
                if count > 0:
                    avg_similarity /= count
                    
                # 如果與其他嵌入的平均相似度高，則保留
                if avg_similarity > 0.9:
                    to_keep.append(embed1)
            
            # 如果過濾後沒有保留任何嵌入，保留原始第一個
            if len(to_keep) == 0 and len(embeddings) > 0:
                to_keep = [embeddings[0]]
                    
            # 更新該說話者的嵌入向量
            self.speaker_db['speakers'][speaker_id]['embeddings'] = [embed.tolist() for embed in to_keep]
            print(f"[Info] 已清理說話者 {speaker_id} 的嵌入，保留 {len(to_keep)}/{len(embeddings)} 個特徵")
            
        self._save_speaker_db()
        print("[Info] 語者資料庫清理完成")

    # ╰─────────────────────────────── 私有方法 ─────────────────────────────╯

    # ╭─────────────────────────────── Public API ───────────────────────────╮
    def wait_for_speech(self) -> bool:
        """持續監聽麥克風，直到偵測到非靜音訊號才返回 True"""
        print("等待語音輸入…")
        while True:
            audio_data = sd.rec(int(self.sample_rate * 0.2),  # 每 0.2 秒檢測一次
                                samplerate=self.sample_rate,
                                channels=self.channels,
                                dtype=np.int16)
            sd.wait()
            if not self._is_silent(audio_data):
                print("偵測到語音！")
                return True

    def record(self):
        """錄製固定長度語音 → 存 wav → 語者識別 → 回傳 (檔名, speaker_id, 相似度)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(self.audio_dir, f"recording_{timestamp}.wav")

        print("開始錄音…")
        recording = sd.rec(int(self.sample_rate * self.record_seconds),
                           samplerate=self.sample_rate,
                           channels=self.channels,
                           dtype=np.int16)
        sd.wait()
        print("錄音完成！")

        # 儲存 wav
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # int16 -> 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(recording.tobytes())
        print(f"音訊已保存至: {wav_path}")

        # 執行語者識別
        speaker_id, similarity = self.identify_speaker(wav_path)
        return wav_path, speaker_id, similarity

    # ╰─────────────────────────────── Public API ───────────────────────────╯

    # ╭─────────────────────────────── Helper Functions ─────────────────────╮
    def _is_silent(self, audio_data, silence_threshold: int = 500) -> bool:
        """檢測短音訊是否安靜 (極簡能量法)"""
        return np.mean(np.abs(audio_data)) < silence_threshold

    def _extract_embedding(self, audio_file: str):
        """利用 Resemblyzer 取得 256‑D 語者嵌入"""
        try:
            wav, sr = sf.read(audio_file)
            wav = preprocess_wav(wav, source_sr=sr)
            embed = self.encoder.embed_utterance(wav)  # ndarray (256,)
            return embed
        except Exception as e:
            print(f"[Error] 提取嵌入時發生錯誤: {e}")
            return None

    def identify_speaker(self, audio_file: str):
        """比對資料庫決定語者 ID；若為新語者則註冊，並列印候選相似度"""
        embed = self._extract_embedding(audio_file)
        # 0) 嵌入失敗 → 回傳 unknown + 0.0
        if embed is None:
            unknown_id = f"unknown_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            return unknown_id, 0.0
            
        # 1) 首次運行 → 直接建立新用戶
        if not self.speaker_db["speakers"]:
            print("[Info] 數據庫為空，這是第一次運行，直接創建新用戶")
            first_id = f"speaker_{str(uuid.uuid4())[:8]}"
            self.speaker_db["speakers"][first_id] = {
                "embeddings": [embed.tolist()],  # 將numpy數組轉換為列表以便JSON序列化
                "created_at": datetime.now().isoformat(),
                "conversations": []
            }
            print(f"🆕 創建首位說話者 → {first_id}")
            self._save_speaker_db()
            return first_id, 1.0  # 返回1.0的相似度，確保不會觸發再次確認

        best_id, best_sim = None, 0.0
        similarities = []

        for spk_id, data in self.speaker_db["speakers"].items():
            # 檢查是否存在embeddings字段
            if "embeddings" not in data or not data["embeddings"]:
                # 如果沒有embeddings，為這個用戶創建一個空的embeddings列表
                self.speaker_db["speakers"][spk_id]["embeddings"] = []
                continue  # 跳過這個用戶，因為沒有embeddings無法計算相似度
                
            # 將JSON中的列表轉換回numpy陣列
            embeddings = []
            for emb in data["embeddings"]:
                if isinstance(emb, list):
                    embeddings.append(np.array(emb))
                else:
                    embeddings.append(emb)
                    
            if not embeddings:
                continue
                
            avg_vec = np.mean(np.vstack(embeddings), axis=0)
            sim = cosine_similarity([embed], [avg_vec])[0][0]
            similarities.append((spk_id, sim))
            if sim > best_sim:
                best_sim, best_id = sim, spk_id

        # 列印所有候選相似度
        print("\n🧠 辨識候選相似度列表：")
        for spk_id, sim in sorted(similarities, key=lambda x: x[1], reverse=True):
            print(f" - 語者 {spk_id}: 相似度 {sim:.4f}")
        print("")

        # 2) 已知語者
        if best_sim >= self.similarity_threshold and best_id is not None:
            print(f"✅ 識別到已知說話者: {best_id} (相似度 {best_sim:.4f})")
            # 將新的嵌入添加到數據庫，確保添加為列表格式
            self.speaker_db["speakers"][best_id]["embeddings"].append(embed.tolist())
            self._save_speaker_db()
            return best_id, best_sim

        # 3) 新語者 → 回傳 new_id + 0.0
        new_id = f"speaker_{str(uuid.uuid4())[:8]}"
        self.speaker_db["speakers"][new_id] = {
            "embeddings": [embed.tolist()],  # 確保存儲為列表，而不是numpy數組
            "created_at": datetime.now().isoformat(),
            "conversations": []  # 確保添加conversations字段
        }
        print(f"🆕 註冊新說話者 → {new_id}")
        self._save_speaker_db()
        return new_id, 0.0


        
    def create_new_speaker(self, audio_file: str):
        """強制創建新說話者（用於手動確認後）"""
        embed = self._extract_embedding(audio_file)
        
        if embed is None:
            unknown_id = f"unknown_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            return unknown_id
        
        new_id = f"speaker_{str(uuid.uuid4())[:8]}"
        self.speaker_db["speakers"][new_id] = {
            "embeddings": [embed.tolist()],  # 轉換為列表以便JSON序列化
            "created_at": datetime.now().isoformat(),
            "conversations": []
        }
        
        print(f"[Info] 已創建新說話者，ID: {new_id}")
        self._save_speaker_db()
        
        return new_id
    # ╰─────────────────────────────── Helper Functions ─────────────────────╯ 