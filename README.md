
# 🎙️ 語音指令辨識與語音回應系統

這是一個基於 Google Gemini + Google Speech-to-Text + Google Text-to-Speech API 的語音指令辨識與互動系統，支援繁體中文，具備語音錄製、語音辨識、指令分類、行動規劃、查詢與聊天等功能，並以語音回應使用者。

---

## 🧩 系統架構與功能模組

### 📥 語音輸入
- 使用 `sounddevice` 錄製語音。
- 偵測語音活動 (Voice Activity Detection) 自動啟動錄音。

### 🧠 語音轉文字 (`SpeechToText`)
- 使用 Google Cloud Speech-to-Text API 將語音轉成文字。
- 支援自動標點、繁體中文辨識。
- 自動分類指令為：
  - `聊天`
  - `查詢`
  - `行動`

### 📚 指令分類與處理 (`CommandClassifier`)
- 使用 Gemini Pro 模型進行指令意圖判斷。
- 三種指令處理方式：
  - `聊天`：與 Gemini 對話。
  - `查詢`：產生搜尋關鍵字，使用 Google Custom Search 抓取資訊並回應。
  - `行動`：從任務產生機器人執行動作計劃。

### 🔊 語音合成與回應 (`ResponseSpeaker`)
- 使用 Google Cloud Text-to-Speech 將回應轉成語音。
- 使用 `pygame` 播放語音。

### 💾 歷史記錄保存
- 自動儲存每次對話或指令執行紀錄於 `data/` 資料夾中。
  - `chat_history/`
  - `query_history/`
  - `movement_history/`

---

## 🛠️ 安裝與設定

### 1️⃣ 安裝必要套件
```bash
pip install -r requirements.txt
```

### 2️⃣ 設定環境變數 `.env`

在 `config/.env` 中填入以下內容：

```env
SAMPLE_RATE=16000
CHANNELS=1
CHUNK_SIZE=1024
RECORD_SECONDS=5

GOOGLE_API_KEY=你的Speech-to-Text金鑰
GOOGLE_API_KEY_Gemini=你的Gemini API金鑰
GOOGLE_SEARCH_API_KEY=你的Google搜尋API金鑰
GOOGLE_SEARCH_CX=你的搜尋引擎ID
```

### 3️⃣ Google Cloud 認證
將你的 `Google Cloud Text-to-Speech` JSON 憑證存放於 `config/google_cloud_credentials.json`。

---

## ▶️ 如何執行

### 執行主程式
```bash
python src/main.py
```

### 測試語音回應
```bash
python src/utils/text_to_speech.py
```

---

## 📂 資料夾結構
```
project/
├── assets/
│   ├── command_type.json          # 指令分類範例
│   └── movement_deployment.json   # 動作設定
├── config/
│   ├── .env
│   └── google_cloud_credentials.json
├── data/
│   ├── audio/                     # 錄音資料
│   ├── audio_output/             # 語音回應檔案
│   ├── chat_history/
│   ├── query_history/
│   ├── movement_history/
│   └── transcripts/
├── src/
│   ├── audio/
│   │   ├── recorder.py
│   │   └── speech_to_text.py
│   ├── utils/
│   │   ├── command_classifier.py
│   │   └── text_to_speech.py
│   └── main.py
```

---

## 📋 範例輸入/輸出

### 🎤 指令：
```
請幫我送這張請購單去給工讀生
```

### 🤖 辨識結果：
```
分類結果：行動
動作計劃：
{
  "動作順序": ["M01", "M04"],
  "說明": [
    "前往影印室領取請購單",
    "將請購單交給工讀生"
  ]
}
```

### 🔊 語音回應：
```
我將執行以下任務：前往影印室領取請購單，並將請購單交給工讀生。
```

---

## ✅ TODO / 未來發展

- [ ] 增加 VAD 降噪與更準確的靜音檢測
- [ ] 整合 UI 前端或 Web 控制介面
- [ ] 支援多語言
- [ ] 整合行動模擬視覺化介面

---

## 🧑‍💻 作者
由專業軟體工程師開發，結合語音辨識、AI 對話、機器人任務規劃的完整範例。

---

## 📜 授權
MIT License
```
