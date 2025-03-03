# Multilingual YouTube Video Transcriber & Summarizer

A Streamlit-based web application that extracts transcripts from YouTube videos, translates them into English, and provides a summarized version using Google's Gemini API. The app works with both captioned videos and those without captions by utilizing Whisper for audio transcription.

## Features
- **Supports Multiple Languages**: Extracts transcripts in any language and provides an English summary.
- **Handles Captions & Audio**: Uses YouTube Transcript API for captions and Whisper AI for audio transcription if captions are unavailable.
- **Google Gemini API**: Generates a concise summary from the extracted transcript.
- **Streamlit UI**: Simple web interface to input a YouTube URL and get the transcript and summary.
- **Downloadable Summary**: Users can download the summarized transcript as a text file.

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jsid21/Youtube-Video-Summarizer.git
   cd youtube-transcriber
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project directory and add your Google API key:
     ```bash
     GOOGLE_API_KEY=your_google_api_key
     ```

## Usage
1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
2. **Enter a YouTube video URL** and click "Get Detailed Summary."
3. The app will:
   - Extract the transcript (via captions or Whisper AI for audio processing).
   - Summarize the transcript using the Gemini API.
   - Display and allow downloading of the summary.

## Technologies Used
- **Python**
- **Streamlit** - For the web interface
- **YouTube Transcript API** - Extracts captions if available
- **Pytube** - Downloads YouTube audio
- **OpenAI Whisper** - Converts audio to text
- **Google Gemini API** - Summarizes the transcript
- **Langdetect** - Detects transcript language
