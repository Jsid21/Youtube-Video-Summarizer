import os
import tempfile
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from youtube_transcript_api.formatters import TextFormatter
from langdetect import detect
import streamlit as st
import pytube
import whisper
import torch

# Configure Google API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

prompt = """You are a YouTube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points with limited words.
The transcript may be in any language but your summary should be in English. Please provide the summary of the text given here (starting giving response directly): """

@st.cache_resource
def load_whisper_model():
    # Load the Whisper model for audio transcription (use "tiny" for faster results)
    model_size = "base"  # Options: "tiny", "base", "small", "medium", "large"
    if torch.cuda.is_available():
        return whisper.load_model(model_size).cuda()
    else:
        return whisper.load_model(model_size)

def download_audio(youtube_url):
    try:
        yt = pytube.YouTube(youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"{yt.video_id}.mp4")
        
        # Download the audio
        audio_stream.download(output_path=temp_dir, filename=f"{yt.video_id}.mp4")
        
        return temp_file, yt.title
    except Exception as e:
        st.error(f"Error downloading audio: {str(e)}")
        return None, None

def transcribe_audio(audio_file_path):
    try:
        model = load_whisper_model()
        result = model.transcribe(audio_file_path)
        return result["text"], result["language"]
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None, None

def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1]
        
        # First try using YouTube Transcript API
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to get manual transcripts first (usually higher quality)
            try:
                transcript_obj = transcript_list.find_manually_created_transcript()
            except:
                # If no manual transcript, get generated ones
                try:
                    transcript_obj = transcript_list.find_generated_transcript()
                except:
                    # Get any available transcript - fix for the dict_values issue
                    if transcript_list._manually_created_transcripts:
                        transcript_obj = list(transcript_list._manually_created_transcripts.values())[0]
                    elif transcript_list._generated_transcripts:
                        transcript_obj = list(transcript_list._generated_transcripts.values())[0]
                    else:
                        raise NoTranscriptFound(video_id)
            
            # Get transcript in original language
            transcript_text = transcript_obj.fetch()
            
            # Format transcript
            formatter = TextFormatter()
            transcript = formatter.format_transcript(transcript_text)
            
            # Detect language
            try:
                language = detect(transcript)
            except:
                language = "unknown"
            
            return transcript, language, "caption"
            
        except (NoTranscriptFound, TranscriptsDisabled, Exception) as e:
            # If no captions available, fallback to audio transcription
            st.info("No captions found. Falling back to audio transcription (this may take a few minutes)...")
            audio_file, _ = download_audio(youtube_video_url)
            
            if audio_file:
                transcript, language = transcribe_audio(audio_file)
                # Clean up the temporary file
                try:
                    os.remove(audio_file)
                except:
                    pass
                
                if transcript:
                    return transcript, language, "audio"
                
            raise Exception("Failed to extract transcript from both captions and audio")

    except Exception as e:
        st.error(f"Error extracting transcript: {str(e)}")
        return None, None, None

def generate_gemini_content(transcript_text, prompt, source_language):
    try:
        # For very large transcripts, we may need to truncate
        max_tokens = 30000
        if len(transcript_text) > max_tokens:
            transcript_text = transcript_text[:max_tokens] + "..."
            
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Add language info to prompt if known
        if source_language and source_language != "unknown":
            enhanced_prompt = prompt + f" Note that the original transcript is in {source_language}. "
        else:
            enhanced_prompt = prompt
            
        response = model.generate_content(enhanced_prompt + transcript_text)
        return response.text
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def get_video_title(video_id):
    try:
        yt = pytube.YouTube(f"https://www.youtube.com/watch?v={video_id}")
        return yt.title
    except:
        return "YouTube Video"

def extract_video_id(youtube_url):
    """Extract video ID from different YouTube URL formats"""
    if "youtube.com/watch?v=" in youtube_url:
        # Standard format: https://www.youtube.com/watch?v=VIDEO_ID
        video_id = youtube_url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
        # Short format: https://youtu.be/VIDEO_ID
        video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
    elif "youtube.com/embed/" in youtube_url:
        # Embed format: https://www.youtube.com/embed/VIDEO_ID
        video_id = youtube_url.split("embed/")[1].split("?")[0]
    elif "youtube.com/v/" in youtube_url:
        # Old embed format: https://www.youtube.com/v/VIDEO_ID
        video_id = youtube_url.split("v/")[1].split("?")[0]
    else:
        raise ValueError("Unsupported YouTube URL format")
    
    return video_id

# Streamlit UI
st.title("Multilingual YouTube Transcript Summarizer")
st.write("This app summarizes YouTube videos in any language and provides the summary in English")
st.write("Works with or without captions!")

youtube_link = st.text_input("Enter YouTube Video Link:")

if youtube_link:
    try:
        video_id = extract_video_id(youtube_link)
        video_title = get_video_title(video_id)
        
        st.subheader(f"Video: {video_title}")
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg")
        
        if st.button("Get Detailed Summary"):
            with st.spinner("Extracting transcript and generating summary..."):
                transcript_text, source_language, source_type = extract_transcript_details(youtube_link)
                
                if transcript_text:
                    if source_type == "caption":
                        st.success(f"Transcript extracted from captions (Detected language: {source_language.upper() if source_language else 'Unknown'})")
                    else:
                        st.success(f"Transcript generated from audio (Detected language: {source_language.upper() if source_language else 'Unknown'})")
                    
                    with st.expander("View Original Transcript"):
                        st.text(transcript_text[:1000] + "..." if len(transcript_text) > 1000 else transcript_text)
                    
                    summary = generate_gemini_content(transcript_text, prompt, source_language)
                    
                    if summary:
                        st.markdown("## Summary (English):")
                        st.write(summary)
                        
                        # Option to download summary
                        st.download_button(
                            label="Download Summary",
                            data=summary,
                            file_name=f"{video_title}_summary.txt",
                            mime="text/plain"
                        )
                else:
                    st.error("Failed to extract transcript from both captions and audio.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please enter a valid YouTube link (e.g., https://www.youtube.com/watch?v=VIDEO_ID or https://youtu.be/VIDEO_ID)")