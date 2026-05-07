import logging
from typing import Tuple, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

logger = logging.getLogger(__name__)

def fetch_transcript(video_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Attempts to fetch the transcript for a YouTube video.
    Returns a tuple of (transcript_text, language_code).
    If it fails, returns (None, None).
    """
    try:
        # Get all available transcripts for the video
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Preferred languages: hi, hi-IN, en
        target_langs = ['hi', 'hi-IN', 'en']
        
        transcript_obj = None
        
        # 1. Try to find manually created transcripts in target languages
        for lang in target_langs:
            try:
                transcript_obj = transcript_list.find_manually_created_transcript([lang])
                break
            except:
                continue
                
        # 2. Try auto-generated transcripts in target languages
        if not transcript_obj:
            for lang in target_langs:
                try:
                    transcript_obj = transcript_list.find_generated_transcript([lang])
                    break
                except:
                    continue
                    
        # 3. Multilingual fallback: If neither, just get whatever the first available transcript is
        if not transcript_obj:
            for transcript in transcript_list:
                transcript_obj = transcript
                break
                
        if transcript_obj:
            # Fetch the actual data
            data = transcript_obj.fetch()
            
            # Format to plain text
            formatter = TextFormatter()
            text = formatter.format_transcript(data)
            
            # Basic cleanup: remove excessive newlines
            clean_text = " ".join([line.strip() for line in text.split('\n') if line.strip()])
            
            return clean_text, transcript_obj.language_code
            
        return None, None
        
    except Exception as e:
        logger.warning(f"Could not fetch transcript for {video_id}: {str(e)}")
        return None, None
