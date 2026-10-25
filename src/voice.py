"""Voice layer for VoiceOps 2026 ZentroMart support agent.

Uses official Gnani STT and TTS SDKs:
- GnaniSTTClient: stt.transcribe(path, language_code="en-IN")["transcript"]
- GnaniTTSClient: tts.synthesize(text, voice="Kaveri", model="timbre-v2.5", language="en-IN", output_file=...)
Enforces graceful degradation on empty transcript or TTS errors, and marks live voice BLOCKED if GNANI_API_KEY is missing.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple
import dotenv

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# Check if Gnani is configured
GNANI_KEY = os.getenv("GNANI_API_KEY")
VOICE_AVAILABLE = bool(GNANI_KEY and GNANI_KEY.strip())

try:
    from gnani.stt import GnaniSTTClient
    from gnani.tts import GnaniTTSClient
    _GNANI_INSTALLED = True
except ImportError:
    _GNANI_INSTALLED = False


class VoiceHandler:
    def __init__(self):
        self.api_key = os.getenv("GNANI_API_KEY")
        if not self.api_key or not self.api_key.strip():
            self.status = "BLOCKED: GNANI_API_KEY not configured in environment"
            self.stt_client = None
            self.tts_client = None
        else:
            try:
                self.stt_client = GnaniSTTClient()
                self.tts_client = GnaniTTSClient()
                self.status = "ACTIVE"
            except Exception as e:
                self.status = f"BLOCKED: Failed to initialize Gnani client: {e}"
                self.stt_client = None
                self.tts_client = None

    def is_blocked(self) -> bool:
        return self.stt_client is None or self.tts_client is None

    def process_audio(
        self,
        audio_path: str,
        agent,
        output_audio_path: Optional[str] = None,
    ) -> Tuple[str, str, Optional[str], list]:
        """Process one customer audio turn.

        Args:
            audio_path: Path to customer input audio (.wav)
            agent: Step 3/4 ConversationalAgent instance
            output_audio_path: Destination path for synthesized reply audio (.wav)

        Returns:
            (transcript, reply_text, output_audio_path_or_none, chunks_used)
        """
        if self.is_blocked():
            logger.warning("Voice processing skipped: %s", self.status)
            raise RuntimeError(f"Live voice is BLOCKED: {self.status}")

        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Input audio file not found: {audio_path}")

        # 1. Transcribe audio using Gnani STT
        try:
            stt_res = self.stt_client.transcribe(str(audio_file), language_code="en-IN")
            heard = stt_res.get("transcript", "")
        except Exception as e:
            logger.error("STT transcription failure on %s: %s", audio_path, e)
            heard = ""

        # 2. Handle empty or unclear transcript
        if not heard or not heard.strip():
            reply_text = "Sorry, I didn't catch that. Could you say it again?"
            chunks_used = []
        else:
            # 3. Call agent.respond()
            reply_text = agent.respond(heard)
            chunks_used = getattr(agent, "last_chunks_used", [])

        # 4. Synthesize speech using Gnani TTS
        saved_audio = None
        if output_audio_path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)
                self.tts_client.synthesize(
                    reply_text,
                    voice="Kaveri",
                    model="timbre-v2.5",
                    language="en-IN",
                    output_file=output_audio_path,
                )
                saved_audio = output_audio_path
            except Exception as e:
                # TTS failure -> save text, log error, continue
                logger.error("TTS synthesis error for text '%s': %s", reply_text, e)
                saved_audio = None

        return heard, reply_text, saved_audio, chunks_used
