CLIPFORGE
=========
frontend/  -> static page (host anywhere, e.g. Cloudflare Pages).
backend/   -> the engine (Python + FFmpeg). Needs a real server; Pages can't run it.

WHAT IT DOES (all real processing, nothing simulated)
  1. Takes an uploaded video, or a direct link to a video file you have the right to use.
  2. FFmpeg extracts the audio; faster-whisper transcribes it with word-level timestamps.
  3. The whole video is analysed: speech, audio energy (laughs/shouts/reactions) and
     scene changes. Long transcripts are split into chunks so nothing is skipped.
  4. With ANTHROPIC_API_KEY set, Claude picks and ranks the best moments. Without a key
     the server ranks windows by the same audio/speech/scene signals and says so in the
     clip's description - it never claims AI picked them.
  5. Start/end are snapped to sentence and pause boundaries so nothing is cut mid-sentence.
  6. Each clip is cut, reframed to the chosen format (9:16, 16:9, 1:1, 4:5) following the
     main face (or the main motion if no face), captioned word-by-word with your style
     settings, and encoded as H.264/AAC MP4 with the original audio.
  7. Every MP4 is verified (video + audio present, full decode check) before it is offered.
     A clip that fails verification is reported as failed, never shown as done.

SETUP (server with Python 3.10+ and FFmpeg, 2-4 GB RAM; a GPU makes transcription faster)
   pip install -r backend/requirements.txt
   export ANTHROPIC_API_KEY=your_key               # AI moment picking (recommended)
   export CORS_ORIGINS=https://yourname.pages.dev  # only if frontend is hosted separately
   cd backend && uvicorn server:app --host 0.0.0.0 --port 8000
   Single computer: open http://localhost:8000 (leave API_BASE='' in frontend/index.html).
   Separate hosting: set  window.API_BASE='https://your-backend'  in frontend/index.html.
   Check the install:  GET /api/health

OPTIONAL SETTINGS (environment variables)
   WHISPER=small|medium|large-v3   speech model (bigger = more accurate, slower)
   WHISPER_DEVICE=auto|cpu|cuda    WHISPER_LANG=en  (default: auto-detect)
   MODEL=claude-sonnet-4-6         model used for moment picking
   MAX_JOBS=1                      videos processed at once
   MAX_UPLOAD_GB=10  MAX_DURATION_HOURS=6  RETENTION_HOURS=24
   SCENE_DETECT=0                  skip scene-change analysis (faster on very long videos)
   X264_CRF=18  X264_PRESET=fast   output quality / speed
   FONTS_DIR=/path/to/fonts        extra fonts for captions
   DATA_DIR=data                   where jobs are stored
   TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET   show Twitch title/live status for links

LINKS: WHAT IS AND ISN'T SUPPORTED
   - Direct video file links (.mp4/.mov/.webm...), incl. Dropbox share links: supported.
     Private/local network addresses are blocked.
   - YouTube and Twitch links are recognised (title, creator, live status), but neither
     platform offers an authorised way to download video files from a link, so CLIPFORGE
     does not do it. Download your own video from YouTube Studio / Twitch Creator Dashboard
     (or get the owner's permission) and upload the file instead.
   - Live streams can't be clipped while live; clip the recording once it finishes.

NOTES
   - Captions need the font installed on the server. If Impact/Arial/Georgia/Courier New
     aren't installed, the closest installed look-alike is used (e.g. Liberation Sans).
     Put .ttf files in a folder and set FONTS_DIR to get the exact fonts.
   - Output never upscales beyond the source's real resolution (e.g. a 720p source gives
     404x720 for 9:16) so quality is preserved rather than faked.
   - The backend also accepts two optional form/JSON settings that have no control in the
     current UI (kept unchanged on purpose): "mute":true on /api/upload and /api/import
     (remove audio) and "bg":"#000000" inside the subtitle settings (caption background box).
