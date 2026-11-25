# Voice Intent Collection Template

Fill `voice_intents_template.csv` with human-written Egyptian Arabic commands.

- Allowed `intent` values (`VoiceIntent` enum in `src/contracts/enums.py`):
  `READ_DOCUMENT`, `RECOGNIZE_CURRENCY`, `INDOOR_NAVIGATION`, `DESCRIBE_SCENE`,
  `DETECT_COLOR`, `READ_TRAFFIC_SIGN`, `DETECT_OBSTACLE`, `DETECT_EXPRESSION`
- Target: at least 30 sentences per intent from at least 3 distinct `speaker_id` contributors.
- Keep sentences natural Egyptian Arabic written by team members (never synthetic LLM text).
