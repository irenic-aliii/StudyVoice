# StudyVoice

An AI voice copilot for students: record a spoken question, StudyVoice listens,
remembers conversation context, answers concisely, and speaks the answer back.

## Architecture

```
Mic recording (Gradio, forced WAV)
        |
        v
Read raw audio bytes
        |
        v
Send audio bytes directly to Gemini 2.5 Flash
(no separate speech-to-text step; audio is appended
 to a running multi-turn conversation history)
        |
        v
Gemini returns a concise text answer
        |
        v
gTTS converts the answer to a unique MP3
        |
        v
Text shown in transcript box + audio autoplays
```

## Features

- Microphone recording directly in the browser, no file uploads needed.
- Gemini 2.5 Flash understands the audio natively - no separate STT service.
- Conversation memory: later questions can reference facts from earlier turns.
- Every response is spoken back with a unique, non-colliding audio file.
- Clear, specific error messages in the UI; full tracebacks in the terminal.

## Tech stack

- Python
- Gradio (simple transcript box UI, chosen for compatibility/reliability
  over a fancier Chatbot component)
- `google-genai` (current official SDK - not the deprecated `google-generativeai`)
- Gemini 2.5 Flash
- gTTS

## Setup

See `RUN_ME_FIRST.txt` for exact Windows PowerShell commands.

Short version:

```
python -m pip install -r requirements.txt
$env:GEMINI_API_KEY="your-key-here"
python app.py
```

## Demo flow

1. Record: "My name is Ali and I have a Python exam on Friday."
2. StudyVoice replies out loud, e.g. acknowledging the exam.
3. Record: "What exam do I have and when?"
4. StudyVoice correctly answers "Python exam on Friday" - proving the
   conversation memory works across turns.

## Phase 2: voice-to-study-plan

StudyVoice can also detect when you're asking for a study plan and show it
in a dedicated "TODAY'S STUDY PLAN" panel, instead of just replying in
speech/transcript.

**How intent is detected:** rather than fragile keyword matching (e.g.
`if "plan" in text`), every Gemini call now uses structured JSON output
(`response_mime_type="application/json"` + a pydantic `response_schema`).
Gemini itself classifies each turn as one of:

- `chat` - a normal question/answer, plan panel untouched
- `create_plan` - a new study plan, shown in the panel
- `update_plan` - a change to the existing plan, replacing it in the panel

**How memory carries the plan across turns:** the model's full structured
JSON reply (not just the spoken summary) is stored as its turn in the same
conversation history used for regular memory. So when you ask to modify the
plan, Gemini already has the exact previous plan JSON in context and returns
a complete revised plan, which replaces the one in the panel.

**Demo flow:**

1. Record: "I have a Python exam tomorrow. Make me a 5 hour study plan
   focused on loops, functions, and OOP."
   -> StudyVoice speaks a short summary, and the full plan appears in the
   "TODAY'S STUDY PLAN" panel.
2. Record: "Make OOP two hours and reduce loops."
   -> The existing plan panel updates in place with the new timing -
   it does not create a second, unrelated plan.
3. Record: "What's a good way to remember Python scope rules?"
   -> Normal chat answer; the plan panel stays exactly as it was.

## Phase 3: UI/UX redesign

This phase only touched presentation. No backend architecture, Gemini logic,
structured output schema, memory handling, or dependency versions changed.

**What changed:**
- New dark, premium visual theme (near-black/navy background, violet + blue
  accents) applied entirely through `CUSTOM_CSS` in `app.py`.
- Header bar with the StudyVoice name, an "AI Online" status badge, and the
  tagline "Speak your goal. Get a plan. Refine it by voice."
- Three-card layout: Voice Copilot (left), Conversation (center), Today's
  Study Plan (right, styled as the visually dominant "hero" panel).
- `render_plan_markdown()` now builds a styled HTML timeline (time chip +
  topic + goal per block, breaks visually differentiated) instead of plain
  markdown headers. It is still fed the exact same `StudyPlan` data and is
  called from the exact same place in `handle_turn` - only the string it
  returns looks different.
- Removed the separate "Ask StudyVoice" button. There is now exactly one way
  to submit: record, stop, and StudyVoice responds automatically
  (`audio_in.stop_recording(...)`), which was already wired up before and is
  unchanged here.
- The conversation transcript keeps the original `gr.Textbox` component and
  the original text it receives from `handle_turn` untouched - it is only
  restyled with CSS, per the "don't break transcript logic" requirement.

**What did NOT change:** `StudyBlock`/`StudyPlan`/`AssistantAction` schemas,
`SYSTEM_INSTRUCTION`, `MODEL_NAME`, `get_client()`, `extract_assistant_action()`,
`handle_turn()`, the Gemini call/config, gTTS usage, and `requirements.txt`
are all byte-for-byte identical to Phase 2 (verified with an AST diff).

## Fix note: structured response parsing

Earlier builds parsed only `response.text` with `json.loads`. When a full
study plan pushed the JSON near the output token limit, generation could be
cut short mid-string, `json.loads` would fail, and the code fell back to
showing the raw (partial) JSON in the transcript.

Fixed by:
- Preferring `response.parsed` (the `google-genai` SDK's own validated
  result) over manual text parsing, with a dict and then a text-based
  fallback behind it.
- Raising `MAX_OUTPUT_TOKENS` so a full study plan's JSON has enough room
  to finish generating instead of being truncated.
- If structured parsing genuinely fails anyway, the turn is dropped safely:
  the Status box shows a short message, the terminal gets the technical
  detail, and neither the transcript nor conversation memory is touched by
  the broken output.
