"""
StudyVoice - AI Voice Copilot for Students (Phase 4: Voice Quiz)
-----------------------------------------------------------------
Phase 3 Study Agent preserved intact.
Phase 4 adds: VOICE QUIZ mode (PLAN → PRACTICE → DIAGNOSE).

Run with:
    python app.py

Requires the GEMINI_API_KEY environment variable to be set before launch.
"""

import os
import sys
import json
import traceback
import uuid
import tempfile
from typing import List, Optional

import gradio as gr
from google import genai
from google.genai import types
from gtts import gTTS
from pydantic import BaseModel


# ==========================================================================
# STUDY AGENT SCHEMAS  (Phase 3 – unchanged)
# ==========================================================================

class StudyBlock(BaseModel):
    time_range: str
    topic: str
    goal: str


class StudyPlan(BaseModel):
    title: str
    blocks: List[StudyBlock]


class AssistantAction(BaseModel):
    action_type: str          # "chat" | "create_plan" | "update_plan"
    spoken_reply: str
    study_plan: Optional[StudyPlan] = None


# ==========================================================================
# VOICE QUIZ SCHEMAS  (Phase 4)
# ==========================================================================

class QuizStartResult(BaseModel):
    """Returned by the first Gemini call that parses user intent + writes Q1."""
    topic: str
    total_questions: int
    difficulty: str           # "easy" | "medium" | "hard"
    first_question: str
    spoken_intro: str         # e.g. "Alright! Quiz time. Question 1: ..."


class QuizTurnResult(BaseModel):
    """Returned by every subsequent Gemini call (evaluate + next question / diagnosis)."""
    evaluation: str                    # "correct" | "partial" | "incorrect"
    points_awarded: float              # 1.0 | 0.5 | 0.0
    feedback: str                      # brief, 1-2 sentences
    strong_areas: List[str]
    weak_areas: List[str]
    new_difficulty: str               # "easy" | "medium" | "hard"
    next_question: Optional[str] = None   # None when quiz is finished
    is_final: bool = False
    # Final diagnosis fields (only when is_final=True)
    final_score_label: str = ""       # e.g. "4 / 5"
    performance_summary: str = ""
    recommended_next_action: str = ""
    spoken_summary: str = ""          # concise spoken close


# ==========================================================================
# GENERAL ASSISTANT SCHEMA  (Phase 5)
# ==========================================================================

class GeneralAction(BaseModel):
    action_type: str          # "chat" | "start_quiz"
    spoken_reply: str         # always filled; concise for spoken TTS
    quiz_topic: Optional[str] = None  # inferred topic when action_type="start_quiz"


# ==========================================================================
# Configuration
# ==========================================================================

MODEL_NAME = "gemini-3.1-flash-lite-preview"
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0.6

# Where generated speech files are written
AUDIO_OUT_DIR = os.path.join(tempfile.gettempdir(), f"studyvoice_{uuid.uuid4().hex[:8]}")
os.makedirs(AUDIO_OUT_DIR, exist_ok=True)

# ---------- Study Agent system prompt (Phase 3 – unchanged) ----------

SYSTEM_INSTRUCTION = (
    "You are StudyVoice, an AI study copilot for students. "
    "You must always respond with a single JSON object matching the required "
    "schema - never plain text outside that JSON. "
    "\n\n"
    "Set action_type to one of: "
    "'chat' for normal conversation and questions, "
    "'create_plan' when the student asks for a new study plan or schedule, "
    "'update_plan' when the student asks to change, adjust, or rebalance a "
    "study plan that was already created earlier in this conversation. "
    "\n\n"
    "spoken_reply must always be filled in, in 2-4 short sentences, written "
    "as if speaking directly to the student. "
    "For 'create_plan' and 'update_plan', spoken_reply must be a SHORT SUMMARY "
    "only (e.g. 'Here is your 5 hour plan covering loops, functions, and OOP.') "
    "- never read the full plan out loud in spoken_reply. "
    "\n\n"
    "study_plan must be included whenever action_type is 'create_plan' or "
    "'update_plan', and must be the COMPLETE plan (not just the changed part). "
    "When updating a plan, base the new full plan on the most recent study "
    "plan JSON found earlier in this conversation, changing only what the "
    "student asked to change and keeping the rest consistent. "
    "Use realistic clock-time ranges like '09:00-10:00' for each block, and "
    "include short breaks for plans longer than about 2 hours. "
    "Leave study_plan out entirely (null) for plain 'chat' turns. "
    "\n\n"
    "Remember facts from earlier turns in this conversation (including "
    "earlier study plans) and use them to answer follow-up requests "
    "correctly. Ask at most one clarifying question when something about a "
    "plan request is genuinely ambiguous (e.g. no duration given at all). "
    "Never say that you transcribed, listened to, or processed audio - just "
    "respond naturally as if you heard the student directly. "
    "Stay focused on studying, learning, planning, revision, and productivity."
)

# ---------- Quiz system prompts (Phase 4) ----------

QUIZ_START_SYSTEM = (
    "You are StudyVoice Quiz Master. The student has just spoken a quiz request. "
    "Extract: the topic, the number of questions (default 5, clamp to 3-10), and "
    "an initial difficulty (easy/medium/hard, default medium). "
    "Then write the first quiz question on that topic at the chosen difficulty. "
    "Return ONLY a JSON object matching the QuizStartResult schema. "
    "spoken_intro must be a friendly, concise spoken sentence introducing the quiz "
    "and stating Question 1 out loud (e.g. 'Alright! Five questions on Python OOP. "
    "Question 1: What is the difference between a class and an instance?'). "
    "Never include raw JSON in spoken_intro. "
    "Never say you processed audio."
)

QUIZ_TURN_SYSTEM = (
    "You are StudyVoice Quiz Master evaluating a spoken answer. "
    "You receive the quiz context (topic, question, question number, total, "
    "current difficulty, history) and the student's audio answer. "
    "Evaluate the answer as 'correct' (1.0 pts), 'partial' (0.5 pts), or "
    "'incorrect' (0.0 pts). Give brief 1-2 sentence feedback. "
    "Update strong_areas and weak_areas based on the full quiz history. "
    "Adapt difficulty: strong correct → may increase; partial → same; "
    "incorrect → reinforce weakness or decrease. "
    "If this is NOT the final question, set next_question to the next question, "
    "is_final to false, and leave final diagnosis fields empty. "
    "If this IS the final question, set is_final to true, next_question to null, "
    "fill final_score_label (e.g. '4 / 5'), performance_summary (2-3 sentences), "
    "recommended_next_action (1 sentence), and spoken_summary (concise, under "
    "30 words, e.g. 'Great effort! You scored 4 out of 5. Your main strength is "
    "inheritance; keep working on polymorphism.'). "
    "Return ONLY a JSON object matching the QuizTurnResult schema. "
    "Never include raw JSON in any spoken or feedback fields. "
    "Never say you processed audio."
)


# ---------- General Assistant system prompt (Phase 5) ----------

GENERAL_SYSTEM = (
    "You are StudyVoice General Assistant, a friendly and knowledgeable AI. "
    "Answer ANY question clearly and helpfully — academic, creative, technical, or general. "
    "You must always respond with a single JSON object matching the GeneralAction schema. "
    "\n\n"
    "Set action_type to: "
    "'chat' for all normal questions and conversation. "
    "'start_quiz' ONLY when the user explicitly asks to be tested, quizzed, or asked questions "
    "about a topic from this conversation (e.g. 'quiz me on that', 'test me', "
    "'ask me questions about what you just explained', 'let's do a quiz on this'). "
    "\n\n"
    "spoken_reply must always be filled in, concise (2-4 sentences for chat, "
    "1 sentence for start_quiz like 'Starting a quiz on [topic] now!'). "
    "For start_quiz, quiz_topic must be the exact topic inferred from the conversation context "
    "(e.g. 'Polymorphism in Python', 'Black holes', 'Recursion'). "
    "For chat, leave quiz_topic null. "
    "\n\n"
    "Remember the conversation context — follow-up references like 'it', 'that', 'this topic' "
    "refer to the most recently discussed subject. "
    "Never expose raw JSON in spoken_reply. "
    "Never say you processed audio or transcribed speech."
)


# ==========================================================================
# Gemini client
# ==========================================================================

def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "GEMINI_API_KEY is not set in this terminal session. "
            "Set it with: $env:GEMINI_API_KEY=\"your-key-here\" "
            "then restart the app."
        )
    return genai.Client(api_key=api_key)


def _is_rate_limit_error(exc):
    """Returns True if the exception looks like a 429 / rate-limit error."""
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg or "rate" in msg


RATE_LIMIT_MESSAGE = (
    "StudyVoice is temporarily at capacity. Please wait a moment and try again."
)


# ==========================================================================
# TTS helper
# ==========================================================================

def _speak(text: str):
    """Synthesise text to MP3, return path. Raises on failure."""
    unique_name = f"reply_{uuid.uuid4().hex}.mp3"
    out_path = os.path.join(AUDIO_OUT_DIR, unique_name)
    tts = gTTS(text=text, lang="en")
    tts.save(out_path)
    return out_path


# ==========================================================================
# STUDY AGENT parsing helpers  (Phase 3 – unchanged)
# ==========================================================================

PLAN_PLACEHOLDER = (
    '<div class="plan-placeholder">'
    '<div class="plan-placeholder-icon">🗓️</div>'
    "<div>No study plan yet.</div>"
    "<div>Ask StudyVoice to create one, for example:</div>"
    '<div class="plan-placeholder-example">'
    '"I have a Python exam tomorrow, make me a 5 hour study plan '
    'focused on loops, functions, and OOP."'
    "</div>"
    "</div>"
)


def _plan_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_plan_markdown(plan_dict):
    if not plan_dict:
        return PLAN_PLACEHOLDER

    title = plan_dict.get("title") or "Today's Study Plan"
    blocks = plan_dict.get("blocks") or []

    html = [f'<div class="plan-title">{_plan_escape(title)}</div>']
    html.append('<div class="plan-timeline">')
    for b in blocks:
        time_range = _plan_escape(b.get("time_range", ""))
        topic = _plan_escape(b.get("topic", ""))
        goal = _plan_escape(b.get("goal", ""))
        is_break = "break" in topic.lower()
        block_class = "plan-block plan-break" if is_break else "plan-block"
        goal_html = f'<div class="plan-goal">{goal}</div>' if goal else ""
        html.append(
            f'<div class="{block_class}">'
            f'<div class="plan-time">{time_range}</div>'
            f'<div class="plan-block-content">'
            f'<div class="plan-topic">{topic}</div>'
            f"{goal_html}"
            f"</div>"
            f"</div>"
        )
    html.append("</div>")
    return "\n".join(html)


def _strip_code_fence(raw_text):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def extract_assistant_action(response):
    """Returns (ok, spoken_reply, action_type, study_plan_dict, debug_type)"""
    parsed_obj = getattr(response, "parsed", None)
    debug_type = type(parsed_obj).__name__

    def _from_action(action: AssistantAction):
        spoken_reply = (action.spoken_reply or "").strip()
        if not spoken_reply:
            return None
        study_plan = action.study_plan.model_dump() if action.study_plan else None
        return spoken_reply, action.action_type or "chat", study_plan

    if isinstance(parsed_obj, AssistantAction):
        result = _from_action(parsed_obj)
        if result:
            return (True, *result, debug_type)

    if isinstance(parsed_obj, dict):
        try:
            action = AssistantAction.model_validate(parsed_obj)
            result = _from_action(action)
            if result:
                return (True, *result, debug_type)
        except Exception:
            pass

    raw_text = getattr(response, "text", None) or ""
    try:
        cleaned = _strip_code_fence(raw_text)
        action = AssistantAction.model_validate_json(cleaned)
        result = _from_action(action)
        if result:
            return (True, *result, debug_type)
    except Exception:
        pass

    return (False, None, None, None, debug_type)


# ==========================================================================
# STUDY AGENT turn handler  (Phase 3 – unchanged)
# ==========================================================================

def handle_turn(audio_path, history, transcript_text, plan_state):
    history = history or []
    plan_markdown_unchanged = render_plan_markdown(plan_state)

    if not audio_path:
        return (None, None, history, transcript_text,
                "No audio detected. Please record a message first.",
                plan_state, plan_markdown_unchanged)

    if not os.path.isfile(audio_path):
        return (None, None, history, transcript_text,
                "The recorded audio file could not be found on disk. Please try recording again.",
                plan_state, plan_markdown_unchanged)

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        if not audio_bytes:
            raise ValueError("Recorded audio file is empty.")
    except Exception:
        traceback.print_exc()
        return (None, None, history, transcript_text,
                "Could not read the recorded audio. Please try recording again.",
                plan_state, plan_markdown_unchanged)

    mime_type = "audio/wav"
    user_turn = types.Content(
        role="user",
        parts=[types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)],
    )
    working_history = history + [user_turn]

    try:
        client = get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=working_history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
                response_mime_type="application/json",
                response_schema=AssistantAction,
            ),
        )
    except RuntimeError as e:
        traceback.print_exc()
        return (None, None, history, transcript_text, str(e), plan_state, plan_markdown_unchanged)
    except Exception as e:
        traceback.print_exc()
        user_msg = RATE_LIMIT_MESSAGE if _is_rate_limit_error(e) else (
            f"Gemini request failed: {type(e).__name__}. Check the terminal for details."
        )
        return (None, None, history, transcript_text, user_msg, plan_state, plan_markdown_unchanged)

    raw_text = getattr(response, "text", None)
    if not raw_text or not raw_text.strip():
        return (None, None, history, transcript_text,
                "Gemini returned an empty response. Please try asking again.",
                plan_state, plan_markdown_unchanged)

    ok, spoken_reply, action_type, new_plan, debug_type = extract_assistant_action(response)
    print(f"[StudyVoice DEBUG] type(response.parsed)={debug_type} parsed_ok={ok} action_type={action_type!r}")

    if not ok:
        print("[StudyVoice DEBUG] raw response.text that failed to parse:")
        print(raw_text)
        return (None, None, history, transcript_text,
                "StudyVoice couldn't understand its own structured response. Please try asking again.",
                plan_state, plan_markdown_unchanged)

    updated_plan_state = new_plan if new_plan else plan_state
    updated_plan_markdown = render_plan_markdown(updated_plan_state)

    model_turn = types.Content(
        role="model",
        parts=[types.Part.from_text(text=raw_text.strip())],
    )
    updated_history = working_history + [model_turn]

    try:
        out_path = _speak(spoken_reply)
    except Exception:
        traceback.print_exc()
        turn_block = f"You: (voice message)\nStudyVoice: {spoken_reply}\n\n"
        return (None, None, updated_history, transcript_text + turn_block,
                "Text-to-speech failed, but here is the text answer above.",
                updated_plan_state, updated_plan_markdown)

    turn_block = f"You: (voice message)\nStudyVoice: {spoken_reply}\n\n"
    return (out_path, None, updated_history, transcript_text + turn_block,
            "", updated_plan_state, updated_plan_markdown)


# ==========================================================================
# VOICE QUIZ parsing helpers  (Phase 4)
# ==========================================================================

def _extract_quiz_start(response) -> Optional[QuizStartResult]:
    """Extract QuizStartResult from Gemini response. Returns None on failure."""
    parsed = getattr(response, "parsed", None)

    if isinstance(parsed, QuizStartResult):
        return parsed

    if isinstance(parsed, dict):
        try:
            return QuizStartResult.model_validate(parsed)
        except Exception:
            pass

    raw = getattr(response, "text", None) or ""
    try:
        return QuizStartResult.model_validate_json(_strip_code_fence(raw))
    except Exception:
        pass

    print(f"[QuizDebug] Failed to parse QuizStartResult. raw={raw[:300]}")
    return None


def _extract_quiz_turn(response) -> Optional[QuizTurnResult]:
    """Extract QuizTurnResult from Gemini response. Returns None on failure."""
    parsed = getattr(response, "parsed", None)

    if isinstance(parsed, QuizTurnResult):
        return parsed

    if isinstance(parsed, dict):
        try:
            return QuizTurnResult.model_validate(parsed)
        except Exception:
            pass

    raw = getattr(response, "text", None) or ""
    try:
        return QuizTurnResult.model_validate_json(_strip_code_fence(raw))
    except Exception:
        pass

    print(f"[QuizDebug] Failed to parse QuizTurnResult. raw={raw[:300]}")
    return None


# ==========================================================================
# QUIZ STATE helpers
# ==========================================================================

def _blank_quiz_state():
    return {
        "active": False,
        "completed": False,
        "topic": "",
        "total_questions": 5,
        "current_question_num": 0,
        "current_question": "",
        "score": 0.0,
        "difficulty": "medium",
        "history": [],          # list of {question, answer_eval, points, feedback}
        "strong_areas": [],
        "weak_areas": [],
    }


def _quiz_context_text(qs: dict) -> str:
    """Build a text context block to include in Gemini turn calls."""
    history_lines = []
    for i, h in enumerate(qs.get("history", []), 1):
        history_lines.append(
            f"Q{i}: {h.get('question','')}\n"
            f"  Eval: {h.get('answer_eval','')}, Points: {h.get('points',0)}\n"
            f"  Feedback: {h.get('feedback','')}"
        )
    history_text = "\n".join(history_lines) if history_lines else "None yet."

    return (
        f"QUIZ CONTEXT\n"
        f"Topic: {qs['topic']}\n"
        f"Question {qs['current_question_num']} of {qs['total_questions']}\n"
        f"Current difficulty: {qs['difficulty']}\n"
        f"Current score: {qs['score']}\n"
        f"Current question: {qs['current_question']}\n"
        f"Is this the final question? {'YES' if qs['current_question_num'] >= qs['total_questions'] else 'NO'}\n"
        f"Strong areas so far: {', '.join(qs.get('strong_areas', [])) or 'None'}\n"
        f"Weak areas so far: {', '.join(qs.get('weak_areas', [])) or 'None'}\n"
        f"History:\n{history_text}"
    )


# ==========================================================================
# QUIZ UI rendering helpers
# ==========================================================================

def _e(t):
    return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_quiz_left(qs: dict) -> str:
    if not qs.get("active") and not qs.get("completed"):
        return (
            '<div class="quiz-side-panel">'
            '<div class="quiz-side-label">TOPIC</div>'
            '<div class="quiz-side-value quiz-topic-idle">Waiting to start…</div>'
            '<div class="quiz-side-label" style="margin-top:18px">STATUS</div>'
            '<div class="quiz-side-value">Say: <em>"Quiz me on Python OOP for five questions."</em></div>'
            "</div>"
        )
    topic = _e(qs.get("topic", ""))
    if qs.get("completed"):
        status_html = '<span class="quiz-badge quiz-badge-done">✓ COMPLETE</span>'
    elif qs.get("active"):
        status_html = '<span class="quiz-badge quiz-badge-active">● ACTIVE</span>'
    else:
        status_html = ""
    return (
        '<div class="quiz-side-panel">'
        f'<div class="quiz-side-label">TOPIC</div>'
        f'<div class="quiz-side-value quiz-topic-active">{topic}</div>'
        f'<div class="quiz-side-label" style="margin-top:18px">STATUS</div>'
        f'<div class="quiz-side-value">{status_html}</div>'
        "</div>"
    )


def render_quiz_center(qs: dict, feedback: str = "") -> str:
    if not qs.get("active") and not qs.get("completed"):
        return (
            '<div class="quiz-idle-hero">'
            '<div class="quiz-idle-icon">🧠</div>'
            '<div class="quiz-idle-title">Voice Quiz</div>'
            '<div class="quiz-idle-sub">Test what you know. StudyVoice adapts to every answer.</div>'
            '<div class="quiz-idle-hint">"Quiz me on Python OOP for five questions."</div>'
            "</div>"
        )

    if qs.get("completed"):
        score = qs.get("score", 0)
        total = qs.get("total_questions", 5)
        strong = ", ".join(qs.get("strong_areas", [])) or "—"
        weak = ", ".join(qs.get("weak_areas", [])) or "—"
        perf = _e(qs.get("performance_summary", ""))
        next_action = _e(qs.get("recommended_next_action", ""))
        diagnosis = _e(qs.get("diagnosis", ""))
        return (
            '<div class="quiz-center-panel quiz-complete-panel">'
            '<div class="quiz-complete-header">'
            '<div>'
            '<div class="quiz-complete-title">🏁 Quiz Complete</div>'
            f'<div class="quiz-score-big">{score} <span class="quiz-score-denom">/ {total}</span></div>'
            '<div class="quiz-score-sub">Final Score</div>'
            '</div>'
            '</div>'
            '<div class="quiz-complete-body">'
            f'<div class="quiz-complete-section-full"><span class="quiz-label-sm">Performance Summary</span><div>{perf}</div></div>'
            '<div class="quiz-complete-row">'
            f'<div class="quiz-complete-section"><span class="quiz-label-sm">Strong Areas</span><div class="quiz-area-good">{_e(strong)}</div></div>'
            f'<div class="quiz-complete-section"><span class="quiz-label-sm">Needs Work</span><div class="quiz-area-weak">{_e(weak)}</div></div>'
            '</div>'
            f'<div class="quiz-complete-section-full"><span class="quiz-label-sm">AI Diagnosis</span><div>{diagnosis}</div></div>'
            f'<div class="quiz-complete-section-full"><span class="quiz-label-sm">Recommended Next Action</span><div class="quiz-next-action">{next_action}</div></div>'
            '</div>'
            "</div>"
        )

    qnum = qs.get("current_question_num", 0)
    total = qs.get("total_questions", 5)
    question = _e(qs.get("current_question", ""))
    feedback_html = (
        f'<div class="quiz-feedback">{_e(feedback)}</div>' if feedback else ""
    )
    return (
        '<div class="quiz-center-panel">'
        f'<div class="quiz-qnum">QUESTION {qnum} OF {total}</div>'
        f'<div class="quiz-question">{question}</div>'
        f'{feedback_html}'
        "</div>"
    )


def render_quiz_right(qs: dict) -> str:
    if not qs.get("active") and not qs.get("completed"):
        return (
            '<div class="quiz-side-panel">'
            '<div class="quiz-side-label">SCORE</div>'
            '<div class="quiz-side-value">—</div>'
            "</div>"
        )

    score = qs.get("score", 0)
    total = qs.get("total_questions", 5)
    diff = _e(qs.get("difficulty", "medium").upper())
    qnum = qs.get("current_question_num", 0)
    strong = qs.get("strong_areas", [])
    weak = qs.get("weak_areas", [])

    # Progress bar
    pct = int((qnum / total) * 100) if total else 0
    strong_html = "".join(f'<div class="quiz-area-tag quiz-area-strong">{_e(a)}</div>' for a in strong) or "<em>None yet</em>"
    weak_html = "".join(f'<div class="quiz-area-tag quiz-area-weak-tag">{_e(a)}</div>' for a in weak) or "<em>None yet</em>"

    diff_color = {"easy": "#34D399", "medium": "#FBBF24", "hard": "#FB7185"}.get(qs.get("difficulty", "medium"), "#94A3B8")

    return (
        '<div class="quiz-side-panel">'
        f'<div class="quiz-side-label">SCORE</div>'
        f'<div class="quiz-score-display">{score} <span class="quiz-score-denom">/ {total}</span></div>'
        f'<div class="quiz-side-label" style="margin-top:14px">DIFFICULTY</div>'
        f'<div class="quiz-diff-badge" style="color:{diff_color}">{diff}</div>'
        f'<div class="quiz-side-label" style="margin-top:14px">PROGRESS</div>'
        f'<div class="quiz-progress-bar"><div class="quiz-progress-fill" style="width:{pct}%"></div></div>'
        f'<div class="quiz-progress-label">{qnum} / {total} answered</div>'
        f'<div class="quiz-side-label" style="margin-top:14px">STRONG AREAS</div>'
        f'<div class="quiz-areas-container">{strong_html}</div>'
        f'<div class="quiz-side-label" style="margin-top:12px">WEAK AREAS</div>'
        f'<div class="quiz-areas-container">{weak_html}</div>'
        "</div>"
    )


# ==========================================================================
# QUIZ TURN HANDLERS  (Phase 4)
# ==========================================================================

def handle_quiz_start(audio_path, quiz_state):
    """
    Called when the user records their quiz request (quiz not yet active).
    Makes ONE Gemini call to parse intent + generate Question 1.
    Returns updated quiz_state, left/center/right HTML, audio path, status.
    """
    qs = _blank_quiz_state()

    if not audio_path or not os.path.isfile(audio_path):
        return (quiz_state,
                render_quiz_left(qs), render_quiz_center(qs), render_quiz_right(qs),
                None, None, "No audio detected. Please record your quiz request.")

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        if not audio_bytes:
            raise ValueError("Empty audio.")
    except Exception:
        traceback.print_exc()
        return (quiz_state,
                render_quiz_left(qs), render_quiz_center(qs), render_quiz_right(qs),
                None, None, "Could not read the recorded audio.")

    try:
        client = get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")],
            )],
            config=types.GenerateContentConfig(
                system_instruction=QUIZ_START_SYSTEM,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
                response_mime_type="application/json",
                response_schema=QuizStartResult,
            ),
        )
    except RuntimeError as e:
        traceback.print_exc()
        return (quiz_state,
                render_quiz_left(qs), render_quiz_center(qs), render_quiz_right(qs),
                None, None, str(e))
    except Exception as e:
        traceback.print_exc()
        user_msg = RATE_LIMIT_MESSAGE if _is_rate_limit_error(e) else (
            f"Gemini request failed: {type(e).__name__}. Check the terminal."
        )
        return (quiz_state,
                render_quiz_left(qs), render_quiz_center(qs), render_quiz_right(qs),
                None, None, user_msg)

    result = _extract_quiz_start(response)
    if result is None:
        return (quiz_state,
                render_quiz_left(qs), render_quiz_center(qs), render_quiz_right(qs),
                None, None, "Could not parse quiz start response. Please try again.")

    # Build new quiz state
    new_qs = _blank_quiz_state()
    new_qs["active"] = True
    new_qs["completed"] = False
    new_qs["topic"] = result.topic
    new_qs["total_questions"] = max(3, min(10, result.total_questions))
    new_qs["difficulty"] = result.difficulty or "medium"
    new_qs["current_question_num"] = 1
    new_qs["current_question"] = result.first_question
    new_qs["score"] = 0.0

    spoken = result.spoken_intro or f"Question 1: {result.first_question}"

    try:
        audio_out = _speak(spoken)
    except Exception:
        traceback.print_exc()
        audio_out = None

    return (
        new_qs,
        render_quiz_left(new_qs),
        render_quiz_center(new_qs),
        render_quiz_right(new_qs),
        audio_out,
        None,  # clear mic
        "",
    )


def handle_quiz_answer(audio_path, quiz_state):
    """
    Called when the user records an answer during an active quiz.
    Makes ONE Gemini call: evaluate answer + generate next question (or final diagnosis).
    """
    qs = dict(quiz_state) if quiz_state else _blank_quiz_state()

    if not qs.get("active"):
        return (qs,
                render_quiz_left(qs), render_quiz_center(qs), render_quiz_right(qs),
                None, None, "No active quiz. Start one by recording a quiz request.")

    if not audio_path or not os.path.isfile(audio_path):
        return (qs,
                render_quiz_left(qs), render_quiz_center(qs), render_quiz_right(qs),
                None, None, "No audio detected. Please record your answer.")

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        if not audio_bytes:
            raise ValueError("Empty audio.")
    except Exception:
        traceback.print_exc()
        return (qs,
                render_quiz_left(qs), render_quiz_center(qs), render_quiz_right(qs),
                None, None, "Could not read the recorded audio.")

    context_text = _quiz_context_text(qs)

    try:
        client = get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=context_text),
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                ],
            )],
            config=types.GenerateContentConfig(
                system_instruction=QUIZ_TURN_SYSTEM,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
                response_mime_type="application/json",
                response_schema=QuizTurnResult,
            ),
        )
    except RuntimeError as e:
        traceback.print_exc()
        return (qs,
                render_quiz_left(qs), render_quiz_center(qs), render_quiz_right(qs),
                None, None, str(e))
    except Exception as e:
        traceback.print_exc()
        user_msg = RATE_LIMIT_MESSAGE if _is_rate_limit_error(e) else (
            f"Gemini request failed: {type(e).__name__}. Check the terminal."
        )
        return (qs,
                render_quiz_left(qs), render_quiz_center(qs), render_quiz_right(qs),
                None, None, user_msg)

    turn_result = _extract_quiz_turn(response)
    if turn_result is None:
        # Parse failure: do not corrupt state
        return (qs,
                render_quiz_left(qs), render_quiz_center(qs), render_quiz_right(qs),
                None, None, "Could not parse answer evaluation. Please try your answer again.")

    # Update state (deep-copy approach to not mutate shared state)
    new_qs = dict(qs)
    new_qs["score"] = round(qs["score"] + turn_result.points_awarded, 1)
    new_qs["difficulty"] = turn_result.new_difficulty or qs["difficulty"]
    new_qs["strong_areas"] = list(turn_result.strong_areas or [])
    new_qs["weak_areas"] = list(turn_result.weak_areas or [])
    history = list(qs.get("history", []))
    history.append({
        "question": qs["current_question"],
        "answer_eval": turn_result.evaluation,
        "points": turn_result.points_awarded,
        "feedback": turn_result.feedback,
    })
    new_qs["history"] = history

    feedback_text = turn_result.feedback or ""

    if turn_result.is_final:
        new_qs["active"] = False
        new_qs["completed"] = True
        new_qs["current_question"] = ""
        new_qs["performance_summary"] = turn_result.performance_summary
        new_qs["recommended_next_action"] = turn_result.recommended_next_action
        new_qs["diagnosis"] = turn_result.performance_summary  # shown in AI Diagnosis
        spoken = turn_result.spoken_summary or (
            f"Quiz complete! You scored {new_qs['score']} out of {new_qs['total_questions']}."
        )
        try:
            audio_out = _speak(spoken)
        except Exception:
            traceback.print_exc()
            audio_out = None
        return (
            new_qs,
            render_quiz_left(new_qs),
            render_quiz_center(new_qs, feedback_text),
            render_quiz_right(new_qs),
            audio_out,
            None,
            "",
        )

    # Not final: advance to next question
    new_qs["current_question_num"] = qs["current_question_num"] + 1
    new_qs["current_question"] = turn_result.next_question or ""

    spoken_parts = []
    if feedback_text:
        spoken_parts.append(feedback_text)
    spoken_parts.append(
        f"Question {new_qs['current_question_num']}: {new_qs['current_question']}"
    )
    spoken = " ".join(spoken_parts)

    try:
        audio_out = _speak(spoken)
    except Exception:
        traceback.print_exc()
        audio_out = None

    return (
        new_qs,
        render_quiz_left(new_qs),
        render_quiz_center(new_qs, feedback_text),
        render_quiz_right(new_qs),
        audio_out,
        None,
        "",
    )


def handle_quiz_audio(audio_path, quiz_state):
    """
    Router: dispatches to start or answer handler based on quiz state.
    This is the single function wired to audio_in.stop_recording in Quiz tab.
    """
    qs = quiz_state if isinstance(quiz_state, dict) else _blank_quiz_state()
    if not qs.get("active"):
        return handle_quiz_start(audio_path, qs)
    else:
        return handle_quiz_answer(audio_path, qs)


# ==========================================================================
# GENERAL ASSISTANT  (Phase 5)
# ==========================================================================

# Max recent turns to keep in General Assistant context (bounded for quota)
_GA_MAX_CONTEXT_TURNS = 10


def _extract_general_action(response) -> Optional[GeneralAction]:
    """Extract GeneralAction from Gemini response. Returns None on parse failure."""
    parsed = getattr(response, "parsed", None)

    if isinstance(parsed, GeneralAction):
        return parsed

    if isinstance(parsed, dict):
        try:
            return GeneralAction.model_validate(parsed)
        except Exception:
            pass

    raw = getattr(response, "text", None) or ""
    try:
        return GeneralAction.model_validate_json(_strip_code_fence(raw))
    except Exception:
        pass

    print(f"[GADebug] Failed to parse GeneralAction. raw={raw[:300]}")
    return None


def _ga_context_messages(ga_history: list) -> list:
    """
    Convert stored GA history [{role, text}] into Gemini Content objects.
    Keeps last _GA_MAX_CONTEXT_TURNS turns to stay quota-efficient.
    """
    recent = ga_history[-_GA_MAX_CONTEXT_TURNS:]
    contents = []
    for turn in recent:
        role = turn.get("role", "user")
        text = turn.get("text", "")
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=text)],
            )
        )
    return contents


def _render_ga_conversation(ga_history: list) -> str:
    """Render General Assistant conversation history as styled HTML."""
    if not ga_history:
        return (
            '<div class="ga-empty-state">'
            '<div class="ga-empty-icon">💬</div>'
            '<div class="ga-empty-title">Ask anything.</div>'
            '<div class="ga-empty-sub">Type it. Say it. Learn it. Then turn it into a quiz.</div>'
            '<div class="ga-empty-hints">'
            '<div class="ga-hint-chip">Explain recursion simply.</div>'
            '<div class="ga-hint-chip">Give me three startup ideas.</div>'
            '<div class="ga-hint-chip">What causes black holes?</div>'
            '</div>'
            '</div>'
        )

    parts = []
    for turn in ga_history:
        role = turn.get("role", "user")
        text = _plan_escape(turn.get("text", ""))
        if role == "user":
            parts.append(
                f'<div class="ga-msg ga-msg-user">'
                f'<span class="ga-msg-label">You</span>'
                f'<div class="ga-msg-text">{text}</div>'
                f'</div>'
            )
        else:
            parts.append(
                f'<div class="ga-msg ga-msg-assistant">'
                f'<span class="ga-msg-label">StudyVoice</span>'
                f'<div class="ga-msg-text">{text}</div>'
                f'</div>'
            )
    return '<div class="ga-conversation">' + "".join(parts) + '</div>'


def _ga_call_gemini(contents: list, ga_history: list):
    """
    Single Gemini call for General Assistant.
    Returns (action, raw_text) or (None, error_message).
    contents: list of Gemini Content objects (history + new user turn).
    """
    try:
        client = get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=GENERAL_SYSTEM,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
                response_mime_type="application/json",
                response_schema=GeneralAction,
            ),
        )
    except RuntimeError as e:
        traceback.print_exc()
        return None, str(e)
    except Exception as e:
        traceback.print_exc()
        msg = RATE_LIMIT_MESSAGE if _is_rate_limit_error(e) else (
            f"Gemini request failed: {type(e).__name__}. Check the terminal."
        )
        return None, msg

    raw = getattr(response, "text", None)
    if not raw or not raw.strip():
        return None, "Gemini returned an empty response. Please try again."

    action = _extract_general_action(response)
    if action is None:
        print("[GADebug] raw response that failed to parse:", raw)
        return None, "StudyVoice couldn't parse its own response. Please try again."

    return action, None


def handle_ga_text(user_text: str, ga_history: list, conv_html: str, quiz_state: dict):
    """
    Handle a typed text message in the General Assistant.
    Returns: (ga_history, conv_html, ga_status, ga_audio_out, ga_text_in, quiz_state,
               quiz_left, quiz_center, quiz_right)
    """
    ga_history = list(ga_history or [])
    user_text = (user_text or "").strip()

    if not user_text:
        return (ga_history, conv_html, "Please type a message first.",
                None, "", quiz_state,
                render_quiz_left(quiz_state), render_quiz_center(quiz_state), render_quiz_right(quiz_state))

    # Build Gemini contents: prior context + new user message
    context_contents = _ga_context_messages(ga_history)
    new_user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_text)],
    )
    all_contents = context_contents + [new_user_content]

    action, err = _ga_call_gemini(all_contents, ga_history)
    if err:
        return (ga_history, conv_html, err,
                None, user_text, quiz_state,
                render_quiz_left(quiz_state), render_quiz_center(quiz_state), render_quiz_right(quiz_state))

    # Update history (store plain text for context efficiency)
    ga_history.append({"role": "user", "text": user_text})
    ga_history.append({"role": "model", "text": action.spoken_reply})

    # TTS
    audio_out = None
    try:
        audio_out = _speak(action.spoken_reply)
    except Exception:
        traceback.print_exc()

    updated_html = _render_ga_conversation(ga_history)

    # Handle quiz routing
    new_quiz_state = quiz_state
    if action.action_type == "start_quiz" and action.quiz_topic:
        topic = action.quiz_topic.strip()
        new_quiz_state, ql, qc, qr, _, _, _ = _ga_launch_quiz(topic, quiz_state)
        status = f"Your quiz on '{topic}' is ready — open the Voice Quiz tab."
        return (ga_history, updated_html, status, audio_out, "",
                new_quiz_state, ql, qc, qr)

    ql = render_quiz_left(new_quiz_state)
    qc = render_quiz_center(new_quiz_state)
    qr = render_quiz_right(new_quiz_state)
    return (ga_history, updated_html, "", audio_out, "", new_quiz_state, ql, qc, qr)


def handle_ga_voice(audio_path, ga_history: list, conv_html: str, quiz_state: dict):
    """
    Handle a voice message in the General Assistant.
    Sends audio directly to Gemini (ONE call) — no separate transcription step.
    Returns same tuple as handle_ga_text.
    """
    ga_history = list(ga_history or [])

    if not audio_path or not os.path.isfile(audio_path):
        return (ga_history, conv_html, "No audio detected. Please record a message.",
                None, None, quiz_state,
                render_quiz_left(quiz_state), render_quiz_center(quiz_state), render_quiz_right(quiz_state))

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        if not audio_bytes:
            raise ValueError("Empty audio file.")
    except Exception:
        traceback.print_exc()
        return (ga_history, conv_html, "Could not read the recorded audio. Please try again.",
                None, None, quiz_state,
                render_quiz_left(quiz_state), render_quiz_center(quiz_state), render_quiz_right(quiz_state))

    # Context as text turns + new audio turn (ONE Gemini call)
    context_contents = _ga_context_messages(ga_history)
    new_user_content = types.Content(
        role="user",
        parts=[types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")],
    )
    all_contents = context_contents + [new_user_content]

    action, err = _ga_call_gemini(all_contents, ga_history)
    if err:
        return (ga_history, conv_html, err,
                None, None, quiz_state,
                render_quiz_left(quiz_state), render_quiz_center(quiz_state), render_quiz_right(quiz_state))

    # Store voice turn as "(voice)" placeholder in history text
    ga_history.append({"role": "user", "text": "(voice message)"})
    ga_history.append({"role": "model", "text": action.spoken_reply})

    audio_out = None
    try:
        audio_out = _speak(action.spoken_reply)
    except Exception:
        traceback.print_exc()

    updated_html = _render_ga_conversation(ga_history)

    new_quiz_state = quiz_state
    if action.action_type == "start_quiz" and action.quiz_topic:
        topic = action.quiz_topic.strip()
        new_quiz_state, ql, qc, qr, _, _, _ = _ga_launch_quiz(topic, quiz_state)
        status = f"Your quiz on '{topic}' is ready — open the Voice Quiz tab."
        return (ga_history, updated_html, status, audio_out, None,
                new_quiz_state, ql, qc, qr)

    ql = render_quiz_left(new_quiz_state)
    qc = render_quiz_center(new_quiz_state)
    qr = render_quiz_right(new_quiz_state)
    return (ga_history, updated_html, "", audio_out, None, new_quiz_state, ql, qc, qr)


def _ga_launch_quiz(topic: str, current_quiz_state: dict):
    """
    Reuse the existing quiz engine to start a quiz on a given topic.
    Makes ONE Gemini call (same as handle_quiz_start) using a text prompt
    so we don't need a recording. Returns the same 7-tuple as handle_quiz_start.
    """
    qs = _blank_quiz_state()
    prompt_text = f"Quiz me on {topic} for five questions at medium difficulty."

    try:
        client = get_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt_text)],
            )],
            config=types.GenerateContentConfig(
                system_instruction=QUIZ_START_SYSTEM,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
                response_mime_type="application/json",
                response_schema=QuizStartResult,
            ),
        )
    except Exception as e:
        traceback.print_exc()
        return (current_quiz_state,
                render_quiz_left(qs), render_quiz_center(qs), render_quiz_right(qs),
                None, None, f"Could not start quiz: {type(e).__name__}")

    result = _extract_quiz_start(response)
    if result is None:
        return (current_quiz_state,
                render_quiz_left(qs), render_quiz_center(qs), render_quiz_right(qs),
                None, None, "Could not parse quiz start response.")

    new_qs = _blank_quiz_state()
    new_qs["active"] = True
    new_qs["completed"] = False
    new_qs["topic"] = result.topic
    new_qs["total_questions"] = max(3, min(10, result.total_questions))
    new_qs["difficulty"] = result.difficulty or "medium"
    new_qs["current_question_num"] = 1
    new_qs["current_question"] = result.first_question
    new_qs["score"] = 0.0

    spoken = result.spoken_intro or f"Question 1: {result.first_question}"
    audio_out = None
    try:
        audio_out = _speak(spoken)
    except Exception:
        traceback.print_exc()

    return (
        new_qs,
        render_quiz_left(new_qs),
        render_quiz_center(new_qs),
        render_quiz_right(new_qs),
        audio_out,
        None,
        "",
    )


# ==========================================================================
# UI – CSS (extends Phase 3 styles)
# ==========================================================================

CUSTOM_CSS = """
/* ===================================================================
   StudyVoice Final – Midnight + Electric Cyan
   Palette: #070B14 bg | #0D1424 surface | #131D30 elevated
   Accent:  #22D3EE cyan | #60A5FA blue
   Text:    #F8FAFC main | #94A3B8 muted
   Border:  #1E293B subtle
   Status:  #34D399 success | #FBBF24 warning | #FB7185 error
   =================================================================== */

:root {
    --sv-bg:          #070B14;
    --sv-surface:     #0D1424;
    --sv-elevated:    #131D30;
    --sv-accent:      #22D3EE;
    --sv-accent2:     #60A5FA;
    --sv-text:        #F8FAFC;
    --sv-text-sec:    #94A3B8;
    --sv-success:     #34D399;
    --sv-warning:     #FBBF24;
    --sv-error:       #FB7185;
    --sv-border:      #1E293B;
    --sv-border-soft: rgba(255, 255, 255, 0.06);
}

/* ── Base ── */
.gradio-container {
    background: var(--sv-bg) !important;
    max-width: 1440px !important;
    color: var(--sv-text) !important;
    min-height: 100vh;
    padding: 0 8px !important;
}
.gradio-container, .gradio-container * {
    font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
    box-sizing: border-box;
}
/* Kill default Gradio white backgrounds everywhere */
.gradio-container .wrap,
.gradio-container .form,
.gradio-container .block,
.gradio-container > div,
.gradio-container .gap,
.gradio-container .tabs,
.gradio-container .tabitem {
    background: transparent !important;
}
footer { display: none !important; }

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(34,211,238,0.25); border-radius: 4px; }

/* ── Header ── */
#app_header { padding: 10px 4px 0 4px; }
.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.header-left { display: flex; align-items: center; gap: 10px; }
.logo-icon { font-size: 22px; line-height: 1; }
.app-title {
    font-size: 18px;
    font-weight: 700;
    color: var(--sv-text);
    letter-spacing: -0.3px;
}
.app-subtitle-small {
    font-size: 11px;
    color: var(--sv-text-sec);
    font-weight: 400;
    margin-top: 1px;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(52, 211, 153, 0.07);
    border: 1px solid rgba(52, 211, 153, 0.22);
    color: var(--sv-success);
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 999px;
    white-space: nowrap;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--sv-success);
    flex-shrink: 0;
}
/* Remove verbose tagline – wasted vertical space */
.header-tagline { display: none; }

/* ── Tab nav ── */
.tab-nav button {
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px !important;
    padding: 7px 18px !important;
    color: var(--sv-text-sec) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
}
.tab-nav button.selected {
    color: var(--sv-accent) !important;
    border-bottom: 2px solid var(--sv-accent) !important;
    background: transparent !important;
}
.tab-nav {
    border-bottom: 1px solid var(--sv-border) !important;
    background: transparent !important;
    margin-bottom: 10px !important;
    margin-top: 6px !important;
}

/* ── Panel cards ── */
.panel-card {
    background: var(--sv-surface) !important;
    border: 1px solid var(--sv-border) !important;
    border-radius: 10px !important;
    padding: 12px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.4) !important;
}
.card-label {
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 1.8px;
    color: var(--sv-accent);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
    text-transform: uppercase;
}
.card-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--sv-border);
}

/* ── Native Gradio component overrides ── */
.gradio-container .audio-component,
.gradio-container [data-testid="audio"] {
    background: var(--sv-elevated) !important;
    border: 1px solid var(--sv-border) !important;
    border-radius: 8px !important;
}
/* Compact audio output – no giant player */
.audio-output { margin-bottom: 6px !important; }
.audio-output > div { padding: 6px 8px !important; }

/* All textareas */
.gradio-container textarea {
    background: var(--sv-elevated) !important;
    border: 1px solid var(--sv-border) !important;
    border-radius: 8px !important;
    color: var(--sv-text) !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
    padding: 8px 12px !important;
}
.gradio-container textarea::placeholder { color: var(--sv-text-sec) !important; opacity: 0.6; }
/* Labels */
.gradio-container label span,
.gradio-container .label-wrap span {
    color: var(--sv-text-sec) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
}

/* ── Mic inputs – compact, no extra chrome ── */
#mic_input, #quiz_mic_input {
    border-radius: 8px !important;
    background: var(--sv-elevated) !important;
}
.mic-instruction {
    text-align: center;
    color: var(--sv-text-sec);
    font-size: 12px;
    margin: 5px 0 4px 0;
    line-height: 1.5;
}

/* Status textbox – hide when empty, compact when visible */
.status-textbox textarea {
    background: var(--sv-elevated) !important;
    border: 1px solid var(--sv-border) !important;
    border-radius: 8px !important;
    color: var(--sv-text-sec) !important;
    font-size: 12.5px !important;
    min-height: 36px !important;
    max-height: 52px !important;
}

/* ── STUDY AGENT LAYOUT ── */

/* Voice control bar on top */
#sa_top_bar {
    margin-bottom: 8px;
}
#sa_top_bar .panel-card {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 14px;
    padding: 10px 16px !important;
    flex-wrap: wrap;
}
#sa_top_bar .card-label { display: none; }
#sa_top_bar #mic_input { flex: 0 0 auto; min-width: 220px; max-width: 300px; }
#sa_top_bar .mic-instruction { flex: 1; text-align: left; margin: 0; }
#sa_top_bar .status-textbox { flex: 1; min-width: 160px; }
#sa_top_bar .status-textbox textarea { min-height: 32px !important; max-height: 40px !important; }

/* Conversation + Plan main row */
.conversation-card { border-color: var(--sv-border) !important; }
.plan-card { border-color: rgba(96,165,250,0.2) !important; }

.transcript-textbox textarea {
    background: var(--sv-elevated) !important;
    border: 1px solid var(--sv-border) !important;
    border-radius: 8px !important;
    color: var(--sv-text) !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    padding: 10px 12px !important;
    min-height: 280px !important;
    max-height: 420px !important;
}

/* Plan panel – dense timeline */
#plan_panel { max-height: 440px; overflow-y: auto; padding-right: 2px; }
#plan_panel .plan-title {
    font-size: 14px;
    font-weight: 700;
    color: var(--sv-text);
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--sv-border);
}
#plan_panel .plan-timeline { display: flex; flex-direction: column; gap: 5px; }
#plan_panel .plan-block {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    background: var(--sv-elevated);
    border: 1px solid var(--sv-border);
    border-left: 3px solid var(--sv-accent);
    border-radius: 7px;
    padding: 7px 11px;
}
#plan_panel .plan-block.plan-break {
    border-left: 3px dashed rgba(148,163,184,0.3);
    background: rgba(255,255,255,0.015);
    opacity: 0.75;
}
#plan_panel .plan-time {
    flex-shrink: 0;
    min-width: 82px;
    font-size: 11px;
    font-weight: 700;
    color: var(--sv-accent2);
    background: rgba(96,165,250,0.08);
    border: 1px solid rgba(96,165,250,0.18);
    border-radius: 5px;
    padding: 2px 6px;
    text-align: center;
}
#plan_panel .plan-block.plan-break .plan-time {
    color: var(--sv-text-sec);
    background: rgba(255,255,255,0.03);
    border-color: var(--sv-border);
}
#plan_panel .plan-topic { font-weight: 600; color: var(--sv-text); font-size: 13px; margin-bottom: 1px; }
#plan_panel .plan-goal { font-size: 11.5px; color: var(--sv-text-sec); line-height: 1.4; }
#plan_panel .plan-placeholder { text-align: center; padding: 24px 10px; color: var(--sv-text-sec); }
#plan_panel .plan-placeholder-icon { font-size: 24px; margin-bottom: 7px; }
#plan_panel .plan-placeholder-example {
    margin-top: 8px;
    font-size: 11.5px;
    font-style: italic;
    color: var(--sv-accent);
    background: rgba(34,211,238,0.05);
    border: 1px solid rgba(34,211,238,0.13);
    border-radius: 7px;
    padding: 7px 11px;
    opacity: 0.85;
}

/* ========== VOICE QUIZ STYLES ========== */

/* Quiz idle hero – centered, no wasted whitespace */
.quiz-idle-hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 32px 20px 24px;
    gap: 10px;
    text-align: center;
}
.quiz-idle-icon { font-size: 36px; line-height: 1; margin-bottom: 4px; }
.quiz-idle-title {
    font-size: 22px;
    font-weight: 700;
    color: var(--sv-text);
    letter-spacing: -0.3px;
}
.quiz-idle-sub {
    font-size: 13px;
    color: var(--sv-text-sec);
    max-width: 320px;
    line-height: 1.55;
}
.quiz-idle-hint {
    font-size: 12px;
    font-style: italic;
    color: var(--sv-accent);
    background: rgba(34,211,238,0.05);
    border: 1px solid rgba(34,211,238,0.15);
    border-radius: 7px;
    padding: 6px 14px;
    margin-top: 4px;
}

/* Quiz side panel – right sidebar */
.quiz-side-panel { padding: 0; }
.quiz-side-label {
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 1.6px;
    color: var(--sv-accent);
    margin-bottom: 3px;
    margin-top: 10px;
    text-transform: uppercase;
}
.quiz-side-label:first-child { margin-top: 0; }
.quiz-side-value {
    font-size: 13px;
    color: var(--sv-text);
    line-height: 1.5;
}
.quiz-topic-idle { color: var(--sv-text-sec); font-style: italic; }
.quiz-topic-active { color: var(--sv-text); font-weight: 600; font-size: 13px; }

.quiz-badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.quiz-badge-active {
    background: rgba(34,211,238,0.1);
    border: 1px solid rgba(34,211,238,0.35);
    color: var(--sv-accent);
}
.quiz-badge-done {
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.3);
    color: var(--sv-success);
}

/* Quiz center panel – question hero */
.quiz-center-panel {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.quiz-idle-msg {
    text-align: center;
    color: var(--sv-text-sec);
    font-size: 14px;
    padding: 32px 20px;
    line-height: 1.6;
}
.quiz-qnum {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--sv-accent);
    text-transform: uppercase;
}
.quiz-question {
    font-size: 22px;
    font-weight: 600;
    color: var(--sv-text);
    line-height: 1.55;
    background: rgba(34,211,238,0.04);
    border: 1px solid rgba(34,211,238,0.14);
    border-left: 3px solid var(--sv-accent);
    border-radius: 0 9px 9px 0;
    padding: 14px 16px;
}
.quiz-feedback {
    font-size: 13.5px;
    color: var(--sv-text);
    background: var(--sv-elevated);
    border: 1px solid var(--sv-border);
    border-radius: 8px;
    padding: 9px 13px;
    line-height: 1.6;
}

/* Quiz COMPLETE card – compact, one viewport */
.quiz-complete-panel { gap: 8px; }
.quiz-complete-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--sv-border);
    margin-bottom: 4px;
}
.quiz-complete-title {
    font-size: 12px;
    font-weight: 700;
    color: var(--sv-success);
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.quiz-score-big {
    font-size: 36px;
    font-weight: 800;
    color: var(--sv-text);
    line-height: 1;
}
.quiz-score-sub {
    font-size: 11px;
    color: var(--sv-text-sec);
    margin-top: 1px;
}
.quiz-complete-body {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.quiz-complete-row {
    display: flex;
    gap: 6px;
}
.quiz-complete-section {
    display: flex;
    flex-direction: column;
    gap: 3px;
    font-size: 13px;
    color: var(--sv-text);
    background: var(--sv-elevated);
    border: 1px solid var(--sv-border);
    border-radius: 8px;
    padding: 8px 11px;
    flex: 1;
}
.quiz-complete-section-full {
    display: flex;
    flex-direction: column;
    gap: 3px;
    font-size: 13px;
    color: var(--sv-text);
    background: var(--sv-elevated);
    border: 1px solid var(--sv-border);
    border-radius: 8px;
    padding: 8px 11px;
}
.quiz-label-sm {
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: var(--sv-accent);
    margin-bottom: 2px;
    text-transform: uppercase;
}
.quiz-area-good { color: var(--sv-success); font-weight: 500; }
.quiz-area-weak { color: var(--sv-error); font-weight: 500; }
.quiz-next-action {
    color: var(--sv-accent2);
    font-style: italic;
    font-size: 13px;
    line-height: 1.5;
}

/* Quiz score display in sidebar */
.quiz-score-display {
    font-size: 30px;
    font-weight: 800;
    color: var(--sv-text);
    line-height: 1;
    margin: 2px 0 3px 0;
}
.quiz-score-denom {
    font-size: 15px;
    font-weight: 400;
    color: var(--sv-text-sec);
}
.quiz-diff-badge {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.quiz-progress-bar {
    width: 100%;
    height: 4px;
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    overflow: hidden;
    margin: 4px 0 2px 0;
}
.quiz-progress-fill {
    height: 100%;
    background: var(--sv-accent);
    border-radius: 999px;
    transition: width 0.4s ease;
}
.quiz-progress-label {
    font-size: 11px;
    color: var(--sv-text-sec);
}
.quiz-areas-container {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 2px;
}
.quiz-area-tag {
    padding: 2px 7px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
}
.quiz-area-strong {
    background: rgba(52, 211, 153, 0.08);
    border: 1px solid rgba(52, 211, 153, 0.25);
    color: var(--sv-success);
}
.quiz-area-weak-tag {
    background: rgba(251, 113, 133, 0.08);
    border: 1px solid rgba(251, 113, 133, 0.25);
    color: var(--sv-error);
}

/* ========== GENERAL ASSISTANT STYLES (Phase 5) ========== */

/* Conversation panel */
.ga-conversation {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 4px 2px;
}
.ga-msg {
    display: flex;
    flex-direction: column;
    gap: 3px;
    max-width: 96%;
}
.ga-msg-user { align-self: flex-end; align-items: flex-end; }
.ga-msg-assistant { align-self: flex-start; align-items: flex-start; }
.ga-msg-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--sv-text-sec);
    margin-bottom: 2px;
}
.ga-msg-text {
    font-size: 14px;
    line-height: 1.65;
    padding: 10px 14px;
    border-radius: 10px;
    color: var(--sv-text);
    white-space: pre-wrap;
}
.ga-msg-user .ga-msg-text {
    background: rgba(34,211,238,0.08);
    border: 1px solid rgba(34,211,238,0.18);
    border-radius: 10px 10px 2px 10px;
}
.ga-msg-assistant .ga-msg-text {
    background: var(--sv-elevated);
    border: 1px solid var(--sv-border);
    border-radius: 2px 10px 10px 10px;
}

/* Empty state */
.ga-empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 36px 20px 28px;
    gap: 8px;
    text-align: center;
}
.ga-empty-icon { font-size: 34px; line-height: 1; margin-bottom: 4px; }
.ga-empty-title {
    font-size: 22px;
    font-weight: 700;
    color: var(--sv-text);
    letter-spacing: -0.3px;
}
.ga-empty-sub {
    font-size: 13px;
    color: var(--sv-text-sec);
    max-width: 340px;
    line-height: 1.55;
    margin-bottom: 4px;
}
.ga-empty-hints {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    justify-content: center;
    margin-top: 4px;
}
.ga-hint-chip {
    font-size: 12px;
    font-style: italic;
    color: var(--sv-accent);
    background: rgba(34,211,238,0.05);
    border: 1px solid rgba(34,211,238,0.15);
    border-radius: 999px;
    padding: 4px 12px;
}

/* Text input row */
.ga-text-input textarea {
    background: var(--sv-elevated) !important;
    border: 1px solid var(--sv-border) !important;
    border-radius: 8px !important;
    color: var(--sv-text) !important;
    font-size: 14px !important;
    resize: none !important;
    min-height: 44px !important;
    max-height: 88px !important;
}
.ga-send-btn {
    background: rgba(34,211,238,0.12) !important;
    border: 1px solid rgba(34,211,238,0.3) !important;
    color: var(--sv-accent) !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    min-height: 44px !important;
}
.ga-send-btn:hover {
    background: rgba(34,211,238,0.2) !important;
}

/* Voice side panel */
.ga-voice-card {
    border-color: rgba(34,211,238,0.18) !important;
}
.ga-context-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--sv-accent);
    margin-bottom: 4px;
    margin-top: 12px;
}
.ga-context-topic {
    font-size: 13px;
    font-weight: 600;
    color: var(--sv-text);
    background: var(--sv-elevated);
    border: 1px solid var(--sv-border);
    border-left: 3px solid var(--sv-accent);
    border-radius: 0 7px 7px 0;
    padding: 6px 10px;
    margin-top: 2px;
}
.ga-context-idle {
    font-size: 12px;
    font-style: italic;
    color: var(--sv-text-sec);
}
.ga-quiz-ready-banner {
    background: rgba(34,211,238,0.07);
    border: 1px solid rgba(34,211,238,0.22);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12.5px;
    color: var(--sv-accent);
    margin-top: 8px;
    line-height: 1.5;
}
"""


# ==========================================================================
# UI BUILD
# ==========================================================================

_blank_qs = _blank_quiz_state()

with gr.Blocks(title="StudyVoice", css=CUSTOM_CSS) as demo:

    # ------ Compact Header ------
    with gr.Row(elem_id="app_header"):
        gr.HTML(
            '<div class="header-content">'
            '<div class="header-left">'
            '<div class="logo-icon">🎙️</div>'
            "<div>"
            '<div class="app-title">StudyVoice</div>'
            '<div class="app-subtitle-small">Voice-first AI learning copilot</div>'
            "</div>"
            "</div>"
            '<div class="status-badge">'
            '<span class="status-dot"></span> AI Online'
            "</div>"
            "</div>"
        )

    # ------ Top-level tabs ------
    with gr.Tabs(elem_classes=["tab-nav"]):

        # ==================== TAB 1: GENERAL ASSISTANT (Phase 5) ====================
        with gr.Tab("💬 General Assistant"):

            ga_history_state = gr.State([])

            with gr.Row():
                # LEFT 70%: conversation + text input
                with gr.Column(scale=70, elem_classes=["panel-card"]):
                    gr.HTML('<div class="card-label">Conversation</div>')
                    ga_conv_html = gr.HTML(
                        value=_render_ga_conversation([]),
                        elem_id="ga_conv",
                    )
                    with gr.Row(equal_height=True):
                        ga_text_in = gr.Textbox(
                            label="",
                            placeholder="Ask anything — or say 'Quiz me on that' after learning something.",
                            lines=1,
                            elem_classes=["ga-text-input"],
                            show_label=False,
                            scale=5,
                        )
                        ga_send_btn = gr.Button(
                            "Send",
                            elem_classes=["ga-send-btn"],
                            scale=1,
                        )
                    ga_audio_out = gr.Audio(
                        label="🔊 StudyVoice",
                        autoplay=True,
                        interactive=False,
                        elem_classes=["audio-output"],
                    )

                # RIGHT 30%: voice + context indicator
                with gr.Column(scale=30, elem_classes=["panel-card", "ga-voice-card"]):
                    gr.HTML('<div class="card-label">Voice &amp; Context</div>')
                    ga_voice_in = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        format="wav",
                        label="",
                        elem_id="ga_mic_input",
                    )
                    gr.HTML(
                        '<div class="mic-instruction">'
                        "Record your question or say 'Quiz me on that'.<br>"
                        "Auto-processes when you stop."
                        "</div>"
                    )
                    ga_status = gr.Textbox(
                        label="Status",
                        value="",
                        interactive=False,
                        lines=1,
                        elem_classes=["status-textbox"],
                    )
                    ga_context_display = gr.HTML(
                        value=(
                            '<div class="ga-context-label">CURRENT TOPIC</div>'
                            '<div class="ga-context-idle">No topic yet — start a conversation.</div>'
                        ),
                    )

            # Shared quiz state with Voice Quiz tab (same gr.State object, wired below)
            # We declare it here as a placeholder; the actual shared reference is set
            # after the quiz tab wires its own quiz_state. We use a separate State here
            # and pass it through handler outputs to keep Quiz tab in sync.
            ga_quiz_state_ref = gr.State(_blank_quiz_state())
            ga_quiz_left_ref = gr.HTML(visible=False)
            ga_quiz_center_ref = gr.HTML(visible=False)
            ga_quiz_right_ref = gr.HTML(visible=False)

            # GA text submit outputs:
            # ga_history_state, ga_conv_html, ga_status, ga_audio_out, ga_text_in,
            # ga_quiz_state_ref, ga_quiz_left_ref, ga_quiz_center_ref, ga_quiz_right_ref
            _ga_outputs = [
                ga_history_state,
                ga_conv_html,
                ga_status,
                ga_audio_out,
                ga_text_in,
                ga_quiz_state_ref,
                ga_quiz_left_ref,
                ga_quiz_center_ref,
                ga_quiz_right_ref,
            ]

            ga_send_btn.click(
                fn=handle_ga_text,
                inputs=[ga_text_in, ga_history_state, ga_conv_html, ga_quiz_state_ref],
                outputs=_ga_outputs,
            )
            ga_text_in.submit(
                fn=handle_ga_text,
                inputs=[ga_text_in, ga_history_state, ga_conv_html, ga_quiz_state_ref],
                outputs=_ga_outputs,
            )

            # GA voice outputs same list but ga_text_in replaced with ga_voice_in clear
            _ga_voice_outputs = [
                ga_history_state,
                ga_conv_html,
                ga_status,
                ga_audio_out,
                ga_voice_in,
                ga_quiz_state_ref,
                ga_quiz_left_ref,
                ga_quiz_center_ref,
                ga_quiz_right_ref,
            ]

            ga_voice_in.stop_recording(
                fn=handle_ga_voice,
                inputs=[ga_voice_in, ga_history_state, ga_conv_html, ga_quiz_state_ref],
                outputs=_ga_voice_outputs,
            )

        # ==================== TAB 2: STUDY AGENT (Phase 3 – preserved) ====================
        with gr.Tab("📅 Study Agent"):

            history_state = gr.State([])
            plan_state = gr.State(None)

            # TOP CONTROL BAR – compact voice strip
            with gr.Row(elem_id="sa_top_bar"):
                with gr.Column(elem_classes=["panel-card"]):
                    gr.HTML('<div class="card-label">Voice Copilot</div>')
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=3, min_width=200):
                            audio_in = gr.Audio(
                                sources=["microphone"],
                                type="filepath",
                                format="wav",
                                label="",
                                elem_id="mic_input",
                            )
                            gr.HTML(
                                '<div class="mic-instruction">'
                                "Record your request — auto-processes when you stop."
                                "</div>"
                            )
                        with gr.Column(scale=5, min_width=200):
                            error_box = gr.Textbox(
                                label="Status",
                                value="",
                                interactive=False,
                                lines=1,
                                elem_classes=["status-textbox"],
                            )
                            audio_out = gr.Audio(
                                label="🔊 StudyVoice",
                                autoplay=True,
                                interactive=False,
                                elem_classes=["audio-output"],
                            )

            # MAIN CONTENT – Conversation (42%) + Plan (58%)
            with gr.Row():
                with gr.Column(scale=42, elem_classes=["panel-card", "conversation-card"]):
                    gr.HTML('<div class="card-label">Conversation</div>')
                    transcript_box = gr.Textbox(
                        label="",
                        value="",
                        interactive=False,
                        lines=16,
                        elem_id="transcript_box",
                        elem_classes=["transcript-textbox"],
                    )

                with gr.Column(scale=58, elem_classes=["panel-card", "plan-card"]):
                    gr.HTML('<div class="card-label">Today\'s Study Plan</div>')
                    plan_panel = gr.Markdown(
                        value=PLAN_PLACEHOLDER,
                        elem_id="plan_panel",
                    )

            turn_outputs = [
                audio_out,
                audio_in,
                history_state,
                transcript_box,
                error_box,
                plan_state,
                plan_panel,
            ]

            audio_in.stop_recording(
                fn=handle_turn,
                inputs=[audio_in, history_state, transcript_box, plan_state],
                outputs=turn_outputs,
            )

        # ==================== TAB 3: VOICE QUIZ (Phase 4 – preserved) ====================
        with gr.Tab("🧠 Voice Quiz"):

            quiz_state = gr.State(_blank_quiz_state())

            with gr.Row():
                # LEFT MAIN 68%: question hero + audio + feedback
                with gr.Column(scale=68, elem_classes=["panel-card"]):
                    gr.HTML('<div class="card-label">Question</div>')
                    quiz_audio_out = gr.Audio(
                        label="🔊 StudyVoice",
                        autoplay=True,
                        interactive=False,
                        elem_classes=["audio-output"],
                    )
                    quiz_center = gr.HTML(
                        value=render_quiz_center(_blank_qs),
                    )

                # RIGHT SIDEBAR 32%: mic + stats
                with gr.Column(scale=32, elem_classes=["panel-card", "quiz-card"]):
                    gr.HTML('<div class="card-label">Answer &amp; Progress</div>')
                    quiz_audio_in = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        format="wav",
                        label="",
                        elem_id="quiz_mic_input",
                    )
                    gr.HTML(
                        '<div class="mic-instruction">'
                        "Speak your request or answer — auto-processes when you stop."
                        "</div>"
                    )
                    quiz_status = gr.Textbox(
                        label="Status",
                        value="",
                        interactive=False,
                        lines=1,
                        elem_classes=["status-textbox"],
                    )
                    quiz_left = gr.HTML(
                        value=render_quiz_left(_blank_qs),
                    )
                    quiz_right = gr.HTML(
                        value=render_quiz_right(_blank_qs),
                    )

            quiz_outputs = [
                quiz_state,
                quiz_left,
                quiz_center,
                quiz_right,
                quiz_audio_out,
                quiz_audio_in,
                quiz_status,
            ]

            quiz_audio_in.stop_recording(
                fn=handle_quiz_audio,
                inputs=[quiz_audio_in, quiz_state],
                outputs=quiz_outputs,
            )

            # Cross-tab sync: when GA launches a quiz, propagate state + rendered HTML
            # into the Voice Quiz tab components so the tab is ready immediately.
            def _sync_quiz_from_ga(ga_qs, ga_ql, ga_qc, ga_qr):
                return ga_qs, ga_ql, ga_qc, ga_qr

            ga_quiz_state_ref.change(
                fn=_sync_quiz_from_ga,
                inputs=[ga_quiz_state_ref, ga_quiz_left_ref, ga_quiz_center_ref, ga_quiz_right_ref],
                outputs=[quiz_state, quiz_left, quiz_center, quiz_right],
            )

            def _render_ga_context(ga_history):
                """Show the last non-voice user message as current topic."""
                if not ga_history:
                    return (
                        '<div class="ga-context-label">CURRENT TOPIC</div>'
                        '<div class="ga-context-idle">No topic yet — start a conversation.</div>'
                    )
                # Find last user text that isn't a voice placeholder
                last_topic = None
                for turn in reversed(ga_history):
                    if turn.get("role") == "user" and turn.get("text", "") != "(voice message)":
                        last_topic = turn["text"]
                        break
                if not last_topic:
                    last_topic = "(voice)"
                escaped = _plan_escape(last_topic[:120] + ("…" if len(last_topic) > 120 else ""))
                return (
                    '<div class="ga-context-label">LAST MESSAGE</div>'
                    f'<div class="ga-context-topic">{escaped}</div>'
                    '<div class="ga-context-label" style="margin-top:10px">QUIZ ME ON THAT</div>'
                    '<div class="ga-context-idle">Say or type "Quiz me on that" to start a quiz on the current topic.</div>'
                )

            ga_history_state.change(
                fn=_render_ga_context,
                inputs=[ga_history_state],
                outputs=[ga_context_display],
            )


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY", "").strip():
        print(
            "\n[StudyVoice] WARNING: GEMINI_API_KEY is not set in this terminal.\n"
            "Set it before using the app:\n"
            '  $env:GEMINI_API_KEY="your-key-here"\n'
            "The app will still launch, but every request will show an error "
            "until the key is set and the app is restarted.\n"
        )
    demo.launch()