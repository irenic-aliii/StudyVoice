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
    '<div class="plan-empty-state fade-in">'
    '<div class="plan-empty-icon">✧</div>'
    '<div class="plan-placeholder-title">Ready for your schedule</div>'
    '<div class="plan-placeholder-copy">'
    'Describe your subjects, how much time you have, and your learning goals to generate an optimized study plan.'
    '</div>'
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

    html = [f'<div class="plan-title slide-up">{_plan_escape(title)}</div>']
    html.append('<div class="plan-timeline slide-up">')
    for b in blocks:
        time_range = _plan_escape(b.get("time_range", ""))
        topic = _plan_escape(b.get("topic", ""))
        goal = _plan_escape(b.get("goal", ""))
        is_break = "break" in topic.lower()
        block_class = "plan-block plan-break" if is_break else "plan-block"
        goal_html = f'<div class="plan-goal">{goal}</div>' if goal else ""
        
        marker_class = "timeline-marker-break" if is_break else "timeline-marker-study"
        
        html.append(
            f'<div class="{block_class}">'
            f'<div class="plan-time-col">'
            f'<div class="plan-time">{time_range}</div>'
            f'<div class="timeline-marker {marker_class}"></div>'
            f'</div>'
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
    def _meta_item(label, value):
        return (
            '<div class="quiz-meta-item">'
            f'<div class="quiz-meta-label">{_e(label)}</div>'
            f'<div class="quiz-meta-value">{_e(value)}</div>'
            '</div>'
        )

    if not qs.get("active") and not qs.get("completed"):
        return (
            '<div class="quiz-meta-row fade-in">'
            + _meta_item("Topic", "Waiting to start")
            + _meta_item("Question", "0 of 5")
            + '</div>'
        )
    topic = _e(qs.get("topic", ""))
    qnum = qs.get("current_question_num", 0)
    total = qs.get("total_questions", 5)
    return (
        '<div class="quiz-meta-row slide-up">'
        f'<div class="quiz-meta-item"><div class="quiz-meta-label">Topic</div><div class="quiz-meta-value">{topic}</div></div>'
        + _meta_item("Question", f"{qnum} of {total}")
        + '</div>'
    )


def render_quiz_center(qs: dict, feedback: str = "") -> str:
    if not qs.get("active") and not qs.get("completed"):
        return (
            '<div class="quiz-idle-hero fade-in">'
            '<div class="quiz-idle-icon">✦</div>'
            '<div class="quiz-idle-title">Voice Quiz</div>'
            '<div class="quiz-idle-sub">Test your knowledge. Speak to begin.</div>'
            '<div class="quiz-idle-hint">"Quiz me on Python OOP for five questions."</div>'
            "</div>"
        )

    if qs.get("completed"):
        score = qs.get("score", 0)
        total = qs.get("total_questions", 5)
        strong = ", ".join(qs.get("strong_areas", [])) or "None yet"
        weak = ", ".join(qs.get("weak_areas", [])) or "None yet"
        perf = _e(qs.get("performance_summary", ""))
        next_action = _e(qs.get("recommended_next_action", ""))
        diagnosis = _e(qs.get("diagnosis", ""))
        return (
            '<div class="quiz-complete-panel slide-up">'
            '<div class="quiz-complete-header">'
            '<div class="quiz-complete-title">Quiz Complete</div>'
            f'<div class="quiz-score-big">{score} <span class="quiz-score-denom">/ {total}</span></div>'
            '</div>'
            '<div class="quiz-complete-body">'
            f'<div class="quiz-complete-section"><div class="quiz-label-sm">Performance summary</div><div class="quiz-section-content">{perf}</div></div>'
            '<div class="quiz-complete-row">'
            f'<div class="quiz-complete-section"><div class="quiz-label-sm">Strong areas</div><div class="quiz-section-content">{_e(strong)}</div></div>'
            f'<div class="quiz-complete-section"><div class="quiz-label-sm">Needs work</div><div class="quiz-section-content">{_e(weak)}</div></div>'
            '</div>'
            f'<div class="quiz-complete-section"><div class="quiz-label-sm">Diagnosis</div><div class="quiz-section-content">{diagnosis}</div></div>'
            f'<div class="quiz-complete-section quiz-action-card"><div class="quiz-label-sm">Recommended Next Action</div><div class="quiz-next-action">{next_action}</div></div>'
            '</div>'
            "</div>"
        )

    qnum = qs.get("current_question_num", 0)
    total = qs.get("total_questions", 5)
    question = _e(qs.get("current_question", ""))
    feedback_html = (
        f'<div class="quiz-feedback slide-up">{_e(feedback)}</div>' if feedback else ""
    )
    return (
        '<div class="quiz-center-panel slide-up">'
        f'<div class="quiz-qnum">Question {qnum} of {total}</div>'
        f'<div class="quiz-question">{question}</div>'
        f'{feedback_html}'
        "</div>"
    )


def render_quiz_right(qs: dict) -> str:
    def _meta_item(label, value):
        return (
            '<div class="quiz-meta-item">'
            f'<div class="quiz-meta-label">{_e(label)}</div>'
            f'<div class="quiz-meta-value">{_e(value)}</div>'
            '</div>'
        )
    
    if not qs.get("active") and not qs.get("completed"):
        return (
            '<div class="quiz-meta-row fade-in">'
            + _meta_item("Score", "0 / 5")
            + _meta_item("Difficulty", "Medium")
            + '</div>'
        )

    score = qs.get("score", 0)
    total = qs.get("total_questions", 5)
    diff = _e(str(qs.get("difficulty", "medium")).title())
    return (
        '<div class="quiz-meta-row slide-up">'
        + _meta_item("Score", f"{score} / {total}")
        + _meta_item("Difficulty", diff)
        + '</div>'
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
            '<div class="ga-empty-state fade-in">'
            '<div class="ga-empty-title">How can I help you learn?</div>'
            '<div class="ga-empty-sub">Type or speak your questions. Turn any concept into a voice quiz when you are ready.</div>'
            '<div class="ga-empty-hints">'
            '<div class="ga-hint-chip">Explain recursion simply</div>'
            '<div class="ga-hint-chip">Give me three startup ideas</div>'
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
                f'<div class="ga-msg ga-msg-user slide-up">'
                f'<div class="ga-msg-content">{text}</div>'
                f'</div>'
            )
        else:
            parts.append(
                f'<div class="ga-msg ga-msg-assistant slide-up">'
                f'<div class="ga-icon-wrapper"><span class="ga-ai-icon">✧</span></div>'
                f'<div class="ga-msg-content">{text}</div>'
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
        status = f"Your quiz on '{topic}' is ready. Open the Voice Quiz tab."
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
        status = f"Your quiz on '{topic}' is ready. Open the Voice Quiz tab."
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
# UI – CSS  (Minimalist Premium AI Product Upgrade)
# ==========================================================================

CUSTOM_CSS = """
@import url('[https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap)');

:root {
  --bg-app: #FAFAFA;
  --bg-surface: #FFFFFF;
  --bg-hover: #F3F4F6;
  --text-main: #111827;
  --text-muted: #6B7280;
  --text-inverse: #FFFFFF;
  --border-color: #E5E7EB;
  --accent: #FF5A1F;
  --accent-hover: #E84D16;
  --accent-soft: #FFF2ED;
  --accent-muted: #FFD8CB;
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --radius-full: 9999px;
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
  --shadow-float: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
  --shell: 1100px;
  --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Base Gradio Overrides */
html, body {
  margin: 0 !important;
  background: var(--bg-app) !important;
  color: var(--text-main) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
  -webkit-font-smoothing: antialiased;
}
.gradio-container {
  width: 100% !important;
  max-width: none !important;
  min-height: 100vh !important;
  margin: 0 !important;
  padding: 0 !important;
  background: var(--bg-app) !important;
}
.gradio-container > div:first-child {
  max-width: var(--shell) !important;
  margin: 0 auto !important;
  padding: 0 24px 64px !important;
}
footer { display: none !important; }
.block, .form, .wrap, .gap, .tabs, .tabitem {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
button, input, textarea, select { font-family: inherit !important; }

/* Hidden Audio Outputs */
.sv-audio-out {
  position: fixed !important;
  width: 1px !important;
  height: 1px !important;
  opacity: 0 !important;
  pointer-events: none !important;
  overflow: hidden !important;
}

/* Animations */
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes slideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
.fade-in { animation: fadeIn 0.4s ease-out forwards; }
.slide-up { animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards; }

/* Header */
#sv-header {
  height: 80px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--border-color);
  background: var(--bg-app);
  margin-bottom: 32px;
}
.sv-logo-row { display: flex; align-items: center; gap: 12px; }
.sv-wordmark { color: var(--text-main); font-size: 20px; font-weight: 700; letter-spacing: -0.02em; }
.sv-tagline { color: var(--text-muted); font-size: 14px; font-weight: 400; margin-left: 12px; padding-left: 12px; border-left: 1px solid var(--border-color); }
.sv-online-dot { display: inline-flex; align-items: center; gap: 8px; color: var(--text-muted); font-size: 13px; font-weight: 500; }
.sv-online-dot:before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: #10B981; }

/* Navigation Tabs */
.tab-nav { border-bottom: 1px solid var(--border-color) !important; background: transparent !important; margin: 0 0 40px 0 !important; gap: 32px !important; display: flex !important; flex-wrap: nowrap !important; overflow-x: auto !important; padding-bottom: 1px !important;}
.tab-nav button {
  padding: 0 0 12px 0 !important; margin: 0 32px 0 0 !important; border: 0 !important; border-radius: 0 !important;
  background: transparent !important; color: var(--text-muted) !important;
  font-size: 15px !important; font-weight: 500 !important; box-shadow: none !important;
  border-bottom: 2px solid transparent !important; transition: var(--transition) !important;
}
.tab-nav button:hover { color: var(--text-main) !important; }
.tab-nav button.selected { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }

/* Typography */
.sv-page-head { margin-bottom: 40px; }
.sv-page-title, .ga-empty-title { font-size: 36px !important; line-height: 1.1 !important; font-weight: 700 !important; color: var(--text-main) !important; letter-spacing: -0.03em !important; }
.sv-page-sub, .ga-empty-sub { margin-top: 12px; color: var(--text-muted) !important; font-size: 16px !important; line-height: 1.6 !important; max-width: 600px !important; font-weight: 400 !important; }

/* General Assistant Workspace */
.sv-workspace { width: 100% !important; min-height: 0 !important; }
.ga-empty-state { text-align: left !important; padding: 40px 0 !important; }
.ga-empty-hints { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 24px; }
.ga-hint-chip, .quiz-idle-hint { padding: 8px 16px; border: 1px solid var(--border-color); border-radius: var(--radius-full); background: var(--bg-surface); color: var(--text-muted); font-size: 13px; font-weight: 500; cursor: pointer; transition: var(--transition); }
.ga-hint-chip:hover { border-color: var(--text-muted); color: var(--text-main); }

/* Conversation */
#ga_conv { min-height: 100px !important; }
.ga-conversation { display: flex; flex-direction: column; gap: 24px; padding-bottom: 40px; }
.ga-msg { display: flex; width: 100%; max-width: 800px; }
.ga-msg-user { justify-content: flex-end; align-self: flex-end; }
.ga-msg-user .ga-msg-content { background: var(--bg-surface); border: 1px solid var(--border-color); color: var(--text-main); border-radius: 16px 16px 4px 16px; }
.ga-msg-assistant { justify-content: flex-start; align-self: flex-start; gap: 16px; }
.ga-msg-assistant .ga-msg-content { background: transparent; color: var(--text-main); padding: 0 !important; }
.ga-icon-wrapper { width: 32px; height: 32px; border-radius: 8px; background: var(--accent-soft); display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.ga-ai-icon { color: var(--accent); font-size: 16px; }
.ga-msg-content { padding: 12px 16px; font-size: 15px; line-height: 1.6; }

/* Composer / Inputs */
.ga-composer-row { display: flex !important; gap: 12px !important; align-items: flex-end !important; background: transparent !important; margin-bottom: 24px !important;}
.ga-text-input { flex: 1 1 auto !important; min-width: 0 !important; }
.gradio-container textarea {
  background: var(--bg-surface) !important; border: 1px solid var(--border-color) !important; border-radius: var(--radius-md) !important; color: var(--text-main) !important;
  box-shadow: var(--shadow-sm) !important; font-size: 15px !important; line-height: 1.5 !important; transition: var(--transition) !important; padding: 16px !important; resize: none !important;
}
.gradio-container textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--accent-soft) !important; outline: none !important; }
.gradio-container textarea::placeholder { color: #9CA3AF !important; }
.ga-composer-row textarea { min-height: 56px !important; max-height: 120px !important; }

.sv-send-btn { min-width: 100px !important; height: 56px !important; border: 0 !important; border-radius: var(--radius-md) !important; background: var(--text-main) !important; color: var(--text-inverse) !important; font-size: 14px !important; font-weight: 600 !important; transition: var(--transition) !important; cursor: pointer !important; }
.sv-send-btn:hover { background: #000000 !important; transform: translateY(-1px) !important; box-shadow: var(--shadow-md) !important; }

.sv-divider { display: flex; align-items: center; gap: 16px; margin: 32px 0; color: var(--text-muted); font-size: 12px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
.sv-divider:before, .sv-divider:after { content: ""; height: 1px; background: var(--border-color); flex: 1; }

.ga-bottom-row { gap: 32px !important; align-items: flex-start !important; }
.ga-context-box { padding: 16px; background: var(--bg-surface); border: 1px solid var(--border-color); border-radius: var(--radius-md); font-size: 13px; line-height: 1.5; }
.ga-context-label { color: var(--text-muted); font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
.ga-context-topic { color: var(--text-main); font-size: 14px; font-weight: 500; }
.ga-context-idle { color: var(--text-muted); font-style: italic; }

/* Status Box */
.sv-status-box { min-height: 0 !important; margin: 0 !important; }
.sv-status-box > div { padding: 0 !important; border: 0 !important; }
.sv-status-box textarea { min-height: 0 !important; height: auto !important; max-height: 60px !important; padding: 12px 16px !important; border: 1px solid var(--accent-muted) !important; border-radius: var(--radius-md) !important; background: var(--accent-soft) !important; color: var(--accent-hover) !important; font-size: 13px !important; }
.sv-status-box textarea:placeholder-shown { display: none !important; }

/* Mic Recording UI */
.sv-mic-hint { color: var(--text-muted); font-size: 13px; font-weight: 500; margin-bottom: 12px; }
#ga_mic_input, #sa_mic_input, #quiz_mic_input { border: 1px solid var(--border-color) !important; border-radius: var(--radius-md) !important; background: var(--bg-surface) !important; padding: 8px !important; transition: var(--transition) !important; }
#ga_mic_input:focus-within, #sa_mic_input:focus-within, #quiz_mic_input:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--accent-soft) !important; }
#ga_mic_input button, #sa_mic_input button, #quiz_mic_input button { border-radius: var(--radius-sm) !important; color: var(--text-main) !important; }
#ga_mic_input [aria-label*="Record"], #sa_mic_input [aria-label*="Record"], #quiz_mic_input [aria-label*="Record"] { color: var(--accent) !important; }

/* Study Agent Layout */
.study-grid { display: grid !important; grid-template-columns: minmax(0, 380px) minmax(0, 1fr) !important; gap: 48px !important; align-items: start !important; }
.study-left, .study-right { min-width: 0 !important; }
.study-panel { background: var(--bg-surface); border: 1px solid var(--border-color); border-radius: var(--radius-lg); padding: 24px; box-shadow: var(--shadow-sm); margin-bottom: 24px; }
#transcript_box textarea { min-height: 200px !important; max-height: 300px !important; border: 0 !important; box-shadow: none !important; padding: 0 !important; background: transparent !important; }
.sa-transcript-box .svelte-1b6s6s { border: 0 !important; background: transparent !important; } /* gradio inner wrap */
.sv-plan-label { font-size: 14px; font-weight: 600; color: var(--text-main); margin-bottom: 16px; display: block; }
#sv_plan_panel > div { padding: 0 !important; border: 0 !important; background: transparent !important; }

/* Premium Timeline Study Plan */
.plan-empty-state { text-align: center; padding: 64px 24px; border: 1px dashed var(--border-color); border-radius: var(--radius-lg); color: var(--text-muted); }
.plan-empty-icon { font-size: 24px; margin-bottom: 16px; color: var(--accent); }
.plan-placeholder-title { font-size: 18px; font-weight: 600; color: var(--text-main); margin-bottom: 8px; }
.plan-placeholder-copy { font-size: 14px; line-height: 1.5; max-width: 320px; margin: 0 auto; }
.plan-title { font-size: 24px; font-weight: 700; color: var(--text-main); margin-bottom: 32px; letter-spacing: -0.02em; }
.plan-timeline { position: relative; padding-left: 0; }
.plan-block { display: flex; gap: 48px; padding-bottom: 32px; position: relative; }
.plan-time-col { width: 100px; flex-shrink: 0; display: flex; flex-direction: column; align-items: flex-end; position: relative; }
.plan-time { font-size: 13px; font-weight: 600; color: var(--text-muted); text-align: right; }
.timeline-marker { position: absolute; right: -30px; top: 2px; width: 10px; height: 10px; border-radius: 50%; z-index: 2; border: 2px solid var(--bg-app); }
.timeline-marker-study { background: var(--text-main); }
.timeline-marker-break { background: var(--border-color); }
.plan-block:not(:last-child) .plan-time-col::after { content: ''; position: absolute; right: -26px; top: 12px; bottom: -34px; width: 2px; background: var(--border-color); z-index: 1; }
.plan-block-content { flex: 1; padding-top: -2px; }
.plan-topic { font-size: 16px; font-weight: 600; color: var(--text-main); margin-bottom: 4px; }
.plan-goal { font-size: 14px; line-height: 1.5; color: var(--text-muted); }
.plan-break .plan-topic { color: var(--text-muted); font-weight: 500; }

/* Voice Quiz Refinement */
.quiz-meta-shell { display: flex !important; justify-content: space-between !important; gap: 16px !important; margin-bottom: 24px !important; }
.quiz-meta-row { display: flex; gap: 12px; }
.quiz-meta-item { background: var(--bg-surface); border: 1px solid var(--border-color); border-radius: var(--radius-md); padding: 12px 16px; min-width: 120px; }
.quiz-meta-label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 4px; }
.quiz-meta-value { font-size: 14px; font-weight: 600; color: var(--text-main); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px; }

.quiz-center-panel, .quiz-idle-hero, .quiz-complete-panel {
  background: var(--bg-surface); border: 1px solid var(--border-color); border-radius: var(--radius-lg); box-shadow: var(--shadow-float); padding: 64px 48px; margin-bottom: 32px; text-align: center; min-height: 380px; display: flex; flex-direction: column; justify-content: center; align-items: center;
}
.quiz-idle-icon { font-size: 32px; color: var(--accent); margin-bottom: 24px; }
.quiz-idle-title { font-size: 24px; font-weight: 700; color: var(--text-main); margin-bottom: 12px; }
.quiz-idle-sub { color: var(--text-muted); font-size: 15px; margin-bottom: 24px; }
.quiz-qnum { font-size: 13px; font-weight: 600; color: var(--accent); letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 24px; }
.quiz-question { font-size: 32px; font-weight: 700; color: var(--text-main); line-height: 1.3; max-width: 800px; letter-spacing: -0.02em; }
.quiz-feedback { margin-top: 32px; padding: 16px 24px; background: var(--accent-soft); border: 1px solid var(--accent-muted); border-radius: var(--radius-md); color: var(--text-main); font-size: 15px; line-height: 1.5; max-width: 600px; text-align: left; }

.quiz-complete-panel { align-items: stretch; text-align: left; padding: 48px; justify-content: flex-start; }
.quiz-complete-header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 40px; padding-bottom: 32px; border-bottom: 1px solid var(--border-color); }
.quiz-complete-title { font-size: 28px; font-weight: 700; color: var(--text-main); letter-spacing: -0.02em; }
.quiz-score-big { font-size: 40px; font-weight: 700; color: var(--accent); line-height: 1; }
.quiz-score-denom { font-size: 20px; color: var(--text-muted); }
.quiz-complete-body { display: flex; flex-direction: column; gap: 24px; }
.quiz-complete-row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.quiz-complete-section { display: flex; flex-direction: column; gap: 8px; }
.quiz-label-sm { font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); }
.quiz-section-content { font-size: 15px; line-height: 1.6; color: var(--text-main); }
.quiz-action-card { background: var(--bg-app); padding: 20px; border-radius: var(--radius-md); border: 1px solid var(--border-color); margin-top: 16px; }
.quiz-next-action { font-size: 16px; font-weight: 600; color: var(--text-main); line-height: 1.5; }

.quiz-mic-strip { align-items: flex-start !important; max-width: 600px !important; margin: 0 auto !important; }

/* Responsive Adjustments */
@media(max-width: 900px) {
  .study-grid { grid-template-columns: 1fr !important; gap: 32px !important; }
  .quiz-meta-shell { flex-direction: column !important; }
  .quiz-meta-row { width: 100%; }
  .quiz-meta-item { flex: 1; }
}
@media(max-width: 600px) {
  .gradio-container > div:first-child { padding: 0 16px 40px !important; }
  #sv-header { flex-direction: column; align-items: flex-start; height: auto; padding: 20px 0; gap: 12px; }
  .sv-tagline { display: none; } /* Simplify header on mobile */
  .sv-page-title { font-size: 28px !important; }
  .ga-composer-row { flex-direction: column !important; align-items: stretch !important; }
  .sv-send-btn { width: 100% !important; }
  .quiz-center-panel { padding: 40px 24px; min-height: 280px; }
  .quiz-question { font-size: 24px; }
  .quiz-complete-header { flex-direction: column; align-items: flex-start; gap: 16px; }
  .quiz-complete-row { grid-template-columns: 1fr; }
}
"""


# ==========================================================================
# UI BUILD  (Presentation layer upgrade)
# ==========================================================================

_blank_qs = _blank_quiz_state()

with gr.Blocks(title="StudyVoice", css=CUSTOM_CSS) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.HTML(
        '<div id="sv-header">'
        '<div class="sv-logo-row"><span class="sv-wordmark">StudyVoice</span>'
        '<span class="sv-tagline">Voice-first AI learning copilot</span></div>'
        '<span class="sv-online-dot">AI Online</span>'
        '</div>',
        elem_id="sv_header_html",
    )

    # ── Mode selector tabs ───────────────────────────────────────────────────
    with gr.Tabs(elem_classes=["tab-nav"]):

        # ══════════════════════════════════════════════════════════════════
        # TAB 1 — GENERAL ASSISTANT
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("General Assistant"):

            ga_history_state = gr.State([])

            # Hidden state refs for quiz cross-tab sync
            ga_quiz_state_ref  = gr.State(_blank_quiz_state())
            ga_quiz_left_ref   = gr.HTML(visible=False)
            ga_quiz_center_ref = gr.HTML(visible=False)
            ga_quiz_right_ref  = gr.HTML(visible=False)

            with gr.Column(elem_classes=["sv-workspace"]):
                gr.HTML(
                    '<div class="sv-page-head fade-in">'
                    '<div class="sv-page-title">Workspace</div>'
                    '<div class="sv-page-sub">Ask questions, explore concepts, and generate quizzes dynamically.</div>'
                    '</div>'
                )

                # Conversation display
                ga_conv_html = gr.HTML(
                    value=_render_ga_conversation([]),
                    elem_id="ga_conv",
                )

                # Text input row
                with gr.Row(equal_height=True, elem_classes=["ga-composer-row fade-in"]):
                    ga_text_in = gr.Textbox(
                        label="",
                        placeholder="Message StudyVoice...",
                        lines=1,
                        show_label=False,
                        scale=6,
                        elem_classes=["ga-text-input"],
                    )
                    ga_send_btn = gr.Button(
                        "Send",
                        scale=1,
                        elem_classes=["sv-send-btn"],
                    )

                # Status (hidden until error)
                ga_status = gr.Textbox(
                    label="",
                    value="",
                    interactive=False,
                    lines=1,
                    show_label=False,
                    elem_classes=["sv-status-box"],
                )

                # Audio output (compact autoplay, rendered but visually hidden)
                ga_audio_out = gr.Audio(
                    label="",
                    autoplay=True,
                    interactive=False,
                    show_label=False,
                    visible=True,
                    elem_classes=["sv-audio-out"],
                )

                # Thin divider before voice section
                gr.HTML('<div class="sv-divider fade-in">Voice Input</div>')

                # Voice input row
                with gr.Row(equal_height=False, elem_classes=["ga-bottom-row fade-in"]):
                    with gr.Column(scale=1, min_width=260):
                        gr.HTML('<div class="sv-mic-hint">Record your question or request.</div>')
                        ga_voice_in = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            format="wav",
                            label="",
                            elem_id="ga_mic_input",
                            show_label=False,
                        )
                    with gr.Column(scale=1, min_width=260):
                        ga_context_display = gr.HTML(
                            value=(
                                '<div class="ga-context-box"><div class="ga-context-label">Current Context</div>'
                                '<div class="ga-context-idle">No topic active.</div></div>'
                            ),
                        )

            # ── Event wiring (GA) ─────────────────────────────────────────
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
            ga_voice_in.stop_recording(
                fn=handle_ga_voice,
                inputs=[ga_voice_in, ga_history_state, ga_conv_html, ga_quiz_state_ref],
                outputs=_ga_voice_outputs,
            )

        # ══════════════════════════════════════════════════════════════════
        # TAB 2 — STUDY AGENT
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("Study Agent"):

            history_state = gr.State([])
            plan_state    = gr.State(None)

            with gr.Column(elem_classes=["sv-workspace"]):
                gr.HTML(
                    '<div class="sv-page-head fade-in">'
                    '<div class="sv-page-title">Study Planner</div>'
                    '<div class="sv-page-sub">Tell StudyVoice your subjects and constraints to map out your optimal schedule.</div>'
                    '</div>'
                )

                with gr.Row(elem_classes=["study-grid fade-in"]):
                    with gr.Column(elem_classes=["study-left"], min_width=320):
                        with gr.Column(elem_classes=["study-panel"]):
                            gr.HTML('<div class="sv-mic-hint">Record your study requirements</div>')
                            audio_in = gr.Audio(
                                sources=["microphone"],
                                type="filepath",
                                format="wav",
                                label="",
                                elem_id="sa_mic_input",
                                show_label=False,
                            )
                            error_box = gr.Textbox(
                                label="",
                                value="",
                                interactive=False,
                                lines=1,
                                show_label=False,
                                elem_classes=["sv-status-box"],
                            )
                            audio_out = gr.Audio(
                                label="",
                                autoplay=True,
                                interactive=False,
                                show_label=False,
                                visible=True,
                                elem_classes=["sv-audio-out"],
                            )
                        transcript_box = gr.Textbox(
                            label="Conversation History",
                            value="",
                            interactive=False,
                            lines=8,
                            show_label=True,
                            elem_id="transcript_box",
                            elem_classes=["sa-transcript-box"],
                            placeholder="Interactions will appear here...",
                        )

                    with gr.Column(elem_classes=["study-right"], min_width=420):
                        gr.HTML('<div class="sv-plan-label">Generated Schedule</div>')
                        plan_panel = gr.Markdown(
                            value=PLAN_PLACEHOLDER,
                            elem_id="sv_plan_panel",
                        )

            # ── Turn outputs (exact order preserved) ──────────────────────
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

        # ══════════════════════════════════════════════════════════════════
        # TAB 3 — VOICE QUIZ
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("Voice Quiz"):

            quiz_state = gr.State(_blank_quiz_state())

            with gr.Column(elem_classes=["sv-workspace"]):
                gr.HTML(
                    '<div class="sv-page-head fade-in" style="margin-bottom: 24px;">'
                    '<div class="sv-page-title">Evaluation</div>'
                    '</div>'
                )

                # Hidden audio output — callback-connected and autoplay-enabled
                quiz_audio_out = gr.Audio(
                        label="",
                        autoplay=True,
                        interactive=False,
                        show_label=False,
                        visible=True,
                        elem_classes=["sv-audio-out"],
                    )

                with gr.Row(equal_height=False, elem_classes=["quiz-meta-shell"]):
                    with gr.Column(min_width=200):
                        quiz_left = gr.HTML(value=render_quiz_left(_blank_qs))
                    with gr.Column(min_width=200):
                        quiz_right = gr.HTML(value=render_quiz_right(_blank_qs))

                quiz_center = gr.HTML(
                    value=render_quiz_center(_blank_qs),
                )

                with gr.Row(elem_classes=["quiz-mic-strip fade-in"], elem_id="quiz_mic_strip_row"):
                    with gr.Column(min_width=180):
                        gr.HTML(
                            '<div class="sv-mic-hint" style="text-align:left;margin-bottom:8px;">'
                            'Respond to the question</div>'
                        )
                        quiz_audio_in = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            format="wav",
                            label="",
                            elem_id="quiz_mic_input",
                            show_label=False,
                        )
                    with gr.Column(min_width=120):
                        quiz_status = gr.Textbox(
                            label="",
                            value="",
                            interactive=False,
                            lines=1,
                            show_label=False,
                            elem_classes=["sv-status-box"],
                        )

            # ── Quiz outputs (exact order preserved) ──────────────────────
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

            # ── Cross-tab sync: GA → Quiz ─────────────────────────────────
            def _sync_quiz_from_ga(ga_qs, ga_ql, ga_qc, ga_qr):
                return ga_qs, ga_ql, ga_qc, ga_qr

            ga_quiz_state_ref.change(
                fn=_sync_quiz_from_ga,
                inputs=[ga_quiz_state_ref, ga_quiz_left_ref, ga_quiz_center_ref, ga_quiz_right_ref],
                outputs=[quiz_state, quiz_left, quiz_center, quiz_right],
            )

            # ── Context display sync in GA sidebar ────────────────────────
            def _render_ga_context(ga_history):
                if not ga_history:
                    return (
                        '<div class="ga-context-box"><div class="ga-context-label">Current Context</div>'
                        '<div class="ga-context-idle">No topic active.</div></div>'
                    )
                last_topic = None
                for turn in reversed(ga_history):
                    if turn.get("role") == "user" and turn.get("text", "") != "(voice message)":
                        last_topic = turn["text"]
                        break
                if not last_topic:
                    last_topic = "(voice)"
                escaped = _plan_escape(last_topic[:120] + ("..." if len(last_topic) > 120 else ""))
                return (
                    '<div class="ga-context-box">'
                    '<div class="ga-context-label">Last Topic Identified</div>'
                    f'<div class="ga-context-topic">{escaped}</div>'
                    '<div class="ga-context-label" style="margin-top:16px">Quick Action</div>'
                    '<div class="ga-context-idle" style="font-style: normal; color: var(--text-main); font-weight: 500;">Say <span style="background: var(--bg-app); padding: 2px 6px; border-radius: 4px; border: 1px solid var(--border-color); font-size: 12px;">"Quiz me on that"</span> to test this.</div>'
                    '</div>'
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
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )