# 🎙️ StudyVoice

### A voice-first AI learning copilot that turns natural conversation into action.

## 🎬 Demo

▶️ **[Watch the StudyVoice Demo](https://drive.google.com/file/d/1DkKv24EA2fp_Kne-ggjk4up2jA3Wbhsr/view?usp=drive_link)**

The demo showcases all three core modes of StudyVoice:

1. **General Assistant** — asks StudyVoice to explain polymorphism in simple words.
2. **Study Agent** — creates a structured 5-hour Python study plan focused on OOP and strings.
3. **Voice Quiz** — launches a 3-question basic mathematics quiz, evaluates each answer, tracks the score, and generates a final learning diagnosis.

Together, these workflows demonstrate the core StudyVoice learning loop:

**ASK → PLAN → PRACTICE → DIAGNOSE → ADAPT**


Most AI voice assistants follow a simple loop:

> **Voice → LLM → Voice**

StudyVoice goes further:

> **Voice / Text → Context → Intent → Structured Action → Adaptive Workflow**

Instead of only answering questions, StudyVoice can explain concepts, build and modify study plans, launch contextual quizzes, adapt difficulty based on performance, diagnose weak areas, and recommend what to do next.

## The Core Learning Loop

**ASK → PLAN → PRACTICE → DIAGNOSE → ADAPT**

StudyVoice connects the full learning journey through natural conversation.

---

## ✨ What StudyVoice Can Do

### 💬 General Assistant

A contextual AI learning assistant that supports both text and voice.

- Ask questions by typing or speaking
- Receive concise text and spoken responses
- Maintain multi-turn conversation memory
- Understand contextual follow-ups
- Route conversations into learning workflows

**Example:**

> “Explain polymorphism simply.”

Then:

> “Quiz me on that.”

StudyVoice remembers the topic and launches an adaptive voice quiz on polymorphism.

---

### 📅 Study Agent

Turns natural language into structured, editable study plans.

**Example:**

> “I have a Python exam tomorrow. Make me a 5-hour study plan for OOP, strings and arrays.”

StudyVoice:

- Understands the request directly from voice
- Detects the user's intent
- Generates a structured visual study plan
- Speaks a short summary
- Remembers the current plan

The user can then say:

> “Give OOP one more hour and reduce strings.”

StudyVoice updates the existing plan instead of creating an unrelated new one.

---

### 🧠 Adaptive Voice Quiz

A stateful quiz system designed to practice, evaluate, and adapt.

**Example:**

> “Quiz me on Python OOP for 3 questions.”

StudyVoice:

- Generates one question at a time
- Speaks every question aloud
- Accepts spoken answers
- Evaluates each response
- Awards **1 / 0.5 / 0 points**
- Gives concise feedback
- Adapts difficulty
- Tracks strong and weak areas
- Generates a final diagnosis
- Recommends the next learning action

---

## 🚀 Hero Demo

The key cross-mode workflow:

1. Ask: **“Explain polymorphism simply.”**
2. Follow up: **“Quiz me on that.”**
3. StudyVoice remembers the topic.
4. It automatically launches the Voice Quiz workflow.
5. The learner answers by voice.
6. Difficulty adapts based on performance.
7. StudyVoice diagnoses weak concepts and recommends the next action.

This is the core difference between a voice chatbot and an AI learning copilot.

---

## 🏗️ Architecture

```text
                 ┌───────────────────┐
                 │   Voice / Text    │
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │ Context + Memory  │
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │  Gemini Reasoning │
                 │ + Intent Detection│
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │ Structured Output │
                 │     (Pydantic)    │
                 └─────────┬─────────┘
                           ↓
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
  General Assistant    Study Agent      Voice Quiz
          │                │                │
          ↓                ↓                ↓
    Contextual Q&A    Create / Update   Evaluate / Adapt
          └────────────────┼────────────────┘
                           ↓
                 ┌───────────────────┐
                 │   Text + Voice    │
                 │      Response     │
                 └───────────────────┘
```

---

## 🛠️ Tech Stack

- **Python 3.13**
- **Gradio 5.49.1** — interactive voice and text interface
- **Gemini API** — native audio understanding and reasoning
- **google-genai SDK** — official Gemini SDK
- **Pydantic** — validated structured outputs
- **gTTS** — spoken responses

**Current model:** `gemini-3.1-flash-lite-preview`

---

## 🔑 Technical Highlights

### Native Audio Understanding

Microphone audio is sent directly to Gemini.

There is no separate speech-to-text service in the core voice pipeline.

```text
Microphone
    ↓
Raw Audio
    ↓
Gemini Native Audio Understanding
    ↓
Reasoning + Structured Action
```

### Structured Intent Detection

StudyVoice does not depend on fragile keyword matching.

Gemini returns validated structured outputs that determine whether the user wants to:

- Continue a general conversation
- Create a study plan
- Update an existing plan
- Start or continue a quiz
- Clarify an incomplete request

### Stateful Learning Workflows

StudyVoice maintains state across turns, allowing users to:

- Reference previous conversation topics
- Modify existing study plans
- Continue multi-question quizzes
- Carry context across learning modes

### Defensive Structured Output Parsing

To prevent malformed or truncated model output from reaching the interface:

1. `response.parsed` is preferred
2. Validated Pydantic objects are supported
3. Defensive dictionary parsing is supported
4. Text parsing is used only as a fallback
5. Failed parsing is handled safely without corrupting state
6. Raw JSON is never exposed in the UI

---

## 🎨 Interface

StudyVoice uses a **Midnight + Electric Cyan** visual system designed around three focused modes:

1. **General Assistant**
2. **Study Agent**
3. **Voice Quiz**

Voice interactions use a simple flow:

> **Record → Stop → Auto-process**

No duplicate submit step is required.

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/irenic-aliii/StudyVoice.git
cd StudyVoice
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Set your Gemini API key

**Windows PowerShell:**

```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

### 4. Run StudyVoice

```bash
python app.py
```

For Windows-specific instructions, see `RUN_ME_FIRST.txt`.

---

## 🔒 Security

API keys are never committed to the repository.

The application reads the Gemini API key from the environment:

```text
GEMINI_API_KEY
```

The `.gitignore` excludes local secrets, generated audio files, Python cache files, and Gradio runtime files.

---

## 🎯 Why StudyVoice?

Students usually use separate tools to:

- Understand a concept
- Build a study plan
- Practice questions
- Identify weak areas
- Decide what to study next

StudyVoice connects these steps through one continuous voice-first experience.

> **Speak naturally. Learn adaptively. Take the next action.**

---

## 🔮 Future Direction

StudyVoice can evolve into a persistent learning system with:

- Long-term learner profiles
- Progress tracking across sessions
- Subject-specific learning paths
- Personalized revision scheduling
- Deeper performance analytics

The current prototype focuses on proving the core interaction model:

**Conversation should lead to action, and action should lead to adaptation.**

---

## 👨‍💻 Built By

**Ali**  
Built for an AI Voice Assistant Bootcamp.

---

## 📄 License

This project is currently provided for demonstration and educational purposes.