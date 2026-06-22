"""Back-end: OpenAI request handling — input cleaning, question generation,
job-title validation, answer feedback, and audio transcription."""

import re
from typing import Dict, List, Tuple, Union

from openai import OpenAI
from pydantic import BaseModel

from data_schemas import InterviewOutput, ValidationOutput
from prompts import format_user_prompt


def clean_input_text(input_text: str) -> str:
    cleaned_input = re.sub(r"[^a-zA-Z\s'-]", "", input_text)
    cleaned_input = cleaned_input.lower()

    if cleaned_input == "" or re.match(r"^[\s'-]*$", cleaned_input):
        return "Invalid input. Your input can only include letters, spaces, apostrophes, and hyphens."

    if len(cleaned_input) > 150:
        return "Job position name exceeds the maximum length of 150 characters. Please provide a shorter job title."

    return cleaned_input


def get_completion(
    params: Dict[str, Union[str, int, float]],
    messages: List[Dict[str, str]],
    client: OpenAI,
    response_format: BaseModel,
):
    response = client.beta.chat.completions.parse(
        model=params["model"],
        messages=messages,
        temperature=params["temperature"],
        max_tokens=params["max_tokens"],
        top_p=params["top_p"],
        frequency_penalty=params["frequency_penalty"],
        presence_penalty=params["presence_penalty"],
        response_format=response_format,
    )
    return response


def generate_interview_questions(
    job_position: str,
    system_prompt: str,
    user_prompt: str,
    client: OpenAI,
    params: Dict[str, Union[str, int, float]],
    num_questions: int = 1,
    response_format=InterviewOutput,
    question_history=None,
) -> Tuple[str, List[str]]:
    formatted_user_prompt = format_user_prompt(job_position, num_questions, user_prompt)

    try:
        messages = [{"role": "system", "content": system_prompt}]

        if question_history:
            previous_q_block = "\n".join(question_history)
            memory_message = (
                "Here are questions that have already been asked:\n"
                f"{previous_q_block}\n\n"
                "Please do not repeat these or ask closely related variations. "
                "Try to ensure that the new questions are diverse and varied, "
                "covering a range of relevant topics and including different question types."
            )
            messages.append({"role": "user", "content": memory_message})

        messages.append({"role": "user", "content": formatted_user_prompt})

        response = get_completion(
            params=params,
            messages=messages,
            client=client,
            response_format=response_format,
        )

        parsed_content = response.choices[0].message.parsed
        questions = parsed_content.questions

        display_question = questions[0].question
        if question_history is not None:
            question_history.append(display_question)

        return display_question, question_history

    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        return None, question_history


def generate_validation_response(
    job_position: str,
    system_prompt: str,
    user_prompt: str,
    client: OpenAI,
    params: Dict[str, Union[str, int, float]],
    response_format=ValidationOutput,
) -> str:
    formatted_user_prompt = user_prompt.format(job_position=job_position)

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt},
        ]

        response = get_completion(
            params=params,
            messages=messages,
            client=client,
            response_format=response_format,
        )

        parsed_content = response.choices[0].message.parsed
        return parsed_content.validation_result

    except Exception as e:
        print(f"Error generating validation: {str(e)}")
        return "error"


def get_answer_feedback(
    job_position: str,
    interview_type: str,
    question: str,
    answer: str,
    client: OpenAI,
) -> str:
    """
    Generate AI feedback on the candidate's answer to an interview question.
    Evaluates what was strong, what was missing, and how to improve, speaking
    directly to the user in the second person.
    """
    system_prompt = (
        "You are an experienced interview coach who gives clear, constructive, and encouraging feedback. "
        "Your goal is to help the person improve their interview performance. "
        "Be specific, practical, and balanced — highlight genuine strengths as well as areas for improvement. "
        "Speak directly to the person in the second person (use \"you\" and \"your\"), as if coaching them face to face. "
        "Never refer to them as \"the candidate\" or in the third person."
    )

    user_prompt = f"""
You are coaching someone preparing for a **{interview_type}** interview for a **{job_position}** role.

**Interview Question:**
{question}

**Their Answer:**
{answer}

Please provide structured feedback with the following three sections, speaking directly to them using "you" and "your":

**What was strong** – Specific things you did well in your answer.
**What was missing or could be stronger** – Key points, skills, or details you should have included or elaborated on.
**How to improve** – Concrete, actionable tips to make your answer more compelling in a real interview.

Keep your tone encouraging and professional. Be concise but specific.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
            max_tokens=600,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating feedback: {str(e)}"


def transcribe_audio(audio_file, client: OpenAI) -> str:
    """Transcribe audio using OpenAI's Whisper model."""
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
        return transcript.text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"
