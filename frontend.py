"""Front-end: the Streamlit user interface for the interview preparation app."""

import streamlit as st

from config import client, set_openai_params, validator_params
from data_schemas import ValidationOutput
from prompts import (
    get_system_prompt,
    get_user_prompt,
    validator_system_prompt,
    validator_user_prompt,
)
from backend import (
    clean_input_text,
    generate_interview_questions,
    generate_validation_response,
    get_answer_feedback,
    transcribe_audio,
)


def main():
    st.title("AI Interview Preparation Assistant")

    # Initialize session state
    if 'question_history' not in st.session_state:
        st.session_state.question_history = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'current_job_position' not in st.session_state:
        st.session_state.current_job_position = None
    if 'current_interview_type' not in st.session_state:
        st.session_state.current_interview_type = None

    # --- Job position input ---
    job_position = st.text_input("Please enter the job position for which you'd like to receive interview questions:")

    cleaned_input = ""
    validation_result = None

    if job_position:
        cleaned_input = clean_input_text(job_position)

        if cleaned_input.startswith("Invalid input") or cleaned_input.startswith("Job position name exceeds"):
            st.error(cleaned_input)
        else:
            validation_result = generate_validation_response(
                job_position=cleaned_input,
                system_prompt=validator_system_prompt,
                user_prompt=validator_user_prompt,
                client=client,
                params=validator_params,
                response_format=ValidationOutput
            )

    st.subheader("What type of interview would you like to practice for?")
    st.markdown(
        "Select from different interview types to prepare for your upcoming interview. "
        "Each type focuses on different aspects of the interview process. "
        "If you are unsure with whom you will be having an interview, select General."
    )

    display_names = {
        "HR Screening": "HR",
        "Technical": "Technical",
        "Interview with a Manager": "Manager",
        "General": "Unspecified"
    }

    selected_display = st.radio(
        "Choose the type of interview:",
        list(display_names.keys()),
        label_visibility="collapsed"
    )

    interview_type = display_names[selected_display]
    st.markdown("---")

    # Question settings in sidebar (numeric sliders with worded endpoints)
    st.sidebar.header("Question Settings")
    st.sidebar.caption("Fine-tune how the interview questions are generated.")

    # Hide the built-in numeric min/max tick labels so only the worded endpoints show
    st.markdown(
        """
        <style>
        [data-testid="stSliderTickBarMin"],
        [data-testid="stSliderTickBarMax"],
        [data-testid="stTickBarMin"],
        [data-testid="stTickBarMax"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def labeled_slider(label, min_v, max_v, default, step, low_label, high_label, help_text):
        value = st.sidebar.slider(label, min_v, max_v, default, step, help=help_text)
        left, right = st.sidebar.columns(2)
        left.markdown(f"<small>{low_label}</small>", unsafe_allow_html=True)
        right.markdown(
            f"<small style='float:right'>{high_label}</small>",
            unsafe_allow_html=True,
        )
        return value

    temperature = labeled_slider(
        "Creativity", 0.0, 1.0, 0.6, 0.1,
        "Less creative", "More creative",
        "Lower = focused, predictable questions. Higher = more varied and unexpected questions.",
    )
    max_tokens = labeled_slider(
        "Question length", 150, 2000, 1000, 100,
        "Short", "Long",
        "How long and detailed each generated question can be.",
    )
    top_p = labeled_slider(
        "Word variety", 0.0, 1.0, 0.8, 0.1,
        "More focused", "More varied",
        "Higher allows a wider range of vocabulary and phrasing.",
    )
    frequency_penalty = labeled_slider(
        "Word repetition", 0.0, 2.0, 0.4, 0.1,
        "Allow repetition", "Avoid repetition",
        "Higher reduces repeated words and phrasing across questions.",
    )
    presence_penalty = labeled_slider(
        "New topics", 0.0, 2.0, 0.2, 0.1,
        "Familiar themes", "Explore new topics",
        "Higher pushes the AI to explore fresh topics instead of sticking to the same themes.",
    )

    # --- Generate Question button ---
    if st.button("Generate Question"):
        if job_position and not cleaned_input.startswith("Invalid input") and not cleaned_input.startswith("Job position name exceeds"):
            if validation_result == "yes":
                with st.spinner("Generating question..."):
                    system_prompt = get_system_prompt(interview_type)
                    user_prompt = get_user_prompt(interview_type, job_position, 1)

                    generator_params = set_openai_params(
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty
                    )

                    new_question, updated_history = generate_interview_questions(
                        job_position=job_position,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        client=client,
                        params=generator_params,
                        num_questions=1,
                        question_history=st.session_state.question_history
                    )

                    if new_question:
                        st.session_state.question_history = updated_history
                        st.session_state.current_question = new_question
                        st.session_state.current_job_position = job_position
                        st.session_state.current_interview_type = interview_type
                        # Clear any previous answer/feedback when a new question is generated
                        st.session_state.pop('transcribed_answer', None)
                        st.session_state.pop('typed_answer', None)
                        st.session_state.pop('answer_feedback', None)
                    else:
                        st.error("Failed to generate question.")

            elif validation_result == "no":
                st.error("The job position title seems not valid. Please try a different job title.")
            elif validation_result == "uncertain":
                st.warning("The AI assistant cannot seem to recognize the job position. Please try a different job title.")
            else:
                st.error("An error occurred during validation. Please try again.")
        else:
            st.warning("Please enter a valid job position.")

    # --- Display the current question ---
    if st.session_state.current_question:
        st.subheader("Generated Question:")
        st.write(st.session_state.current_question)

        st.markdown("---")

        # --- Answer section: choose how to respond ---
        answer_mode = st.radio(
            "How would you like to answer?",
            ["Speak my answer", "Type my answer"],
            horizontal=True,
        )

        final_answer = None

        if answer_mode == "Speak my answer":
            st.markdown("Press the microphone button to record your answer, then click **Get Feedback** when you're done.")

            audio_input = st.audio_input(
                "Record your answer here",
                label_visibility="collapsed",
            )

            if audio_input:
                # Show transcription
                if 'transcribed_answer' not in st.session_state or st.session_state.get('last_audio') != audio_input:
                    with st.spinner("Transcribing your answer..."):
                        transcribed_text = transcribe_audio(audio_input, client)
                        st.session_state.transcribed_answer = transcribed_text
                        st.session_state.last_audio = audio_input
                        # Clear old feedback when a new recording comes in
                        st.session_state.pop('answer_feedback', None)

                st.markdown("**Your answer (transcript):**")
                st.info(st.session_state.transcribed_answer)

                if not st.session_state.transcribed_answer.startswith("Error"):
                    final_answer = st.session_state.transcribed_answer

        else:
            st.markdown("Type your answer below, then click **Get Feedback** when you're done.")

            typed_answer = st.text_area(
                "Type your answer here",
                value=st.session_state.get("typed_answer", ""),
                height=200,
                label_visibility="collapsed",
            )
            st.session_state.typed_answer = typed_answer
            if typed_answer.strip():
                final_answer = typed_answer.strip()

        # --- Feedback button (shared by both modes) ---
        if st.button("Get Feedback"):
            if not final_answer:
                st.warning("Please provide an answer first — type something or record your response.")
            else:
                with st.spinner("Analysing your answer..."):
                    feedback = get_answer_feedback(
                        job_position=st.session_state.current_job_position,
                        interview_type=st.session_state.current_interview_type,
                        question=st.session_state.current_question,
                        answer=final_answer,
                        client=client
                    )
                    st.session_state.answer_feedback = feedback

        # Display feedback if available
        if 'answer_feedback' in st.session_state:
            st.markdown("---")
            st.subheader("Feedback on Your Answer")
            st.markdown(st.session_state.answer_feedback)
