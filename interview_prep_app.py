import streamlit as st
import re

from openai import OpenAI
from openai.types.chat import ChatCompletion

from typing import Callable, List, Any, Dict, Union, Literal, Tuple
from pydantic import BaseModel

# Output classes
class ValidationOutput(BaseModel):
    job_position: str
    validation_result: Literal["yes", "no", "uncertain"]

class InterviewQuestion(BaseModel):
    question: str

class InterviewOutput(BaseModel):
    job_position: str
    interview_type: str
    questions: List[InterviewQuestion]

# Helper functions
def clean_input_text(input_text: str) -> str:
    """
    Clean the input text by:
    1. Removing any non-alphabetic characters except spaces, apostrophes, and hyphens.
    2. Converting the text to lowercase.
    3. Checking if the input is valid (not only spaces, apostrophes, or hyphens).
    4. Ensuring the input is within the max character limit of 150 characters.
    """
    cleaned_input = re.sub(r'[^a-zA-Z\s\'-]', '', input_text)
    cleaned_input = cleaned_input.lower()

    # Check for empty input or invalid characters
    if cleaned_input == "" or re.match(r'^[\s\'-]*$', cleaned_input):
        return "Invalid input. Your input can only include letters, spaces, apostrophes, and hyphens."
    
    # Check if the input exceeds 150 characters
    if len(cleaned_input) > 150:
        return "Job position name exceeds the maximum length of 150 characters. Please provide a shorter job title."

    return cleaned_input

def set_openai_params(
    model: str = "gpt-4o",
    temperature: float = 0,
    max_tokens: int = 1000,
    top_p: float = 1,
    frequency_penalty: float = 0,
    presence_penalty: float = 0
) -> Dict[str, Union[str, int, float]]:
    """
    Set OpenAI parameters
    """

    openai_params = {
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p,
        'frequency_penalty': frequency_penalty,
        'presence_penalty': presence_penalty
    }
    return openai_params

def format_user_prompt(
    job_position: str,
    num_questions: int,
    user_prompt: str
) -> str:
    """
    Format user prompt with job position and question count
    """

    formatted_prompt = user_prompt.format(
        job_position=job_position,
        num_questions=num_questions
    )
    return formatted_prompt

def get_api_key(key_name: str = "OPEN_API_KEY") -> str:
    """
    Retrieve API key from Streamlit secrets
    """

    api_key = st.secrets[key_name]
    return api_key

def get_openai_client(api_key: str) -> OpenAI:
    """
    Initialize and return OpenAI client
    """

    client = OpenAI(api_key=api_key)
    return client

def get_completion(
    params: Dict[str, Union[str, int, float]],
    messages: List[Dict[str, str]],
    client: OpenAI,
    response_format: BaseModel
) -> InterviewOutput:
    """
    Get completion from OpenAI API
    """
    
    response = client.beta.chat.completions.parse(
        model=params['model'],
        messages=messages,
        temperature=params['temperature'],
        max_tokens=params['max_tokens'],
        top_p=params['top_p'],
        frequency_penalty=params['frequency_penalty'],
        presence_penalty=params['presence_penalty'],
        response_format=response_format
    )
    
    return response

def generate_interview_questions(
    job_position: str,
    system_prompt: str,
    user_prompt: str,
    client: OpenAI,
    params: Dict[str, Union[str, int, float]],
    num_questions: int = 1,
    response_format = InterviewOutput,
    question_history = None
) -> Tuple[str, List[str]]:
    """
    Generate interview questions for a given job position using OpenAI's API.
    """

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
            response_format=response_format
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

def run_interview_question_generation(
    job_position: str,
    num_questions: int,
    client: Any,
    params: Dict[str, Any],
    get_system_prompt: Callable[[str], str],
    get_user_prompt: Callable[[str, str, int], str],
    roles: List[str] = ["HR", "Technical", "Manager", "Unspecified"],
    response_output = InterviewOutput
) -> None:
    """
    This function generates interview questions for a specified job position and role types (HR, Technical, Manager, Unspecified).
    It uses the provided system and user prompts to generate tailored interview questions for each role, based on the
    steps defined in the prompt.
    """
        
    for role in roles:
        print(f"\n--- {role} Interview Questions ---\n")

        system_prompt = get_system_prompt(interview_type=role)
        user_prompt = get_user_prompt(
            interview_type=role,
            job_position=job_position,
            num_questions=num_questions
        )

        questions = generate_interview_questions(
            job_position=job_position,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            client=client,
            params=params,
            num_questions=num_questions
        )

        for question in questions:
            print(question.question)

def generate_validation_response(
    job_position: str,
    system_prompt: str,
    user_prompt: str,
    client: OpenAI,
    params: Dict[str, Union[str, int, float]],
    response_format = ValidationOutput
) -> str:
    """
    Generate a validation result for a given job position using OpenAI's API.
    """

    formatted_user_prompt = validator_user_prompt.format(job_position=job_position)

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_user_prompt}
        ]

        response = get_completion(
            params=params,
            messages=messages,
            client=client,
            response_format=response_format
        )

        parsed_content = response.choices[0].message.parsed
        validation_result = parsed_content.validation_result

        return validation_result

    except Exception as e:
        print(f"Error generating validation: {str(e)}")
        return "error"

# Functions and inouts for prompts
def get_system_prompt(interview_type: str) -> str:
    """
    Generates a system-level prompt tailored to a specific interview type.
    The generated prompt provides guidance on how to approach the interview for different roles such as HR, Technical, Manager, or Unspecified.
    Each prompt focuses on the most relevant aspects for the respective interview type.
    """

    base = (
        "You are an expert interviewer with a deep understanding of diverse interview styles and techniques. "
        "You generate high-quality interview questions tailored to specific job role. "
    )

    if interview_type == "HR":
        return base + (
            "You are an HR recruiter conducting an initial screening interview for a Data Science role. "
            "Your goal is to evaluate the candidate's overall suitability for the position and company. "
            "Focus on assessing their motivation for applying, understanding of the role, communication style, "
            "career background, and alignment with the company's culture and values. You may also ask about availability, "
            "work preferences, and general logistics. Avoid deep technical or team-specific questions."
        )

    elif interview_type == "Technical":
        return base + (
            "You are a technical interviewer assessing the candidate's ability to apply technical skills and knowledge in a professional context. "
            "This may include evaluating their proficiency with relevant tools, techniques, methods, systems, processes, and methods related to the role. "
            "The focus should be on problem-solving abilities, critical thinking, and how the candidate applies their technical expertise. "
            "Your goal is to assess the candidate's ability to leverage their technical skills to solve real-world challenges, improve efficiency, "
            "and contribute to achieving organizational goals."
        )

    elif interview_type == "Manager":
        return base + (
            "You are a team manager looking for someone who can collaborate, adapt, and contribute to team success. "
            "Focus on questions related to teamwork, leadership, handling conflicts, communication, and how the candidate "
            "fits within a team. Emphasize how the candidate collaborates, takes ownership, manages pressure, "
            "and fits within the team's values and culture."
        )
    
    elif interview_type == "Unspecified":
        return base + (
            "You are conducting an interview where the interviewer's specific focus is not known. "
            "You should include a balanced mix of question types appropriate to the job role — such as technical, behavioral, situational, "
            "reflective, motivation-focused, hypothetical, case-based, quiz-style, and logic-based questions. "
            "Your goal is to broadly evaluate the candidate's qualifications, mindset, and fit for the position."
        )
    
    else:
        raise ValueError(f"Unsupported interview type: {interview_type}")

def get_user_prompt(interview_type: str, job_position: str, num_questions: int) -> str:
    """
    Generates a user-level prompt tailored to a specific interview type and job position.
    The prompt guides the model to generate relevant interview questions for the selected role.
    The number of questions to be generated is also specified in the prompt.
    """

    instructions = (
        "Now generate {num_questions} standalone interview questions for a {job_position} role. "
        "Return each question as plain text on a new line. "
        "Do not include numbering, bullets, explanations, or headings.\n\n"
    )

    if interview_type == "HR":
        example_questions = [
            # Experience Overview
            "What has your career path looked like so far, and what has guided your decisions?",
            "What aspects of your past roles have you enjoyed most and why?",
            # Motivation & Fit
            "What drew you to apply for this role?",
            "What do you hope to gain from this opportunity?",
            # Company Fit
            "What kind of company culture do you work best in?",
            "How do you see yourself aligning with our mission and values?",
            # Logistics
            "When are you available to start?",
            "What are your location or remote work preferences?"
        ]

        role_example_questions = (
            "Here are example questions for an HR interview, focused on first-round screening for a Data Scientist role:\n\n" 
            + "\n".join(["- " + q for q in example_questions])
        )

    elif interview_type == "Technical":
        example_questions = [
            # Technical
            "What's the difference between precision and recall? When would you prioritize one over the other?",
            "Explain the intuition behind regularization in machine learning.",
            # Scenario-Based
            "You're given an imbalanced dataset for a binary classification problem. How would you handle it?",
            "You discover data leakage after training a high-performing model. What would you do?",
            # Quiz-Style
            "What is the time complexity of a binary search algorithm?",
            "Name three assumptions behind linear regression.",
            # Logic-Driven
            "If a model performs well on training data but poorly on test data, what are the possible reasons?",
            "How would you choose between a random forest and a gradient boosting model for a given problem?"
        ]

        role_example_questions = (
             "Here are example questions for a Technical interview, for Data Scientist role for illustration:\n\n"
            + "\n".join(["- " + q for q in example_questions])
        )

    elif interview_type == "Manager":
        example_questions = [
            # Teamwork & Communication
            "How do you usually collaborate with non-technical stakeholders or cross-functional teams?",
            "Tell me about a time you had to explain a complex data insight to a non-technical stakeholder.",
            # Leadership (in a collaborative sense)
            "Describe a time when you took ownership of a failing data project. What did you do?",
            "How do you support your teammates when they are facing challenges or underperforming?",
            # Conflict Resolution & Adaptability
            "How would you approach resolving a disagreement between two data scientists on your team?",
            "How do you handle shifting priorities when working on multiple data projects?",
            # Work Preferences & Environment Fit
            "What kind of work environment helps you thrive (e.g., fast-paced, structured, collaborative)?",
            "How do you prioritize tasks when working on multiple projects with different deadlines?",
            "Do you prefer working independently or as part of a team?"
        ]

        role_example_questions = (
            "Here are example questions for an interview conducted by a Team Manager, for Data Scientist role for illustration:\n\n"
            + "\n".join(["- " + q for q in example_questions])
        )

    elif interview_type == "Unspecified":
        example_questions = [
            # Technical
            "What metrics would you use to evaluate a classification model, and why?",
            "How would you deal with missing data in a dataset with thousands of records?",
            # Behavioral
            "Tell me about a time when you had to work under pressure to meet a tight deadline.",
            "Describe how you handle receiving critical feedback on your analysis.",
            # Situational
            "Imagine your model's predictions contradict a stakeholder's intuition. How would you respond?",
            "You're tasked with building a dashboard for executives. How would you ensure it's effective?",
            # Reflective
            "What's the most important lesson you've learned from working on data science projects?",
            "How have you evolved in your approach to problem-solving over your career?",
            # Team-Culture
            "How do you approach working with engineers or product managers who have different goals than yours?",
            "What strategies do you use to stay aligned with your team's mission?",
            # Motivation-Focused
            "What drives your interest in turning data into business impact?",
            "Why do you want to work in this industry specifically as a data scientist?"
        ]
        
        role_example_questions = (
            "Here are example questions for an Unspecified interview for a Data Scientist role for illustration:\n\n"
            + "\n".join(["- " + q for q in example_questions])
        )
    
    user_prompt = role_example_questions + "\n\n" + instructions

    return user_prompt

validator_system_prompt = "You are an expert in human resources and job market trends."

validator_user_prompt = """
   Your task is to verify if the following job title is valid and professional. Please evaluate it according to the following criteria:

   1. **Clear Role Definition**: The job title should clearly define a professional role. 
      It must be a recognized position in the workforce and not be vague or irrelevant.
      - Titles like "Apple Eater" or "Unicorn Hunter" are invalid, as they don't reflect actual professional roles.

   2. **Avoidance of Ambiguity**: The job title should not be confusing or ambiguous. 
      It must be specific enough that someone reading the title understands the key responsibilities without additional context.
      - Titles like "Manager of Everything" or "Chief of All" are unclear and should be flagged.

   3. **Realistic and Feasible**: The job title should describe a role that is feasible and realistic in the professional world.
      - Titles that describe fictional or exaggerated roles should be flagged.
      - Titles such as "Dream Job Creator" or "Time Travel Consultant" are not realistic job titles.

   **Instructions**:
   - Be cautious and precise in your evaluation. If you are uncertain about the title's validity based on the criteria, return "uncertain."
   - If the title is clearly invalid, ambiguous, or does not meet the professional standards, return "no."
   - If the title passes all criteria and is valid, return "yes."

   **Output Options**:
   - "yes" if the job title is valid, professional, clear, and realistic.
   - "no" if the job title is invalid according to the outlined criteria.
   - "uncertain" if you are unsure about the job title's validity.

   Job title: "{job_position}"
"""


# Initialize OpenAI client
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = get_openai_client(api_key=OPENAI_API_KEY)

validator_params = {
    'model': "gpt-4o",
    'temperature': 0,
    'max_tokens': 100,
    'top_p': 1,
    'frequency_penalty': 0,
    'presence_penalty': 0
}

def main():
    st.title("AI Interview Preparation Assistant")
    
    # Initialize session state for tracking questions
    if 'question_history' not in st.session_state:
        st.session_state.question_history = []
    
    job_position = st.text_input("Please enter the job position for which you'd like to receive interview questions:")
    
    if job_position:
        cleaned_input = clean_input_text(job_position)
        
        # Check if the input is valid
        if cleaned_input.startswith("Invalid input") or cleaned_input.startswith("Job position name exceeds"):
            st.error(cleaned_input)
        else:
            # If the text is valid, verify that it corresponds to a valid job position
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
    

    # Display names for the radio buttons
    display_names = {
        "HR Screening": "HR",
        "Technical": "Technical",
        "Interview with a Manager": "Manager",
        "General": "Unspecified"
    }
    
    # Create radio buttons with display names
    selected_display = st.radio(
        "Choose the type of interview:",
        list(display_names.keys()),
        label_visibility="collapsed"
    )
    
    # Convert display name back to original name for function calls
    interview_type = display_names[selected_display]
    st.markdown("---")
    
    # Model parameters in sidebar with moderate defaults
    st.sidebar.header("Model Parameters")
    temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.6, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens:", 150, 2000, 1000, 100)
    top_p = st.sidebar.slider("Top P:", 0.0, 1.0, 0.8, 0.1)
    frequency_penalty = st.sidebar.slider("Frequency Penalty:", 0.0, 2.0, 0.4, 0.1)
    presence_penalty = st.sidebar.slider("Presence Penalty:", 0.0, 2.0, 0.2, 0.1)
    
    # Generate button
    if st.button("Generate Question"):

        # Check if the job position is valid and validation passed
        if job_position and not cleaned_input.startswith("Invalid input") and not cleaned_input.startswith("Job position name exceeds"):
            if validation_result == "yes":
                with st.spinner("Generating question..."):

                    # Get system and user prompts
                    system_prompt = get_system_prompt(interview_type)
                    user_prompt = get_user_prompt(interview_type, job_position, 1)
                    
                    # Set OpenAI parameters
                    generator_params = set_openai_params(
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty
                    )
                    
                    # Generate questions with history
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
                        # Update the session state with the new history
                        st.session_state.question_history = updated_history
                        st.subheader("Generated Question:")
                        st.write(new_question)
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
            

if __name__ == "__main__":
    main() 