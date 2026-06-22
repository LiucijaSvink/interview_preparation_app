"""System / user / validator prompts and prompt-formatting helpers."""


def format_user_prompt(job_position: str, num_questions: int, user_prompt: str) -> str:
    return user_prompt.format(
        job_position=job_position,
        num_questions=num_questions,
    )


def get_system_prompt(interview_type: str) -> str:
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
    instructions = (
        "Now generate {num_questions} standalone interview questions for a {job_position} role. "
        "Return each question as plain text on a new line. "
        "Do not include numbering, bullets, explanations, or headings.\n\n"
    )

    if interview_type == "HR":
        example_questions = [
            "What has your career path looked like so far, and what has guided your decisions?",
            "What aspects of your past roles have you enjoyed most and why?",
            "What drew you to apply for this role?",
            "What do you hope to gain from this opportunity?",
            "What kind of company culture do you work best in?",
            "How do you see yourself aligning with our mission and values?",
            "When are you available to start?",
            "What are your location or remote work preferences?",
        ]
        role_example_questions = (
            "Here are example questions for an HR interview, focused on first-round screening for a Data Scientist role:\n\n"
            + "\n".join(["- " + q for q in example_questions])
        )

    elif interview_type == "Technical":
        example_questions = [
            "What's the difference between precision and recall? When would you prioritize one over the other?",
            "Explain the intuition behind regularization in machine learning.",
            "You're given an imbalanced dataset for a binary classification problem. How would you handle it?",
            "You discover data leakage after training a high-performing model. What would you do?",
            "What is the time complexity of a binary search algorithm?",
            "Name three assumptions behind linear regression.",
            "If a model performs well on training data but poorly on test data, what are the possible reasons?",
            "How would you choose between a random forest and a gradient boosting model for a given problem?",
        ]
        role_example_questions = (
            "Here are example questions for a Technical interview, for Data Scientist role for illustration:\n\n"
            + "\n".join(["- " + q for q in example_questions])
        )

    elif interview_type == "Manager":
        example_questions = [
            "How do you usually collaborate with non-technical stakeholders or cross-functional teams?",
            "Tell me about a time you had to explain a complex data insight to a non-technical stakeholder.",
            "Describe a time when you took ownership of a failing data project. What did you do?",
            "How do you support your teammates when they are facing challenges or underperforming?",
            "How would you approach resolving a disagreement between two data scientists on your team?",
            "How do you handle shifting priorities when working on multiple data projects?",
            "What kind of work environment helps you thrive (e.g., fast-paced, structured, collaborative)?",
            "How do you prioritize tasks when working on multiple projects with different deadlines?",
            "Do you prefer working independently or as part of a team?",
        ]
        role_example_questions = (
            "Here are example questions for an interview conducted by a Team Manager, for Data Scientist role for illustration:\n\n"
            + "\n".join(["- " + q for q in example_questions])
        )

    elif interview_type == "Unspecified":
        example_questions = [
            "What metrics would you use to evaluate a classification model, and why?",
            "How would you deal with missing data in a dataset with thousands of records?",
            "Tell me about a time when you had to work under pressure to meet a tight deadline.",
            "Describe how you handle receiving critical feedback on your analysis.",
            "Imagine your model's predictions contradict a stakeholder's intuition. How would you respond?",
            "You're tasked with building a dashboard for executives. How would you ensure it's effective?",
            "What's the most important lesson you've learned from working on data science projects?",
            "How have you evolved in your approach to problem-solving over your career?",
            "How do you approach working with engineers or product managers who have different goals than yours?",
            "What strategies do you use to stay aligned with your team's mission?",
            "What drives your interest in turning data into business impact?",
            "Why do you want to work in this industry specifically as a data scientist?",
        ]
        role_example_questions = (
            "Here are example questions for an Unspecified interview for a Data Scientist role for illustration:\n\n"
            + "\n".join(["- " + q for q in example_questions])
        )

    return role_example_questions + "\n\n" + instructions


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
