# Interview Preparation Assistant

An AI-powered application that helps you prepare for different types of job interviews by generating relevant questions based on the job position and interview type.

## Features

- **Job Position Validation**: Validates job titles to ensure they are professional and realistic
- **Multiple Interview Types**: Supports different interview scenarios:
  - HR Screening
  - Technical Interview
  - Manager Interview
  - General Interview (for unspecified interviewers)
- **Customizable Question Generation**: Adjust model parameters to control question style and diversity
- **Question History**: Tracks previously generated questions to avoid repetition
- **Interactive Interface**: User-friendly web interface built with Streamlit

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
   - Create a `.streamlit/secrets.toml` file
   - Add your OpenAI API key:
```toml
OPENAI_API_KEY = "your-api-key-here"
```

## Usage

1. Run the application:
```bash
streamlit run interview_prep_app.py
```

2. Enter the job position you're applying for

3. Select the type of interview you want to practice for

4. Adjust model parameters (optional):
   - Temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
   - Max Tokens: Maximum length of generated questions
   - Top P: Controls diversity of questions
   - Frequency Penalty: Reduces repetition
   - Presence Penalty: Encourages topic diversity

5. Click "Generate Question" to get interview questions

6. View your question history and clear it when needed

## Model Parameters

- **Temperature (0.6 default)**: Higher values make questions more creative, lower values make them more focused
- **Max Tokens (1000 default)**: Controls the length of generated questions
- **Top P (0.8 default)**: Controls diversity of question types
- **Frequency Penalty (0.4 default)**: Reduces repetition of similar questions
- **Presence Penalty (0.2 default)**: Encourages covering different topics

## Requirements

- Python 3.8+
- streamlit==1.44.1
- openai==1.73.0
- pydantic==2.11.3

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the language model API
- Streamlit for the web framework
- Pydantic for data validation 
