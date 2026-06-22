"""OpenAI client setup and model parameter helpers."""

from typing import Dict, Union

import streamlit as st
from openai import OpenAI


def get_api_key(key_name: str = "OPENAI_API_KEY") -> str:
    return st.secrets[key_name]


def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def set_openai_params(
    model: str = "gpt-4o",
    temperature: float = 0,
    max_tokens: int = 1000,
    top_p: float = 1,
    frequency_penalty: float = 0,
    presence_penalty: float = 0,
) -> Dict[str, Union[str, int, float]]:
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }


# Shared client instance and default parameters for the validation step.
client = get_openai_client(api_key=get_api_key("OPENAI_API_KEY"))

validator_params: Dict[str, Union[str, int, float]] = {
    "model": "gpt-4o",
    "temperature": 0,
    "max_tokens": 100,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}
