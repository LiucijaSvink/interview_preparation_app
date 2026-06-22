"""Pydantic data schemas describing the shape of the LLM's JSON responses."""

from typing import List, Literal

from pydantic import BaseModel


class ValidationOutput(BaseModel):
    job_position: str
    validation_result: Literal["yes", "no", "uncertain"]


class InterviewQuestion(BaseModel):
    question: str


class InterviewOutput(BaseModel):
    job_position: str
    interview_type: str
    questions: List[InterviewQuestion]
