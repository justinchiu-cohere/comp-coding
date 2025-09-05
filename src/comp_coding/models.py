"""Shared Pydantic models for the comp_coding pipeline."""

from typing import List, Optional
from pydantic import BaseModel, Field


class TestCase(BaseModel):
    stdin: str
    stdout: str


class ScenarioConfig(BaseModel):
    prompt: str
    tests: List[TestCase]
    test_type: str = "stdinout"
    reward_type: str = "binary"
    execution_server_url: str = "http://codeserver-service.default:80"


class Sample(BaseModel):
    """Individual sample from OCR2 dataset."""

    r1_generation: Optional[str] = None
    qwq_critique: Optional[str] = None
    solution: Optional[str] = None
    judgement: Optional[str] = None
    pass_rate: Optional[float] = None
    source: Optional[str] = None
    license: Optional[str] = None
    difficulty: Optional[str] = None
    id: Optional[str] = None
    question_id: Optional[str] = None


class Problem(BaseModel):
    """Problem with test cases and multiple samples."""

    env_name: str = "code"
    scenario_config: ScenarioConfig
    # Problem metadata
    dataset: Optional[str] = None
    split: Optional[str] = None
    index: Optional[int] = None
    # The unique question_id that maps to this problem
    question_id: Optional[str] = None
    # List of samples for this problem
    samples: List[Sample] = Field(default_factory=list)


# Training format models


class SFTSample(BaseModel):
    """SFT training format with prompt and solution."""

    prompt: str
    solution: str


class RLProblem(BaseModel):
    """RL training format following kylie.json structure."""

    env_name: str = "code"
    scenario_config: ScenarioConfig


class RLProblemWithCompletions(BaseModel):
    """RL training format with sample completions for Luffy configuration."""

    env_name: str = "code"
    scenario_config: ScenarioConfig
    completions: List[str]  # List of sample solutions for this problem
