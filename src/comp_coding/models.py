"""Shared Pydantic models for the comp_coding pipeline."""

from typing import List, Optional
from pydantic import BaseModel


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
    # All question_ids that map to this problem
    question_ids: List[str] = []
    # List of samples for this problem
    samples: List[Sample] = []
