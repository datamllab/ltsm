from pydantic import BaseModel
from typing import Optional, List


class TextMessage(BaseModel):
    """
    passed from Planner to QA Agent"""
    source: str
    content: str
    task: Optional[str] = None

class TSMessage(BaseModel):
    """
    passed from Planner to TS Agent, and from TS Agent to QA Agent
    
    filepath should be a valid path to a csv/tsv file"""
    source: str
    filepath: str
    task_type:Optional[str] = None
    description: Optional[str] = None