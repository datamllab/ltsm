 
 # Quick Command
 The command `CUDA_VISIBLE_DEVICES=1,2,3 uvicorn llm-server:app  --port <port number> --reload` should be run in the `multi_agents_pipeline` directory. e.g. `CUDA_VISIBLE_DEVICES=1,2,3 uvicorn llm-server:app --reload` will run the FastAPI app on http://127.0.0.1:8000/. 

`lsof -i :8000` can be used to check the running local LLM.

To execute the full pipeline, go to the `multi_agents_pipeline` folder and run `python main.py`.

> To use LLama-3-8B-Instruct, please check transformers >= 4.40! 

# Messages and Communication

```python
from pydantic import BaseModel
from typing import Optional, List


class TextMessage(BaseModel):
    """
    pass QA related message"""
    source: str
    content: str

class TSMessage(BaseModel):
    """
    passed from Planner to TS Agent, and from TS Agent to QA Agent
    
    filepath should be a valid path to a csv/tsv file""" 
    source: str
    filepath: str # TO DO : Sopport more possible types
    task_type:Optional[str] = None
    description: Optional[str] = None
```

! To Discuss ! : Planner publishes messages with topic "Planner-QA"(`TextMessage`), "Planner-TS"(`TSMessage`)

TS Agent publishes messages with topic "TS-Info"(`TSMessage`), subscribes "Planner-TS"(`TSMessage`) and "Reward-TS"(`TSMessage`), 

QA Agent publishes messages with topic "QA-Response"(`TextMessage`), subscribes ""Planner-QA"(`TextMessage`)", "TS-Info"(`TSMessage`) and "Reward-QA"(`TextMessage`), 

Reward Agent publishes messages with topic "Reward-QA"(`TextMessage`), , "Reward-TS"(`TSMessage`), and subscribes "TS-Info"(`TSMessage`), "QA-Response"(`TextMessage`).

# Agents

## TS Agent

Handle TSMessage, use LTSM to inference

## QA Agent

Combine TS Info and Planner-QA, get the response of LLM

## Reward Agent

gather output of TS Agent and QA Agent


# Question:
1. context buffer size : decided by the reward agent?

2. should reward give signal to planner to handle the next query?

3. dataset selection (Task selection) : forecasting ? classification ? Answer: Time Reasoning


TODO List (April 7, 2025)
- performance of the framework
- remove TS, test the performance of 
- use different TS models based on the task





