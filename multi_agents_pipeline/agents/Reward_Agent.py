import pandas as pd
from pathlib import Path
from typing import Optional, List

from autogen_core import (
    RoutedAgent,
    message_handler,
    default_subscription,
    MessageContext,
    DefaultTopicId,
    TopicId,
    type_subscription
)
from autogen_core.models import ChatCompletionClient, UserMessage, AssistantMessage
from pydantic import BaseModel
from .custom_messages import TextMessage, TSMessage
from multi_agents_pipeline.ltsm_inference import inference


@type_subscription(topic_type="TS-Info")  # for receiving TS info from TS Agent
@type_subscription(topic_type="QA-Response")  # for receiving QA response to process further
class RewardAgent(RoutedAgent):
    """TODO: verify the output of TS Agent and QA Agent. Then provide feedback.
    """
    def __init__(self, name: str):
        super().__init__(description=f"{name} with LLM support")
        self.name = name
        self._last_qa: Optional[str] = None
        self._last_ts: None
        # self._model_client = model_client

    @message_handler
    async def handle_TS(self, message: TSMessage, ctx: MessageContext) -> None:
        """This is the TS info given by Planner. LTSM will process the TS data and return the answer.
        """
        file_path = message.filepath
        task_type = message.task_type 

        print(f"[{self.name}] Received TS info from TS Agent: {file_path}, task_type: {task_type}")
    
    @message_handler
    async def handle_QA(self, message: TextMessage, ctx: MessageContext) -> None:
        """This is the QA response to process further for reward calculation.
        """
        self._last_qa = message.content
        print(f"[{self.name}] Received QA response: {self._last_qa}")

        



