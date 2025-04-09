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


@type_subscription(topic_type="Planner-TS") # for receiving task from Planner
@type_subscription(topic_type="Reward-TS") # for receiving TS Feedback
class TSAgent(RoutedAgent):
    def __init__(self, name: str, model_client: Optional[ChatCompletionClient] = None):
        super().__init__(description=f"{name} with LTSM Package support")
        self.name = name
        self._last_plan: Optional[str] = None
        self._model_client = model_client
        self._last_ts_response: Optional[str] = None  # for evaluation

    @message_handler
    async def handle_TS(self, message: TSMessage, ctx: MessageContext) -> None:
        """This is the TS info given by Planner. LTSM will process the TS data and return the answer.
        """
        file_path = message.filepath
        task_type = message.task_type 

        ts_response = inference(
            file=file_path, 
            task_type=task_type
        )
        

        # publish
        await self.publish_message(TSMessage(source=self.name,
                                             filepath = ts_response,
                                             task_type="ts-classification"), TopicId(type="TS-Info", source=self.id.key))

    def get_last_response(self) -> Optional[str]:
        return self._last_ts_response

