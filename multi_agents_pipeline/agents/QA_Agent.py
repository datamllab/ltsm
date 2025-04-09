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
from autogen_core.models import ChatCompletionClient, UserMessage, AssistantMessage, SystemMessage
from autogen_core.model_context import BufferedChatCompletionContext
from pydantic import BaseModel
from .custom_messages import TextMessage, TSMessage

@type_subscription(topic_type="Planner-QA") # for receiving task from Planner
@type_subscription(topic_type="Reward-QA") # for receiving QA Feedback 
@type_subscription(topic_type="TS-Info")  # for receiving TS info from TS Agent
class QAAgent(RoutedAgent):
    def __init__(self, name: str, model_client: ChatCompletionClient):
        super().__init__(description=f"{name} with LLM support")
        self.name = name
        self._last_plan: Optional[str] = None
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)
        self._system_messages = [SystemMessage(content="You are a helpful AI assistant.")]
        
        self._last_llm_response: Optional[str] = None  # for evaluation

    @message_handler
    async def handle_plan(self, message: TextMessage, ctx: MessageContext) -> None:
        self._last_plan = message.content
        print(f"[{self.name}] Stored plan from {message.source}: {message.content}")

    @message_handler
    async def handle_TS(self, message: TSMessage, ctx: MessageContext) -> None:
        """This is the TS info given by TS Agent
        """
        df = pd.read_csv(Path(message.filepath))
        stats = df.describe().to_string()

        # below is the prompt that combine the task and the TS Info.
        # TODO : Modify according to the task type and task description. Currently just a placeholder
        prompt = f"""
            You are a Time Serise Expert.

            Here is a task given by the planner: 
            {self._last_plan or "(no plan received)"}

            Here is the output of Time-Series Agent:
            {stats}

            Please finish the task based on the above information.
            """

        print(f"[{self.name}] Sending prompt to LLM...")

        user_message = UserMessage(content=prompt, source=self.name)
        await self._model_context.add_message(user_message)

        # send to LLM for response
        llm_response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,  
        )

        assert isinstance(llm_response.content, str)
        
        self._last_llm_response = llm_response.content  

        await self._model_context.add_message(
            AssistantMessage(content=self._last_llm_response, source=self.name)
        )
        # publish the inference result of QA Agent
        await self.publish_message(
            TextMessage(source=self.name, content=self._last_llm_response),
            TopicId(type="QA-Response", source=self.id.key)  # publish to a specific topic for QA response
        )

    def get_last_response(self) -> Optional[str]:
        return self._last_llm_response
