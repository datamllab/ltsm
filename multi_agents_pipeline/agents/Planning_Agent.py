from .custom_messages import TextMessage, TSTaskMessage, TSMessage
from typing import Optional, List 
from autogen_core import RoutedAgent, default_subscription, message_handler, MessageContext, TopicId
from autogen_core.models import ChatCompletionClient, SystemMessage
from pydantic import ValidationError
from pydantic import BaseModel


@default_subscription
class PlanningAgent(RoutedAgent):
    """A planning agent that uses OpenAI API to generate tasks for a Time Series Agent and QA Agent.

    Args:
        name (str): The name of the agent.
        model_client (ChatCompletionClient): The ChatCompletion client.
    """
    def __init__(self, name: str, model_client: ChatCompletionClient) -> None:
        super().__init__("planning_agent")
        self.name = name
        self._model_client = model_client
        self._system_messages = [SystemMessage(content="You are a helpful AI assistant.")]

    async def send_message_to_openai(self, messages: List[SystemMessage], ctx: MessageContext, json_output: Optional[bool | BaseModel] = False) -> str:
        """Sends messages to OpenAI and returns the response content.

        Args:
            messages (List[SystemMessage]): The list of messages to send to OpenAI.

        Returns:
            str: The response content from OpenAI.
        """
        response = await self._model_client.create(
            messages=self._system_messages + messages,
            cancellation_token=ctx.cancellation_token,  
            json_output=json_output)
        if isinstance(response.content, str):
            return response.content
        else:
            raise ValueError("Response content is not a valid JSON string")
        
    async def generate_ts_task(self, original_message: TSTaskMessage, ctx: MessageContext) -> TSMessage:
        """Generates a time series task message based on the original message.

        Args:
            original_message (TSTaskMessage): The original message containing the task description and filepath.

        Returns:
            TSMessage: A new TSMessage with the task type and description.
        """
        ts_message = SystemMessage(
            source="user",
            content=f"""The task for the time series analysis is: {original_message.description}. 
            The time-series data is stored at {original_message.filepath}. Provide a detailed description of the data
            based on the task description. Also, provide what type of analysis would be required to complete the task among
            the following types: ["statistical forecasting", "anomaly detection"].
            """
        )

        response_content = await self.send_message_to_openai([ts_message], ctx, json_output=TSMessage)

        try:
            ts_task = TSMessage.model_validate_json(response_content)
            ts_task.source = "planner"  # Set the source to the Planning Agent
            ts_task.filepath = original_message.filepath  # Ensure the filepath is preserved
            # Send the generated task to the QA Agent
            return ts_task
        except ValidationError as e:
            raise ValueError(f"Response content is not a valid TextMessage: {e}") from e
        
    async def generate_qa_task(self, original_message: TSTaskMessage, ctx: MessageContext) -> TextMessage:
        """Generates a QA task message based on the original message.

        Args:
            original_message (TSTaskMessage): The original message containing the task description and filepath.

        Returns:
            TextMessage: A new TextMessage with the task description.
        """
        task_message = SystemMessage(
            source="user",
            content=f"""Write a descriptive task for the following prompt: {original_message.description}. 
            The time-series data is stored at {original_message.filepath}.
            """
        )

        response_content = await self.send_message_to_openai([task_message], ctx, json_output=TextMessage)

        try:
            qa_task = TextMessage.model_validate_json(response_content)
            qa_task.source = "planner"  # Set the source to the Planning Agent
            # Send the generated task to the QA Agent
            return qa_task
        except ValidationError as e:
            raise ValueError(f"Response content is not a valid TextMessage: {e}") from e

    @message_handler
    async def handle_ts_task_message(self, message: TSTaskMessage, ctx: MessageContext) -> None:
        """Handles incoming time series task messages and generates a response using the OpenAI Assistant API.

        Args:
            message (TSTaskMessage): The incoming message containing the user's query.
        """
        ts_task = await self.generate_ts_task(message, ctx)
        print(f"[{self.name}] Sending TS task to TS Agent...")
        await self.publish_message(
            ts_task,
            TopicId(type="Planner-TS", source=self.id.key)
        )
        #await self.send_message(ts_task, AgentId("ts_agent", "default"))

        qa_task = await self.generate_qa_task(message, ctx)
        print(f"[{self.name}] Sending QA task to QA Agent...")
        await self.publish_message(
            qa_task,
            TopicId(type="Planner-QA", source=self.id.key)
        )
        #await self.send_message(qa_task, AgentId("qa_agent", "default"))