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
        
    async def generate_ts_task(self, original_message: TSTaskMessage, ctx: MessageContext, message_class: Optional[bool | BaseModel] = False) -> TSMessage:
        """Generates a time series task message based on the original message.

        Args:
            original_message (TSTaskMessage): The original message containing the task description and filepath.

        Returns:
            TSMessage: A new TSMessage with the task type and description.
        """
        ts_message = SystemMessage(
            source="user",
            content=f"""
            The task for the time series analysis is: {original_message.description}.
            The time-series data is stored at: {original_message.filepath}.
            Based on the task description, determine which one of the following types of analysis best matches the requirement:

            Task Type 1: The tasks asks a question where analyzing the properties of the time-series would be necessary for a classification task.
            Task Type 2: "ts-forecasting": The task asks a question where predicting the future values of the time-series based on the historical data would be necessary.
            Task Type 3: "anomaly-detection": The task asks a question where detecting anomalies in the time-series data would be necessary.

            Only reply with one of three numbers representing the task type: [1, 2, 3].
            Do not explain your reasoning.
            Do not add any extra text.
            Only output the chosen task type.
            """
        )

        response_content = await self.send_message_to_openai([ts_message], ctx, json_output=message_class)

        response = response_content.strip()
        task_type = "ts-classifiation"
        if response == "1":
            task_type = "ts-classification"
        elif response == "2":
            task_type = "ts-forecasting"
        elif response == "3":
            task_type = "anomaly-detection"
        
        try:
            if message_class:
                ts_task = TSMessage.model_validate_json(response_content)
                ts_task.source = "planner"  # Set the source to the Planning Agent
                ts_task.filepath = original_message.filepath
            else:
                ts_task = TSMessage(source="planner", filepath=original_message.filepath, task_type=task_type, description=original_message.description)
            
            # Send the generated task to the QA Agent
            return ts_task
        except ValidationError as e:
            raise ValueError(f"Response content is not a valid TextMessage: {e}") from e
        
    async def generate_qa_task(self, original_message: TSTaskMessage, ctx: MessageContext, message_class: Optional[bool | BaseModel] = False) -> TextMessage:
        """Generates a QA task message based on the original message.

        Args:
            original_message (TSTaskMessage): The original message containing the task description and filepath.

        Returns:
            TextMessage: A new TextMessage with the task description.
        """
        task_message = SystemMessage(
            source="user",
            content=f"""You are given the following original task description: {original_message.description}.  
            The time-series data is stored at: {original_message.filepath}.  

            Your task is to generate a **non-trivial multiple-choice question and answer task** that accomplishes the original goal described above.

            **Instructions:**
            1. Think carefully about what the task is asking.
            2. Write a **clear, concise multiple-choice question** relevant to the task.
            3. Include **at least two numbered answer options**, starting from **1**.
            4. If the original task description already defines specific class labels or meanings for numbered outputs,
            you must reuse those exact classes as your multiple-choice answer options.
            5. **Do NOT include the correct answer.**
            6. Clearly state that the goal is for the user to answer the question.
            7. Do not bold any text in the question or answer options.
            8. **Require the user to respond in this exact format (copy this exactly): 
                Reason: <brief reasoning>
                Answer: <number of the correct answer option>**
            """
        )

        response_content = await self.send_message_to_openai([task_message], ctx, json_output=message_class)

        try:
            if message_class:
                qa_task = message_class.model_validate_json(response_content)
                qa_task.source = "planner"  # Set the source to the Planning Agent
            else:
                qa_task = TextMessage(source="planner", content=response_content)
            
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
        qa_task = await self.generate_qa_task(message, ctx, False)
        print(f"[{self.name}] Sending QA task to QA Agent...")
        await self.publish_message(
            qa_task,
            TopicId(type="Planner-QA", source=self.id.key)
        )

        ts_task = await self.generate_ts_task(message, ctx, False)
        print(f"[{self.name}] Sending TS task to TS Agent...")
        await self.publish_message(
            ts_task,
            TopicId(type="Planner-TS", source=self.id.key)
        )