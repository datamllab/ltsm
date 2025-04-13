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


@type_subscription(topic_type="TS-Info")  # for receiving TS info from TS Agent
@type_subscription(topic_type="QA-Response")  # for receiving QA response to process further
class RewardAgent(RoutedAgent):
    """TODO: verify the output of TS Agent and QA Agent. Then provide feedback.
            Attributes (in self):
            ---------------------
            name : str
                The name of the agent (used for logging and identification).
            _last_qa : Optional[str]
                The latest QA agent response content.
            _last_ts : Optional[TSMessage]
                The latest time-series analysis result from TS agent.
            _last_plan : Optional[str]
                The task instruction sent from the planner agent.
            _model_client : Optional[ChatCompletionClient]
                The LLM backend client used for scoring the QA response.
            _model_context : BufferedChatCompletionContext
                Maintains message history for consistent LLM conversation context.
            _system_messages : List[SystemMessage]
                System prompts prepended to each LLM call to define assistant behavior.
            _retry_count : int
                Counter to track how many times a redo has been triggered.
            _retry_limit : int
                Maximum allowed retry attempts before giving up.
            _force_bad_score : bool (for Testing)
                If True, forces a low reward score for testing the retry logic.
    """
    def __init__(self, name: str, model_client: Optional[ChatCompletionClient] = None, force_bad_score = False) -> None:
        super().__init__(description=f"{name} with LLM support")
        self.name = name
        self._last_qa: Optional[str] = None
        self._last_ts: Optional[TSMessage] = None
        self._last_plan: Optional[str] = None
        self._model_client = model_client
        self._model_context = BufferedChatCompletionContext(buffer_size=5)
        self._system_messages = [SystemMessage(content="You are a helpful AI assistant.")]
        self._retry_count = 0
        self._retry_limit = 2
        self._force_bad_score = force_bad_score

    @message_handler
    async def handle_TS(self, message: TSMessage, ctx: MessageContext) -> None:
        """This is the TS info given by Planner. LTSM will process the TS data and return the answer.
        """
        file_path = message.filepath
        task_type = message.task_type 
        self._last_ts = message

        print(f"[{self.name}] Received TS info from TS Agent: {file_path}, task_type: {task_type}")
        await self.try_process_reward(ctx)
    
    @message_handler
    async def handle_QA(self, message: TextMessage, ctx: MessageContext) -> None:
        """This is the QA response to process further for reward calculation.
        """
        self._last_qa = message.content
        self._last_plan = message.task
        print(f"[{self.name}] Received QA response: {self._last_qa}")
        await self.try_process_reward(ctx)
    
    async def try_process_reward(self, ctx: MessageContext) -> None:
        if not self._last_ts or not self._last_qa:
            return  # wait for both messages
        
        # TODO: read TS info from TS Message
        prompt = self.build_evaluation_prompt(self._last_ts, self._last_qa, self._last_plan or "")

        print(f"[{self.name}] Sending evaluation prompt to LLM...")

        user_message = UserMessage(content=prompt, source=self.name)
        await self._model_context.add_message(user_message)

        reward_response = await self._model_client.create(
            self._system_messages + (await self._model_context.get_messages()),
            cancellation_token=ctx.cancellation_token,  
        )
        
        if isinstance(reward_response.content, str):

            await self._model_context.add_message(
                AssistantMessage(content=reward_response.content, source=self.name)
            )

            score = self.extract_score(reward_response.content)
            if self._force_bad_score:
                score = 0.1
            print(f"[{self.name}] LLM evaluation score: {score}\nDetails:\n{reward_response.content}")

            if score > 0.5:
                print(f"[{self.name}] Reward score is satisfactory. No redo needed.\nScore:{score}.\nResponse: {reward_response.content}")
            elif score<= 0.5 and self._retry_count < self._retry_limit:
                self._retry_count += 1
                print(f"[{self.name}] Reward score is unsatisfactory. Sending redo request... Attempt: {self._retry_count}")
                await self.send_redo_request(ctx)
            else:
                print(f"[{self.name}] Retry limit reached ({self._retry_limit}). No further redo will be issued.")


        else:
            print(f"[{self.name}] Unexpected LLM response type: {type(reward_response.content)}")

    def build_evaluation_prompt(self, ts_summary: str, qa_answer: str, plan: str) -> str:
        return f"""
        A planner issued the following task:
        {plan}

        The time-series agent produced this output (summary of data):
        {ts_summary}

        The QA agent responded:
        "{qa_answer}"

        Evaluate if the response correctly reflects the data and fulfills the task.
        Give a reward score from 0 to 1, and briefly explain your reasoning.
        """

    def extract_score(self, text: str) -> float:
        import re
        match = re.search(r"([0-1](?:\.\d+)?)", text)
        if match:
            return float(match.group(1))
        return 0.0  # fallback
    
    async def send_redo_request(self, ctx: MessageContext) -> None:
        msg_ts = TSMessage(source=self.name, filepath=self._last_ts.filepath, task_type="redo-ts")
        msg_qa = TextMessage(source=self.name, content="Please reprocess. Previous result was not satisfactory.")

        # send to TS Agent
        await self.publish_message(msg_ts, TopicId(type="Redo-TS", source=self.id.key))  

        # send to QA Agent
        await self.publish_message(msg_qa,TopicId(type="Redo-QA", source=self.id.key))  

        self._last_plan = None
        self._last_ts = None
        self._last_qa = None

        print(f"[{self.name}] Sent redo request to TS Agent and QA Agent.")