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
from pydantic import BaseModel
from .custom_messages import TextMessage, TSMessage
from multi_agents_pipeline.ltsm_inference import inference


@type_subscription(topic_type="Planner-TS") # for receiving task from Planner
@type_subscription(topic_type="Redo-TS") # for receiving TS Feedback
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
        if task_type == "ts-classification":
            description = await self.write_ts_description(file_path, message.description, ctx)
            print(f"[{self.name}] Generated TS classification response: {description}")

            await self.publish_message(TSMessage(source=self.name,
                                             filepath = file_path,
                                             description = description,
                                             task_type=task_type), TopicId(type="TS-Info", source=self.id.key))
        elif task_type == "ts-forecasting":
            ts_response = inference(
                file=file_path, 
                task_type=task_type
            )

            description = await self.write_ts_prediction_description(ts_response, message.description, ctx)
            print(f"[{self.name}] Generated TS forecasting response: {description}")

            await self.publish_message(TSMessage(source=self.name,
                                             filepath = ts_response,
                                             description = description,
                                             task_type=task_type), TopicId(type="TS-Info", source=self.id.key))
        elif task_type == "anomaly-detection":
            # TODO: Implement anomaly detection logic
            pass
            
    async def write_ts_description(self, file_path: str, original_message:str, ctx: MessageContext) -> None:
        data = pd.read_csv(Path(file_path)).to_csv(index=False)
        message = SystemMessage(
            source="user",
            content=f"""You are a domain expert reviewing a prediction made by an expert time-series analyst. The description of the data and problem is as follows: {original_message}
            The prediction is provided here as a comma-separated list of numerical values: {data}.
            Your goal is to interpret this sequence by identifying and describing key statistical and temporal patterns.
            As you analyze the data, consider the following:
            1. Descriptive Statistics (mean, mode, median, standard deviation, etc.)
            2. Temporal Trends (increasing/decreasing patterns, periodicity, etc.)
            3. Seasonality and Cyclic Behavior
            4. Anomalies or Outliers
            Focus on interpreting patterns rather than listing every number.
            """
        )

        response = await self._model_client.create(
            messages=[message],
            cancellation_token=ctx.cancellation_token)
        
        if isinstance(response.content, str):
            return response.content
        else:
            raise ValueError("Response content is not a valid string")
        
        
    async def write_ts_prediction_description(self, ts_response: str, original_message: str, ctx: MessageContext) -> None:
        prediction = pd.read_csv(Path(ts_response)).to_csv(index=False)
        message = SystemMessage(
            source="user",
            content=f"""You are a domain expert reviewing a prediction made by an expert time-series analyst. The description of the data and problem is as follows: {original_message}
            The prediction is provided here as a comma-separated list of numerical values: {prediction}.
            Your goal is to interpret this sequence by identifying and describing key statistical and temporal patterns.
            As you analyze the data, consider the following:
            1. Descriptive Statistics (mean, mode, median, standard deviation, etc.)
            2. Temporal Trends (increasing/decreasing patterns, periodicity, etc.)
            3. Seasonality and Cyclic Behavior
            4. Anomalies or Outliers
            Focus on interpreting patterns rather than listing every number.
            """
        )

        response = await self._model_client.create(
            messages=[message],
            cancellation_token=ctx.cancellation_token)
        
        if isinstance(response.content, str):
            return response.content
        else:
            raise ValueError("Response content is not a valid string")

    def get_last_response(self) -> Optional[str]:
        return self._last_ts_response

