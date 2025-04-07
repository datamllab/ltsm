import asyncio
import logging
from autogen_core import AgentId, SingleThreadedAgentRuntime, TopicId
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import (
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from agents.QA_Agent import QAAgent
from agents.TS_Agent import TSAgent
from agents.Reward_Agent import RewardAgent
from agents.custom_messages import TextMessage, TSMessage
from autogen_core import TRACE_LOGGER_NAME
import aiofiles
import yaml

QA_MODEL_CONFIG_PATH = "model_config.yaml"

async def get_model_client(model_config_path: str) -> ChatCompletionClient:
    async with aiofiles.open(model_config_path, "r") as file:
        model_config = yaml.safe_load(await file.read())
    return ChatCompletionClient.load_component(model_config)

async def main() -> None:
    
    runtime = SingleThreadedAgentRuntime()

    model_client = await get_model_client(QA_MODEL_CONFIG_PATH)

    await QAAgent.register(
        runtime,
        "QA_Agent",
        lambda: QAAgent(name="QA_Agent", model_client=model_client),
    )

    # Register the TS Agent
    await TSAgent.register(
        runtime,
        "TS_Agent",
        lambda: TSAgent(name="TS_Agent"),
    )

    # Register the Reward Agent
    await RewardAgent.register(
        runtime,
        "Reward_Agent",
        lambda: RewardAgent(name="Reward_Agent"),
    )

    runtime.start()

    # # mock a plan message from planner
    # await runtime.send_message(
    #     TextMessage(source="user", content="TS classification"),
    #     AgentId("QA_Agent", "default"),
    # )

    # # mock a TS Info message from TS Agent
    # await runtime.send_message(
    #     TSMessage(source="user", filepath="../datasets/UCR-gunpoint/sample_0000.csv",task_type="TS_classification", description="TS Data"),
    #     AgentId("TS_Agent", "default"),
    # )

    await runtime.publish_message(
        TextMessage(source="Planner", content="TS classification"),
        topic_id=TopicId(
            type="Planner-QA",  # This is the topic for Planner to send the initial plan
            source="Planner"
        )
    )

    await runtime.publish_message(
        TSMessage(
            source="Planner",
            filepath="../datasets/UCR-gunpoint/sample_0000.csv",  # Example file path
            task_type="ts-classification",  # Example task type
            description="TS Data"
        ),
        topic_id=TopicId(
            type="Planner-TS",  # This is the topic for TS Agent to send the TS info
            source="Planner"
        )
    )


    await runtime.stop_when_idle()

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("autogen_core").setLevel(logging.WARNING)
    logging.getLogger("autogen_core.events").setLevel(logging.WARNING)
    logging.getLogger("autogen_core.runtime").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger(TRACE_LOGGER_NAME).setLevel(logging.WARNING)
    asyncio.run(main())
