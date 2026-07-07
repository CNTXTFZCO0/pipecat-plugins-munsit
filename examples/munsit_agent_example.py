"""Foundational example: Arabic conversational AI agent, all-Munsit voice stack.

Pipeline data flow:
  Microphone -> Daily Transport -> Munsit STT -> OpenAI LLM -> Munsit TTS -> Daily Transport -> Speaker

Requirements:
  pip install pipecat-plugins-munsit "pipecat-ai[daily,openai,silero]" python-dotenv loguru

Environment variables:
  DAILY_API_KEY   - Daily API key (https://dashboard.daily.co)
  OPENAI_API_KEY  - OpenAI API key
  MUNSIT_API_KEY  - Munsit API key (get at https://app.munsit.com)
"""

import asyncio
import os

import aiohttp
from dotenv import load_dotenv

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndFrame, LLMMessagesUpdateFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.daily.transport import DailyParams, DailyTransport

from pipecat_plugins_munsit import MunsitSTTService, MunsitTTSService

load_dotenv(override=True)


async def main():
    async with aiohttp.ClientSession() as session:
        # Auto-create a Daily room
        from pipecat.transports.daily.utils import DailyRESTHelper, DailyRoomParams

        daily_helper = DailyRESTHelper(
            daily_api_key=os.getenv("DAILY_API_KEY", ""),
            aiohttp_session=session,
        )
        room = await daily_helper.create_room(DailyRoomParams())
        token = await daily_helper.get_token(room.url)

        transport = DailyTransport(
            room.url,
            token,
            "Munsit Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=24000,
            ),
        )

        stt = MunsitSTTService(api_key=os.getenv("MUNSIT_API_KEY"))
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY", ""), model="gpt-4o")
        tts = MunsitTTSService(
            api_key=os.getenv("MUNSIT_API_KEY"),
            aiohttp_session=session,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful Arabic-speaking assistant. Respond in Arabic.",
            }
        ]
        context = LLMContext(messages=messages)
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        )

        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_joined(transport, participant):
            await task.queue_frames([LLMMessagesUpdateFrame(messages, run_llm=True)])

        @transport.event_handler("on_participant_left")
        async def on_left(transport, participant, reason):
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
