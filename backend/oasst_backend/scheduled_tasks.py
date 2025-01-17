from __future__ import absolute_import, unicode_literals

from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID, uuid4

from asgiref.sync import async_to_sync
from celery import shared_task
from loguru import logger
from oasst_backend.api.v1.utils import prepare_conversation
from oasst_backend.celery_worker import app
from oasst_backend.models import ApiClient
from oasst_backend.prompt_repository import PromptRepository, UserRepository
from oasst_backend.task_repository import TaskRepository
from oasst_backend.user_repository import User
from oasst_backend.utils.database_utils import default_session_factory
from oasst_backend.utils.hugging_face import HfClassificationModel, HfEmbeddingModel, HfUrl, HuggingFaceAPI
from oasst_backend.utils.open_ai import OpenAIAPI, OpenAIUrl, OpenAIChatModel
from oasst_shared.schemas import protocol as protocol_schema
from oasst_shared.utils import utcnow
from sqlmodel import select

startup_time: datetime = utcnow()


async def useHFApi(text, url, model_name):
    hugging_face_api: HuggingFaceAPI = HuggingFaceAPI(f"{url}/{model_name}")
    result = await hugging_face_api.post(text)
    return result

async def useOpenAIApi(text, url, model):
    openai_api: OpenAIAPI = OpenAIAPI(url, model)
    result = await openai_api.post(text)
    return result

@app.task(name="toxicity")
def toxicity(text, message_id, api_client):
    try:
        logger.info(f"checking toxicity : {api_client}")

        with default_session_factory() as session:
            model_name: str = HfClassificationModel.TOXIC_ROBERTA.value
            url: str = HfUrl.HUGGINGFACE_TOXIC_CLASSIFICATION.value
            toxicity: List[List[Dict[str, Any]]] = async_to_sync(useHFApi)(text=text, url=url, model_name=model_name)
            toxicity = toxicity[0][0]
            logger.info(f"toxicity from HF {toxicity}")
            api_client_m = ApiClient(**api_client)
            if toxicity is not None:
                pr = PromptRepository(db=session, api_client=api_client_m)
                pr.insert_toxicity(
                    message_id=message_id, model=model_name, score=toxicity["score"], label=toxicity["label"]
                )
            session.commit()

    except Exception as e:
        logger.error(f"Could not compute toxicity for text reply to {message_id=} with {text=} by.error {e}")

@app.task(name="openai_gpt")
def openai_gpt(text: str, message_id: UUID, api_client: Dict[str, Any], user: Dict[str, Any], model: OpenAIChatModel):
    try:
        logger.info(f"replying with gpt3.5-turbo : {api_client}")
        
        with default_session_factory() as session:
            replies = async_to_sync(useOpenAIApi)(text=text, url=OpenAIUrl.OPEN_AI_CHAT.value, model=model)
            replies = replies["choices"]
            logger.info(f"replies from OpenAI {replies}")
            api_client_m = ApiClient(**api_client)
            
            logger.info(f"api_client_m {api_client_m}")
            if replies is not None:
                gpt_user = protocol_schema.User(id="__GPT3.5__", display_name="GPT", auth_method="local")
                ur = UserRepository(db=session, api_client=api_client_m)
                tr = TaskRepository(db=session, api_client=api_client_m, client_user=gpt_user, user_repository=ur)
                pr = PromptRepository(db=session, api_client=api_client_m, client_user=gpt_user, user_repository=ur, task_repository=tr)
                for reply in replies:
                    message = pr.fetch_message(message_id=message_id)
                    messages = pr.fetch_message_conversation(message)
                    conversation = prepare_conversation(messages)
                    task = protocol_schema.AssistantReplyTask(conversation=conversation)
                    parent_message_id = messages[-1].id
                    message_tree_id = messages[-1].message_tree_id
                    pr.task_repository.store_task(task, message_tree_id, parent_message_id, collective=False)
                
                    frontend_message_id = str(uuid4())
                    pr.task_repository.bind_frontend_message_id(task_id=task.id, frontend_message_id=frontend_message_id)
                
                    message = pr.store_text_reply(
                        text=reply['message']['content'], 
                        lang="en", 
                        frontend_message_id=frontend_message_id, 
                        user_frontend_message_id=frontend_message_id, 
                        check_duplicate=False
                    )
                
                session.commit()
    except Exception as e:
        logger.error(f"Could not response for text reply to {message_id=} with {text=} by.error {e}")
        
@app.task(name="hf_feature_extraction")
def hf_feature_extraction(text, message_id, api_client):
    try:
        with default_session_factory() as session:
            model_name: str = HfEmbeddingModel.MINILM.value
            url: str = HfUrl.HUGGINGFACE_FEATURE_EXTRACTION.value
            embedding = async_to_sync(useHFApi)(text=text, url=url, model_name=model_name)
            api_client_m = ApiClient(**api_client)
            if embedding is not None:
                logger.info(f"emmbedding from HF {len(embedding)}")
                pr = PromptRepository(db=session, api_client=api_client_m)
                pr.insert_message_embedding(
                    message_id=message_id, model=HfEmbeddingModel.MINILM.value, embedding=embedding
                )
                session.commit()

    except Exception as e:
        logger.error(f"Could not extract embedding for text reply to {message_id=} with {text=} by.error {str(e)}")


@shared_task(name="update_user_streak")
def update_user_streak() -> None:
    logger.info("update_user_streak start...")
    try:
        with default_session_factory() as session:
            current_time = utcnow()
            timedelta = current_time - startup_time
            if timedelta.days > 0:
                # Update only greater than 24 hours . Do nothing
                logger.info("Process timedelta greater than 24h")
                statement = select(User)
                result = session.exec(statement).all()
                if result is not None:
                    for user in result:
                        last_activity_date = user.last_activity_date
                        streak_last_day_date = user.streak_last_day_date
                        # set NULL streak_days to 0
                        if user.streak_days is None:
                            user.streak_days = 0
                        # if the user had completed a task
                        if last_activity_date is not None:
                            lastactitvitydelta = current_time - last_activity_date
                            # if the user missed consecutive days of completing a task
                            # reset the streak_days to 0 and set streak_last_day_date to the current_time
                            if lastactitvitydelta.days > 1 or user.streak_days is None:
                                user.streak_days = 0
                                user.streak_last_day_date = current_time
                        # streak_last_day_date has a current timestamp in DB. Idealy should not be NULL.
                        if streak_last_day_date is not None:
                            streak_delta = current_time - streak_last_day_date
                            # if user completed tasks on consecutive days then increment the streak days
                            # update the streak_last_day_date to current time for the next calculation
                            if streak_delta.days > 0:
                                user.streak_days += 1
                                user.streak_last_day_date = current_time
                        session.add(user)
                        session.commit()

            else:
                logger.info("Not yet 24hours since the process started! ...")
        logger.info("User streak end...")
    except Exception as e:
        logger.error(str(e))
    return
