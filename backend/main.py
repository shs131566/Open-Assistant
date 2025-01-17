import json
from datetime import datetime
from http import HTTPStatus
from math import ceil
from pathlib import Path
from typing import Optional
from uuid import uuid4

from oasst_backend.utils.open_ai import OpenAIChatModel
import alembic.command
import alembic.config
import fastapi
import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
from fastapi_utils.tasks import repeat_every
from loguru import logger
from oasst_backend.api.deps import api_auth, create_api_client
from oasst_backend.api.v1.api import api_router
from oasst_backend.api.v1.utils import prepare_conversation
from oasst_backend.cached_stats_repository import CachedStatsRepository
from oasst_backend.config import settings
from oasst_backend.database import engine
from oasst_backend.models import Message, MessageTreeState, message_tree_state
from oasst_backend.prompt_repository import PromptRepository, UserRepository
from oasst_backend.scheduled_tasks import openai_gpt
from oasst_backend.task_repository import TaskRepository, delete_expired_tasks
from oasst_backend.tree_manager import TreeManager, halt_prompts_of_disabled_users
from oasst_backend.user_stats_repository import UserStatsRepository, UserStatsTimeFrame
from oasst_backend.utils.database_utils import CommitMode, managed_tx_function
from oasst_shared.exceptions import OasstError, OasstErrorCode
from oasst_shared.schemas import protocol as protocol_schema
from oasst_shared.utils import utcnow
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sqlmodel import Session
from starlette.middleware.cors import CORSMiddleware

# from worker.scheduled_tasks import create_task

app = fastapi.FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")
startup_time: datetime = utcnow()
    
@app.exception_handler(OasstError)
async def oasst_exception_handler(request: fastapi.Request, ex: OasstError):
    logger.error(f"{request.method} {request.url} failed: {repr(ex)}")

    return fastapi.responses.JSONResponse(
        status_code=int(ex.http_status_code),
        content=protocol_schema.OasstErrorResponse(
            message=ex.message,
            error_code=OasstErrorCode(ex.error_code),
        ).dict(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: fastapi.Request, ex: Exception):
    logger.exception(f"{request.method} {request.url} failed [UNHANDLED]: {repr(ex)}")
    status = HTTPStatus.INTERNAL_SERVER_ERROR
    return fastapi.responses.JSONResponse(
        status_code=status.value, content={"message": status.name, "error_code": OasstErrorCode.GENERIC_ERROR}
    )


# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if settings.UPDATE_ALEMBIC:

    @app.on_event("startup")
    def alembic_upgrade():
        logger.info("Attempting to upgrade alembic on startup")
        try:
            alembic_ini_path = Path(__file__).parent / "alembic.ini"
            alembic_cfg = alembic.config.Config(str(alembic_ini_path))
            alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URI)
            alembic.command.upgrade(alembic_cfg, "head")
            logger.info("Successfully upgraded alembic on startup")
        except Exception:
            logger.exception("Alembic upgrade failed on startup")


if settings.OFFICIAL_WEB_API_KEY:

    @app.on_event("startup")
    def create_official_web_api_client():
        with Session(engine) as session:
            try:
                api_auth(settings.OFFICIAL_WEB_API_KEY, db=session)
            except OasstError:
                logger.info("Creating official web API client")
                create_api_client(
                    session=session,
                    api_key=settings.OFFICIAL_WEB_API_KEY,
                    description="The official web client for the OASST backend.",
                    frontend_type="web",
                    trusted=True,
                )


if settings.ENABLE_PROM_METRICS:

    @app.on_event("startup")
    async def enable_prom_metrics():
        Instrumentator().instrument(app).expose(app)


if settings.RATE_LIMIT:

    @app.on_event("startup")
    async def connect_redis():
        async def http_callback(request: fastapi.Request, response: fastapi.Response, pexpire: int):
            """Error callback function when too many requests"""
            expire = ceil(pexpire / 1000)
            raise OasstError(
                f"Too Many Requests. Retry After {expire} seconds.",
                OasstErrorCode.TOO_MANY_REQUESTS,
                HTTPStatus.TOO_MANY_REQUESTS,
            )

        try:
            redis_client = redis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0", encoding="utf-8", decode_responses=True
            )
            logger.info(f"Connected to {redis_client=}")
            await FastAPILimiter.init(redis_client, http_callback=http_callback)
        except Exception:
            logger.exception("Failed to establish Redis connection")

@app.on_event("startup")
def ensure_tree_states():
    try:
        logger.info("Startup: TreeManager.ensure_tree_states()")
        with Session(engine) as db:
            api_client = api_auth(settings.OFFICIAL_WEB_API_KEY, db=db)
            tm = TreeManager(db, PromptRepository(db, api_client=api_client))
            tm.ensure_tree_states()

    except Exception:
        logger.exception("TreeManager.ensure_tree_states() failed.")

@app.on_event("startup")
@repeat_every(seconds=1 * settings.LABEL_INITIAL_PROMPT_INTERVAL, wait_first=True)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def label_initial_prompt(session: Session) -> None:
    api_client = api_auth(settings.OFFICIAL_WEB_API_KEY, db=session)
    bot = protocol_schema.User(id="__label_bot__", display_name="label_bot", auth_method="local")
    ur = UserRepository(db=session, api_client=api_client)
    tr = TaskRepository(db=session, api_client=api_client, user_repository=ur, client_user=bot)
    pr = PromptRepository(db=session, api_client=api_client, user_repository=ur, task_repository=tr, client_user=bot)
    tm = TreeManager(db=session, prompt_repository=pr)
    
    prompts_need_review = tm.query_prompts_need_review(lang="ko")
    if prompts_need_review is None:
        return
    
    for message in prompts_need_review:
        logger.info(f"Labeling prompt {message.id}")
        try:
            task = protocol_schema.LabelInitialPromptTask(
                message_id=message.id,
                prompt=message.text,
                conversation=prepare_conversation([message]),
                valid_labels=list(map(lambda x: x.value, tm.cfg.labels_initial_prompt)),
                mandatory_labels=list(map(lambda x: x.value, tm.cfg.mandatory_labels_initial_prompt)),
                mode=protocol_schema.LabelTaskMode.full,
                disposition=protocol_schema.LabelTaskDisposition.quality,
                labels=tm._get_label_descriptions(tm.cfg.labels_initial_prompt),
            )
            parent_message_id = message.id
            message_tree_id = message.message_tree_id
            pr.task_repository.store_task(
                task=task, message_tree_id=message_tree_id, parent_message_id=parent_message_id, collective=False
            )
            
            frontend_message_id = str(uuid4())
            pr.task_repository.bind_frontend_message_id(task_id=task.id, frontend_message_id=frontend_message_id)
            labels = {
                "spam": 0, "lang_mismatch": 0, "quality": 1, "creativity": 1, "humor": 1, 
                "toxicity": 0, "violence": 0, "not_appropriate": 0, "pii": 0, "hate_speech": 0, 
                "sexual_content": 0
            }
            interaction = protocol_schema.TextLabels(
                user=bot, labels=labels, text="", task_id=task.id, message_id=parent_message_id, lang="ko", user_message_id=frontend_message_id
            )
            _, task, message = pr.store_text_labels(interaction)
            message.review_result = True
            session.add(message)
            tm.check_condition_for_prompt_lottery(message.message_tree_id)
            tm.check_condition_for_ranking_state(message.message_tree_id)
            mts = session.query(MessageTreeState).filter(MessageTreeState.message_tree_id == message_tree_id).one_or_none()
            
            if mts is not None:
                mts.state = message_tree_state.State.GROWING
                mts.active = True
                mts.won_prompt_lottery_date = utcnow()
                session.add(mts)

        except Exception as e:
            logger.exception(f"Failed to label prompt {message.id} by {e}")
        
@app.on_event("startup")
@repeat_every(seconds=300 * settings.REPLY_PROMPT_INTERVAL, wait_first=True)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def assitant_reply(session: Session):
    api_client = api_auth(settings.OFFICIAL_WEB_API_KEY, db=session)
    gpt = protocol_schema.User(id="__GPT3.5__", display_name="GPT", auth_method="local")
 
    messages_need_replies = session.query(Message).filter(
        Message.lang == "ko", 
        Message.role == "prompter",
    ).all()
    
    if messages_need_replies is None:
        return
    
    for message in messages_need_replies[1:]:
        openai_gpt.delay(
            text=message.text,
            message_id=message.id,
            api_client=api_client.dict(),
            user=gpt.dict(),
            model=OpenAIChatModel.GPT_3_5_TURBO.value,
        )
            
            
            
    logger.info(f"found {messages_need_replies} messages need replies")
    
    #for message in me
    
@app.on_event("startup")
@repeat_every(seconds=60 * settings.USER_STATS_INTERVAL_DAY, wait_first=False)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def update_leader_board_day(session: Session) -> None:
    try:
        usr = UserStatsRepository(session)
        usr.update_stats(time_frame=UserStatsTimeFrame.day)
    except Exception:
        logger.exception("Error during leaderboard update (daily)")


@app.on_event("startup")
@repeat_every(seconds=60 * settings.USER_STATS_INTERVAL_WEEK, wait_first=False)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def update_leader_board_week(session: Session) -> None:
    try:
        usr = UserStatsRepository(session)
        usr.update_stats(time_frame=UserStatsTimeFrame.week)
    except Exception:
        logger.exception("Error during user states update (weekly)")


@app.on_event("startup")
@repeat_every(seconds=60 * settings.USER_STATS_INTERVAL_MONTH, wait_first=False)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def update_leader_board_month(session: Session) -> None:
    try:
        usr = UserStatsRepository(session)
        usr.update_stats(time_frame=UserStatsTimeFrame.month)
    except Exception:
        logger.exception("Error during user states update (monthly)")


@app.on_event("startup")
@repeat_every(seconds=60 * settings.USER_STATS_INTERVAL_TOTAL, wait_first=False)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def update_leader_board_total(session: Session) -> None:
    try:
        usr = UserStatsRepository(session)
        usr.update_stats(time_frame=UserStatsTimeFrame.total)
    except Exception:
        logger.exception("Error during user states update (total)")


@app.on_event("startup")
@repeat_every(seconds=60 * 60)  # 1 hour
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def cronjob_delete_expired_tasks(session: Session) -> None:
    delete_expired_tasks(session)
    halt_prompts_of_disabled_users(session)


@app.on_event("startup")
@repeat_every(seconds=60 * settings.CACHED_STATS_UPDATE_INTERVAL, wait_first=True)
@managed_tx_function(auto_commit=CommitMode.COMMIT)
def update_cached_stats(session: Session) -> None:
    try:
        csr = CachedStatsRepository(session)
        csr.update_all_cached_stats()
    except Exception:
        logger.exception("Error during cached stats update")


app.include_router(api_router, prefix=settings.API_V1_STR)


def get_openapi_schema():
    return json.dumps(app.openapi())


def retry_scoring_failed_message_trees():
    try:
        logger.info("TreeManager.retry_scoring_failed_message_trees()")
        with Session(engine) as db:
            api_client = api_auth(settings.OFFICIAL_WEB_API_KEY, db=db)

            pr = PromptRepository(db=db, api_client=api_client)
            tm = TreeManager(db, pr)
            tm.retry_scoring_failed_message_trees()

    except Exception:
        logger.exception("TreeManager.retry_scoring_failed_message_trees() failed.")


def main():
    # Importing here so we don't import packages unnecessarily if we're
    # importing main as a module.
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--print-openapi-schema",
        default=False,
        help="Dumps the openapi schema to stdout",
        action="store_true",
    )
    parser.add_argument("--host", help="The host to run the server", default="0.0.0.0")
    parser.add_argument("--port", help="The port to run the server", default=8080)
    parser.add_argument(
        "--retry-scoring",
        default=False,
        help="Retry scoring failed message trees",
        action="store_true",
    )

    args = parser.parse_args()

    if args.print_openapi_schema:
        print(get_openapi_schema())

    if args.retry_scoring:
        retry_scoring_failed_message_trees()

    if not (args.print_openapi_schema or args.retry_scoring):
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
