"""
Telegram bot and mini-app backend for Mafia game.
"""
import asyncio
import logging
import os
import secrets
import string
from dataclasses import dataclass, field
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          ContextTypes)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.getenv("BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
MINI_APP_URL = os.getenv("MINI_APP_URL", "https://example.com/mafia")


def generate_code(length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


@dataclass
class Player:
    user_id: int
    name: str


@dataclass
class Game:
    code: str
    host_id: int
    slots: int
    allowed_roles: List[str]
    started: bool = False
    players: List[Player] = field(default_factory=list)
    assignments: Dict[int, str] = field(default_factory=dict)

    def join(self, user_id: int, name: str) -> None:
        if self.started:
            raise ValueError("Игра уже началась")
        if any(p.user_id == user_id for p in self.players):
            raise ValueError("Вы уже в игре")
        if len(self.players) >= self.slots:
            raise ValueError("Все места заняты")
        self.players.append(Player(user_id=user_id, name=name))

    def start(self) -> None:
        if self.started:
            raise ValueError("Игра уже началась")
        if len(self.players) < 4:
            raise ValueError("Нужно минимум 4 игрока")
        self.started = True
        pool = self.allowed_roles.copy()
        while len(pool) < len(self.players):
            pool.append("Мирный житель")
        rng = secrets.SystemRandom()
        rng.shuffle(pool)
        for player, role in zip(self.players, pool):
            self.assignments[player.user_id] = role


class GameRegistry:
    def __init__(self) -> None:
        self.games: Dict[str, Game] = {}

    def create(self, host_id: int, slots: int, allowed_roles: List[str]) -> Game:
        code = generate_code()
        game = Game(code=code, host_id=host_id, slots=slots, allowed_roles=allowed_roles)
        self.games[code] = game
        logger.info("Created game %s", code)
        return game

    def get(self, code: str) -> Game:
        try:
            return self.games[code]
        except KeyError:
            raise ValueError("Игра не найдена")


registry = GameRegistry()


# Telegram bot setup
ROLE_DESCRIPTIONS = {
    "Мирный житель": "Голосует днем, пытается вычислить мафию.",
    "Мафия": "Убирает игрока ночью. Цель — остаться в большинстве.",
    "Детектив": "Каждую ночь проверяет игрока и узнает его роль.",
    "Доктор": "Ночью лечит игрока, спасая его от устранения.",
    "Офицер": "Может арестовать игрока один раз за игру, блокируя его ход.",
    "Камикадзе": "При устранении забирает с собой одного мафиози.",
    "Фантом": "Появляется как мирный, но один раз может избежать голосования.",
    "Двойной агент": "Смотрит роль одного игрока и меняет сторону в зависимости от роли." ,
}


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("Запустить игру", callback_data="host")],
        [InlineKeyboardButton("Информация о игре", callback_data="info")],
        [InlineKeyboardButton("Открыть мини-апп", url=MINI_APP_URL)],
    ]
    await update.message.reply_text(
        "Привет! Я бот для мафии. Что делаем?",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    await query.answer()
    if query.data == "info":
        text = "\n".join(f"<b>{role}</b> — {desc}" for role, desc in ROLE_DESCRIPTIONS.items())
        await query.edit_message_text(text=text, parse_mode=ParseMode.HTML)
    elif query.data == "host":
        keyboard = [
            [InlineKeyboardButton("Открыть мини-апп", url=MINI_APP_URL)],
        ]
        await query.edit_message_text(
            "Чтобы захостить игру, открой мини-апп и выбери количество мест и роли.",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )


async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = "\n".join(f"<b>{role}</b> — {desc}" for role, desc in ROLE_DESCRIPTIONS.items())
    await update.message.reply_text(text=text, parse_mode=ParseMode.HTML)


async def webhook() -> None:
    application = (
        Application.builder()
        .token(TOKEN)
        .concurrent_updates(True)
        .build()
    )
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("info", info_command))
    application.add_handler(CallbackQueryHandler(handle_callback))

    await application.initialize()
    await application.start()
    logger.info("Bot started. Listening for updates...")
    await application.updater.start_polling()
    await application.updater.idle()


# Mini App backend
class HostRequest(BaseModel):
    slots: int
    roles: List[str]


class JoinRequest(BaseModel):
    user_id: int
    name: str


app = FastAPI(title="Mafia Mini App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/api/games")
def host_game(body: HostRequest):
    if not 4 <= body.slots <= 12:
        raise HTTPException(status_code=400, detail="Количество мест должно быть от 4 до 12")
    allowed_roles = [role for role in body.roles if role in ROLE_DESCRIPTIONS]
    if not allowed_roles:
        allowed_roles = ["Мафия", "Детектив", "Доктор", "Мирный житель"]
    game = registry.create(host_id=0, slots=body.slots, allowed_roles=allowed_roles)
    return {"code": game.code, "slots": game.slots, "roles": game.allowed_roles}


@app.post("/api/games/{code}/join")
def join_game(code: str, body: JoinRequest):
    try:
        game = registry.get(code)
        game.join(body.user_id, body.name)
        return {"status": "joined", "players": [p.name for p in game.players]}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/games/{code}/start")
def start_game(code: str):
    try:
        game = registry.get(code)
        game.start()
        return {
            "status": "started",
            "assignments": game.assignments,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/games/{code}")
def get_game(code: str):
    try:
        game = registry.get(code)
        return {
            "code": game.code,
            "players": [p.name for p in game.players],
            "started": game.started,
            "roles": game.allowed_roles,
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/", response_class=FileResponse)
async def root():
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail="Мини-приложение не найдено")
    return FileResponse(INDEX_FILE)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(webhook())
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
