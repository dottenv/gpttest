import os
import sys
import json
import asyncio
import traceback
import inspect
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from tinydb import TinyDB, Query
import bcrypt
import jwt

# Добавляем локальную папку g4f в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gpt4free'))

# Инициализация FastAPI
app = FastAPI(title="FAI Code Assistant Backend")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# База данных
db = TinyDB('fai_database.json')
users_table = db.table('users')
sessions_table = db.table('sessions')
chats_table = db.table('chats')
settings_table = db.table('settings')

# Конфигурация безопасности
SECRET_KEY = os.getenv("SECRET_KEY", "fai-secret-key-2025-super-secure")
ALGORITHM = "HS256"
security = HTTPBearer()

# Модели данных
class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class ChatMessage(BaseModel):
    message: str
    selected_files: Optional[List[str]] = []
    context_type: str = "all"

class UserSettings(BaseModel):
    ai_model: str = "gpt-4"
    language: str = "en"
    system_prompt: str = ""
    ai_rules: str = ""
    auto_complete: bool = True

class FileOperation(BaseModel):
    file_path: str
    content: str
    operation: str

class CodeCompletion(BaseModel):
    code: str
    cursor_position: int
    file_path: str
    language: str = "python"

# Глобальные переменные для g4f
G4F_AVAILABLE = False
ALL_PROVIDERS = []
WORKING_PROVIDERS = []
PROVIDER_STATUS = {}
CURRENT_PROVIDER_INDEX = 0
LAST_PROVIDER_TEST = None

# Рекурсивно получаем всех провайдеров
def get_all_providers():
    global ALL_PROVIDERS
    try:
        import g4f

        # Отключаем логи и GUI
        os.environ['G4F_NO_GUI'] = '1'
        os.environ['DISPLAY'] = ''
        g4f.debug.logging = False
        g4f.check_version = False

        providers = []

        # Получаем все классы провайдеров из g4f.Provider
        for name in dir(g4f.Provider):
            if not name.startswith('_'):
                try:
                    provider_class = getattr(g4f.Provider, name)
                    if inspect.isclass(provider_class) and hasattr(provider_class, 'create_async'):
                        providers.append(provider_class)
                        print(f"📦 Found provider: {name}")
                except Exception as e:
                    continue

        ALL_PROVIDERS = providers
        print(f"📋 Total providers found: {len(ALL_PROVIDERS)}")
        return providers

    except Exception as e:
        print(f"❌ Error getting providers: {e}")
        return []

# Тестирование провайдеров
async def test_providers():
    global WORKING_PROVIDERS, PROVIDER_STATUS, LAST_PROVIDER_TEST

    if not ALL_PROVIDERS:
        get_all_providers()

    WORKING_PROVIDERS = []
    PROVIDER_STATUS = {}

    print(f"🔄 Testing {len(ALL_PROVIDERS)} providers...")

    async def test_single_provider(provider):
        try:
            import g4f
            response = await asyncio.wait_for(
                g4f.ChatCompletion.create_async(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}],
                    provider=provider
                ),
                timeout=8.0
            )
            return bool(response and len(response.strip()) > 2)
        except Exception as e:
            return False

    # Тестируем провайдеров параллельно (по 5 за раз)
    for i in range(0, len(ALL_PROVIDERS), 5):
        batch = ALL_PROVIDERS[i:i+5]
        tasks = [test_single_provider(provider) for provider in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for provider, result in zip(batch, results):
            name = provider.__name__
            if result is True:
                WORKING_PROVIDERS.append(provider)
                PROVIDER_STATUS[name] = "✅ Working"
                print(f"✅ {name}: Working")
            else:
                PROVIDER_STATUS[name] = "❌ Failed"
                print(f"❌ {name}: Failed")

    LAST_PROVIDER_TEST = datetime.now()
    print(f"🎯 Testing complete: {len(WORKING_PROVIDERS)}/{len(ALL_PROVIDERS)} working")

# Инициализация g4f
async def initialize_g4f():
    global G4F_AVAILABLE

    try:
        import g4f
        print(f"✅ Local g4f version: {g4f.version}")
        G4F_AVAILABLE = True

        # Получаем всех провайдеров
        get_all_providers()

        # Тестируем провайдеров
        await test_providers()

        if not WORKING_PROVIDERS:
            print("⚠️ No working providers found!")

    except ImportError as e:
        print(f"❌ g4f ImportError: {e}")
        G4F_AVAILABLE = False
    except Exception as e:
        print(f"❌ g4f initialization error: {e}")
        G4F_AVAILABLE = False

# Периодическое тестирование провайдеров
async def periodic_provider_test():
    while True:
        try:
            await asyncio.sleep(3600)  # Каждый час
            print("🔄 Hourly provider test...")
            await test_providers()
        except Exception as e:
            print(f"❌ Periodic test error: {e}")

# Балансировка нагрузки - выбор следующего провайдера
def get_next_provider():
    global CURRENT_PROVIDER_INDEX
    if not WORKING_PROVIDERS:
        return None

    provider = WORKING_PROVIDERS[CURRENT_PROVIDER_INDEX]
    CURRENT_PROVIDER_INDEX = (CURRENT_PROVIDER_INDEX + 1) % len(WORKING_PROVIDERS)
    return provider

# Фоновое тестирование провайдеров
async def background_provider_testing():
    """Фоновое тестирование - добавляет провайдеров по мере их проверки"""
    global WORKING_PROVIDERS, PROVIDER_STATUS
    
    if not ALL_PROVIDERS:
        get_all_providers()
    
    print(f"🔄 Background testing {len(ALL_PROVIDERS)} providers...")
    
    async def test_and_add_provider(provider):
        try:
            import g4f
            response = await asyncio.wait_for(
                g4f.ChatCompletion.create_async(
                    model="gpt-4",
                    messages=[{"role": "user", "content": "Hi"}],
                    provider=provider
                ),
                timeout=8.0
            )
            
            if response and len(response.strip()) > 2:
                WORKING_PROVIDERS.append(provider)
                PROVIDER_STATUS[provider.__name__] = "✅ Working"
                print(f"✅ {provider.__name__}: Added to working list ({len(WORKING_PROVIDERS)} total)")
                return True
            else:
                PROVIDER_STATUS[provider.__name__] = "❌ Failed"
                return False
        except Exception as e:
            PROVIDER_STATUS[provider.__name__] = "❌ Failed"
            return False
    
    # Тестируем провайдеров по 3 одновременно и добавляем рабочих сразу
    for i in range(0, len(ALL_PROVIDERS), 3):
        batch = ALL_PROVIDERS[i:i+3]
        tasks = [test_and_add_provider(provider) for provider in batch]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Небольшая пауза между батчами
        await asyncio.sleep(0.5)
    
    print(f"🎯 Background testing complete: {len(WORKING_PROVIDERS)}/{len(ALL_PROVIDERS)} working")

# Быстрая инициализация g4f
async def quick_initialize_g4f():
    global G4F_AVAILABLE
    
    try:
        import g4f
        print(f"✅ Local g4f version: {g4f.version}")
        G4F_AVAILABLE = True
        
        # Получаем всех провайдеров
        get_all_providers()
        
        # Запускаем фоновое тестирование
        asyncio.create_task(background_provider_testing())
        
        print(f"🚀 API ready! Testing {len(ALL_PROVIDERS)} providers in background...")
        
    except ImportError as e:
        print(f"❌ g4f ImportError: {e}")
        G4F_AVAILABLE = False
    except Exception as e:
        print(f"❌ g4f initialization error: {e}")
        G4F_AVAILABLE = False

# Запускаем инициализацию при старте сервера
@app.on_event("startup")
async def startup():
    await quick_initialize_g4f()
    # Запускаем периодическое тестирование в фоне
    asyncio.create_task(periodic_provider_test())

# Функции аутентификации
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=30)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# ... (остальной код остается БЕЗ ИЗМЕНЕНИЙ, как в предыдущем рабочем варианте)
# ВАЖНО: оставить все эндпоинты и функции без изменений

# API Endpoints
@app.get("/users/check/{username}")
async def check_user_exists(username: str):
    """Проверить существует ли пользователь"""
    User = Query()
    exists = bool(users_table.search(User.username == username))
    return {"exists": exists, "username": username}

@app.delete("/users/clear")
async def clear_all_users():
    """ОСТОРОЖНО: Очистить всех пользователей"""
    users_table.truncate()
    settings_table.truncate()
    sessions_table.truncate()
    chats_table.truncate()
    return {"message": "All users cleared"}

@app.post("/register")
async def register(user: UserRegister):
    User = Query()
    if users_table.search(User.username == user.username):
        raise HTTPException(400, "Username already exists")
    if users_table.search(User.email == user.email):
        raise HTTPException(400, "Email already exists")

    user_data = {
        "username": user.username,
        "email": user.email,
        "password": hash_password(user.password),
        "created_at": datetime.now().isoformat(),
        "is_active": True
    }

    user_id = users_table.insert(user_data)

    # Создаем настройки по умолчанию
    settings_table.insert({
        "user_id": user_id,
        "ai_model": "gpt-4",
        "language": "en",
        "system_prompt": "",
        "ai_rules": "",
        "auto_complete": True
    })

    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer", "user_id": user_id}

@app.post("/login")
async def login(user: UserLogin):
    User = Query()
    db_user = users_table.search(User.username == user.username)

    if not db_user:
        raise HTTPException(401, "User not found")
    
    if not verify_password(user.password, db_user[0]['password']):
        raise HTTPException(401, "Invalid password")

    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer", "user_id": db_user[0].doc_id}

@app.get("/me")
async def get_current_user(username: str = Depends(verify_token)):
    User = Query()
    user = users_table.search(User.username == username)[0]
    return {"username": user['username'], "email": user['email'], "user_id": user.doc_id}

# Управление сессиями
@app.post("/create-session")
async def create_session(username: str = Depends(verify_token)):
    User = Query()
    user = users_table.search(User.username == username)[0]

    session_data = {
        "user_id": user.doc_id,
        "files": {},
        "created_at": datetime.now().isoformat(),
        "last_activity": datetime.now().isoformat()
    }

    session_id = sessions_table.insert(session_data)
    return {"session_id": str(session_id)}

# Работа с файлами
@app.post("/upload-files-batch")
async def upload_files_batch(files_data: dict, username: str = Depends(verify_token)):
    try:
        session_id = int(files_data['session_id'])
        files = files_data['files']

        Session = Query()
        User = Query()
        user = users_table.search(User.username == username)[0]

        # Ищем сессию пользователя
        session = sessions_table.search((Session.doc_id == session_id) & (Session.user_id == user.doc_id))
        if not session:
            # Создаем новую сессию если не найдена
            session_data = {
                "user_id": user.doc_id,
                "files": {},
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat()
            }
            new_session_id = sessions_table.insert(session_data)
            session_data['doc_id'] = new_session_id
        else:
            session_data = session[0]

        uploaded_count = 0

        for file_data in files:
            filename = file_data.get('filename')
            content = file_data.get('content', '')

            session_data['files'][filename] = {
                "content": content,
                "size": len(content.encode('utf-8')),
                "type": "text/plain",
                "modified_time": datetime.now().isoformat()
            }
            uploaded_count += 1

        sessions_table.update(session_data, Session.doc_id == session_data.get('doc_id', session_id))

        return {
            "uploaded": uploaded_count,
            "total_files": len(session_data['files'])
        }
    except Exception as e:
        return {"error": str(e), "uploaded": 0, "total_files": 0}

# AI Helper Functions с балансировкой нагрузки
async def get_ai_response(prompt: str, language: str, model: str = "gpt-4") -> str:
    """Получаем ответ от g4f с балансировкой нагрузки"""
    if not G4F_AVAILABLE:
        return get_smart_fallback(prompt, language)
    
    # Если нет рабочих провайдеров, ждем немного (возможно тестирование еще идет)
    if not WORKING_PROVIDERS:
        print("⏳ No working providers yet, waiting for background testing...")
        await asyncio.sleep(2)
        
        if not WORKING_PROVIDERS:
            return get_smart_fallback(prompt, language)

    # Пробуем провайдеров с балансировкой
    attempts = min(3, len(WORKING_PROVIDERS))

    for _ in range(attempts):
        provider = get_next_provider()
        if not provider:
            break

        try:
            import g4f
            response = await asyncio.wait_for(
                g4f.ChatCompletion.create_async(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    provider=provider
                ),
                timeout=20.0
            )

            if response and len(response.strip()) > 10:
                print(f"✅ Response via {provider.__name__} ({len(WORKING_PROVIDERS)} available)")
                return response.strip()

        except Exception as e:
            print(f"❌ Error in {provider.__name__}: {str(e)[:100]}")
            continue

    return get_smart_fallback(prompt, language)

def get_smart_fallback(prompt: str, language: str) -> str:
    """Умный fallback с анализом контекста"""
    user_message = prompt.lower()

    if language == 'ru':
        if any(word in user_message for word in ['привет', 'здравствуй', 'добро пожаловать']):
            return "Привет! Я FAI - ваш AI помощник по коду. Готов помочь с анализом, отладкой и объяснением кода!"
        elif 'анализ' in user_message or 'изучи' in user_message:
            return "Для анализа проекта мне нужен доступ к файлам. Загрузите файлы проекта, и я проанализирую структуру, найду потенциальные проблемы и дам рекомендации."
        elif any(word in user_message for word in ['ошибка', 'баг', 'не работает']):
            return "Чтобы помочь с ошибкой, покажите код и опишите проблему. Я найду причину и предложу решение."
        elif 'объясни' in user_message or 'что делает' in user_message:
            return "Покажите код, который нужно объяснить, и я подробно расскажу как он работает."
        else:
            return "Я готов помочь с программированием! Могу анализировать код, находить ошибки, объяснять функции и предлагать улучшения."
    else:
        if any(word in user_message for word in ['hello', 'hi', 'welcome']):
            return "Hello! I'm FAI - your AI code assistant. Ready to help with analysis, debugging, and code explanations!"
        elif 'analyze' in user_message or 'study' in user_message:
            return "To analyze your project, I need access to the files. Upload your project files and I'll analyze structure, find potential issues, and provide recommendations."
        elif any(word in user_message for word in ['error', 'bug', 'not working']):
            return "To help with the error, show me the code and describe the problem. I'll find the cause and suggest a solution."
        elif 'explain' in user_message or 'what does' in user_message:
            return "Show me the code you'd like explained, and I'll break down how it works in detail."
        else:
            return "I'm ready to help with programming! I can analyze code, find bugs, explain functions, and suggest improvements."

# Основной чат
@app.post("/chat")
async def chat(request: ChatMessage, username: str = Depends(verify_token)):
    User = Query()
    user = users_table.search(User.username == username)[0]

    # Получаем настройки пользователя
    Settings = Query()
    user_settings = settings_table.search(Settings.user_id == user.doc_id)
    settings = user_settings[0] if user_settings else {}

    # Получаем активную сессию
    Session = Query()
    sessions = sessions_table.search(Session.user_id == user.doc_id)
    if not sessions:
        raise HTTPException(404, "No active session")

    session = sessions[-1]

    # Формируем контекст
    context_files = {}
    if request.context_type == "selected" and request.selected_files:
        context_files = {f: session['files'][f] for f in request.selected_files if f in session['files']}
    else:
        context_files = session['files']

    context_str = "\n\n".join([
        f"=== {filename} ===\n{file_data['content']}"
        for filename, file_data in list(context_files.items())[:10]  # Ограничиваем контекст
    ])

    # Формируем системный промт
    system_prompt = settings.get('system_prompt', '') or "You are FAI, an expert code assistant."
    if settings.get('ai_rules'):
        system_prompt += f"\n\nRULES: {settings['ai_rules']}"

    # Инструкции по языку
    language = settings.get('language', 'en')
    lang_instructions = {
        'en': 'Respond in English.',
        'ru': 'Отвечай на русском языке.',
        'es': 'Responde en español.',
        'fr': 'Réponds en français.',
        'de': 'Antworte auf Deutsch.'
    }
    if language != 'en':
        system_prompt += f"\n\n{lang_instructions.get(language, lang_instructions['en'])}"

    full_prompt = f"{system_prompt}\n\nCODE CONTEXT:\n{context_str}\n\nUSER QUERY:\n{request.message}"

    # Используем выбранную модель пользователя
    model = settings.get('ai_model', 'gpt-4')

    try:
        response = await get_ai_response(full_prompt, language, model)

        # Сохраняем историю чата
        chat_data = {
            "user_id": user.doc_id,
            "session_id": session.doc_id,
            "user_message": request.message,
            "ai_response": response,
            "context_files": list(context_files.keys()),
            "timestamp": datetime.now().isoformat()
        }
        chats_table.insert(chat_data)

        return {"response": response}
    except Exception as e:
        print(f"❌ Общая ошибка AI: {e}")
        fallback_response = get_smart_fallback(full_prompt, language)

        # Сохраняем fallback в историю
        chat_data = {
            "user_id": user.doc_id,
            "session_id": session.doc_id,
            "user_message": request.message,
            "ai_response": fallback_response,
            "context_files": list(context_files.keys()),
            "timestamp": datetime.now().isoformat()
        }
        chats_table.insert(chat_data)

        return {"response": fallback_response}

# Автодополнение кода
@app.post("/code-complete")
async def code_complete(request: CodeCompletion, username: str = Depends(verify_token)):
    if not G4F_AVAILABLE or not WORKING_PROVIDERS:
        return {"completion": "", "error": "AI not available"}

    try:
        lines = request.code.split('\n')
        current_line = len(request.code[:request.cursor_position].split('\n')) - 1
        context = '\n'.join(lines[max(0, current_line-5):current_line+3])

        prompt = f"Complete this {request.language} code:\n\n{context}\n\nProvide only the completion:"

        # Ждем провайдеров если их еще нет
        if not WORKING_PROVIDERS:
            await asyncio.sleep(1)
            
        if not WORKING_PROVIDERS:
            return {"completion": "", "error": "No providers available yet"}
        
        # Используем балансировку для автодополнения
        attempts = min(2, len(WORKING_PROVIDERS))

        for _ in range(attempts):
            provider = get_next_provider()
            if not provider:
                break

            try:
                import g4f
                response = await asyncio.wait_for(
                    g4f.ChatCompletion.create_async(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        provider=provider
                    ),
                    timeout=10.0
                )

                if response and len(response.strip()) > 5:
                    return {"completion": response.strip()}

            except Exception as e:
                print(f"Provider error {provider.__name__}: {str(e)[:100]}")
                continue

        return {"completion": "", "error": "All providers failed"}
    except Exception as e:
        return {"completion": "", "error": str(e)}

# Настройки пользователя
@app.get("/settings")
async def get_settings(username: str = Depends(verify_token)):
    User = Query()
    user = users_table.search(User.username == username)[0]

    Settings = Query()
    settings = settings_table.search(Settings.user_id == user.doc_id)

    if settings:
        return settings[0]
    else:
        default_settings = {
            "user_id": user.doc_id,
            "ai_model": "gpt-4",
            "language": "en",
            "system_prompt": "",
            "ai_rules": "",
            "auto_complete": True
        }
        settings_table.insert(default_settings)
        return default_settings

@app.post("/settings")
async def update_settings(settings: UserSettings, username: str = Depends(verify_token)):
    User = Query()
    user = users_table.search(User.username == username)[0]

    Settings = Query()
    settings_data = {
        "user_id": user.doc_id,
        "ai_model": settings.ai_model,
        "language": settings.language,
        "system_prompt": settings.system_prompt,
        "ai_rules": settings.ai_rules,
        "auto_complete": settings.auto_complete
    }

    existing = settings_table.search(Settings.user_id == user.doc_id)
    if existing:
        settings_table.update(settings_data, Settings.user_id == user.doc_id)
    else:
        settings_table.insert(settings_data)

    return {"message": "Settings updated"}

# Управление сессиями чата
@app.get("/chat-sessions")
async def get_chat_sessions(username: str = Depends(verify_token)):
    User = Query()
    user = users_table.search(User.username == username)[0]

    Session = Query()
    sessions = sessions_table.search(Session.user_id == user.doc_id)

    # Получаем последнее сообщение для каждой сессии
    Chat = Query()
    session_list = []
    for session in sessions[-10:]:  # Последние 10 сессий
        last_chat = chats_table.search((Chat.session_id == session.doc_id) & (Chat.user_id == user.doc_id))
        last_message = last_chat[-1] if last_chat else None

        session_list.append({
            "id": session.doc_id,
            "created_at": session["created_at"],
            "last_message": last_message["user_message"][:50] + "..." if last_message else "New chat",
            "message_count": len(last_chat)
        })

    return {"sessions": session_list}

@app.get("/chat-history/{session_id}")
async def get_session_history(session_id: int, username: str = Depends(verify_token)):
    User = Query()
    user = users_table.search(User.username == username)[0]

    Chat = Query()
    chats = chats_table.search((Chat.session_id == session_id) & (Chat.user_id == user.doc_id))

    return {"history": chats}

# История текущей сессии
@app.get("/chat-history")
async def get_chat_history(username: str = Depends(verify_token)):
    User = Query()
    user = users_table.search(User.username == username)[0]

    # Получаем текущую сессию
    Session = Query()
    sessions = sessions_table.search(Session.user_id == user.doc_id)
    if not sessions:
        return {"history": []}

    current_session = sessions[-1]
    Chat = Query()
    chats = chats_table.search((Chat.session_id == current_session.doc_id) & (Chat.user_id == user.doc_id))

    return {"history": chats}

@app.delete("/chat-history")
async def clear_chat_history(username: str = Depends(verify_token)):
    User = Query()
    user = users_table.search(User.username == username)[0]

    Chat = Query()
    chats_table.remove(Chat.user_id == user.doc_id)

    return {"message": "Chat history cleared"}

# Операции с файлами
@app.post("/file-operation")
async def file_operation(operation: FileOperation, username: str = Depends(verify_token)):
    User = Query()
    user = users_table.search(User.username == username)[0]

    Session = Query()
    sessions = sessions_table.search(Session.user_id == user.doc_id)
    if not sessions:
        raise HTTPException(404, "No active session")

    session = sessions[-1]

    if operation.operation == "create" or operation.operation == "update":
        session['files'][operation.file_path] = {
            "content": operation.content,
            "size": len(operation.content.encode('utf-8')),
            "type": "text/plain",
            "modified_time": datetime.now().isoformat()
        }
    elif operation.operation == "delete":
        if operation.file_path in session['files']:
            del session['files'][operation.file_path]

    sessions_table.update(session, Session.doc_id == session.doc_id)

    return {"message": f"File {operation.operation} successful", "file_path": operation.file_path}

# Новые API endpoints для управления провайдерами и моделями
@app.get("/providers/status")
async def get_providers_status():
    """Получить статус всех провайдеров"""
    return {
        "g4f_available": G4F_AVAILABLE,
        "total_providers": len(ALL_PROVIDERS),
        "working_providers_count": len(WORKING_PROVIDERS),
        "providers": PROVIDER_STATUS,
        "current_provider_index": CURRENT_PROVIDER_INDEX,
        "last_test": LAST_PROVIDER_TEST.isoformat() if LAST_PROVIDER_TEST else None
    }

@app.post("/providers/retest")
async def retest_providers(username: str = Depends(verify_token)):
    """Перетестировать всех провайдеров"""
    try:
        await test_providers()
        return {
            "message": "Providers retested successfully",
            "working_count": len(WORKING_PROVIDERS),
            "total_count": len(ALL_PROVIDERS)
        }
    except Exception as e:
        raise HTTPException(500, f"Retest failed: {str(e)}")

@app.get("/models")
async def get_available_models():
    """Получить список доступных моделей"""
    models = [
        "gpt-4", "gpt-4o", "gpt-4o-mini",
        "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
        "claude-3-opus", "claude-3-sonnet",
        "gemini-pro", "gemini-1.5-pro",
        "llama-2-70b", "llama-3-8b",
        "deepseek-coder", "deepseek-chat",
        "qwen-turbo", "qwen-plus"
    ]
    return {"models": models, "default": "gpt-4"}

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 19132)),
        reload=bool(os.getenv("DEBUG", False))
    )
