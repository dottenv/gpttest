# FAI Code Assistant с локальным G4F v3.0

Полностью обновленная версия FAI Code Assistant с рекурсивным поиском провайдеров, автоматическим тестированием и балансировкой нагрузки.

## 🚀 Новые возможности v3.0

- **Рекурсивный поиск провайдеров** - автоматически находит всех доступных провайдеров
- **Параллельное тестирование** - тестирует провайдеров по 5 одновременно
- **Периодическое обновление** - каждый час перетестирует провайдеров
- **Умная балансировка** - распределяет запросы между рабочими провайдерами
- **Пользовательские настройки** - выбор модели, языка, промптов
- **API для управления** - мониторинг и управление провайдерами
- **Поддержка 15+ моделей** - GPT-4, Claude, Gemini, LLaMA, DeepSeek и др.
- **Локальный g4f** - не требует установки через pip

## 📁 Структура проекта

```
gpttest/
├── gpt4free/           # Локальная копия g4f
├── server.py           # Основной сервер с улучшениями
├── test_server.py      # Тестирование сервера
├── test_final.py       # Тестирование g4f
├── test.py            # Базовые тесты
└── README.md          # Эта инструкция
```

## 🛠 Установка и запуск

### 1. Установка зависимостей

```bash
pip install fastapi uvicorn tinydb bcrypt pyjwt python-multipart requests aiohttp pillow curl_cffi beautifulsoup4 cryptography
```

### 2. Запуск сервера

```bash
# Основной способ
python server.py

# Или через uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000
```

Сервер запустится на `http://localhost:8000`

При запуске автоматически:
- Найдет всех провайдеров в g4f
- Протестирует их параллельно
- Запустит периодическое обновление

### 3. Проверка статуса

```bash
# Проверка провайдеров
curl http://localhost:8000/providers/status

# Перетестирование провайдеров
curl -X POST http://localhost:8000/providers/retest \
  -H "Authorization: Bearer YOUR_TOKEN"

# Список моделей
curl http://localhost:8000/models
```

## 📡 API Endpoints

### Новые endpoints для управления провайдерами и моделями:

#### Статус провайдеров
```http
GET /providers/status
```
Возвращает информацию о всех провайдерах и их статусе.

#### Доступные модели
```http
GET /models
```
Возвращает список всех доступных AI моделей.

#### Перетестирование провайдеров
```http
POST /providers/retest
Authorization: Bearer <token>
```
Запускает повторное тестирование всех провайдеров.

### Улучшенные endpoints:

#### Чат с выбором модели
```http
POST /chat
Authorization: Bearer <token>
Content-Type: application/json

{
    "message": "Объясни этот код",
    "context_type": "all",
    "model": "gpt-4o-mini"  // Новый параметр
}
```

#### Автодополнение с выбором модели
```http
POST /code-complete
Authorization: Bearer <token>
Content-Type: application/json

{
    "code": "def hello():",
    "cursor_position": 13,
    "file_path": "test.py",
    "language": "python",
    "model": "gpt-4o-mini"  // Новый параметр
}
```

## 🔧 Конфигурация

### Переменные окружения

- `SECRET_KEY` - Ключ для JWT токенов (по умолчанию: автогенерация)
- `PORT` - Порт сервера (по умолчанию: 8000)
- `DEBUG` - Режим отладки (по умолчанию: False)

### Поддерживаемые модели

- `gpt-4`, `gpt-4o`, `gpt-4o-mini`
- `gpt-4.1-mini`, `gpt-4.1-nano`
- `blackboxai`
- `deepseek-v3`, `deepseek-r1`
- `qwen-2.5`, `qwen-3`
- `gemini-2.0-flash`
- И многие другие...

## 🔍 Мониторинг

### Проверка статуса провайдеров

```bash
curl http://localhost:8000/providers/status
```

Ответ:
```json
{
    "g4f_available": true,
    "working_providers_count": 5,
    "providers": {
        "Blackbox": {
            "status": "✅ Working",
            "last_check": "2025-01-27T10:30:00",
            "response_time": 2.5,
            "error_count": 0
        }
    },
    "current_provider_index": 2
}
```

## 🚨 Устранение неполадок

### Проблема: RuntimeError: no running event loop

**ИСПРАВЛЕНО!** Теперь используется ленивая инициализация AI сервиса:
1. AIService создается только при первом обращении
2. Asyncio tasks не создаются при импорте
3. Event loop запускается только когда нужно

### Проблема: G4F не инициализируется

1. Проверьте наличие папки `gpt4free/`
2. Убедитесь, что установлены все зависимости
3. Проверьте логи при запуске сервера

### Проблема: Нет рабочих провайдеров

1. Запустите перетестирование: `POST /providers/retest`
2. Проверьте интернет-соединение
3. Некоторые провайдеры могут быть временно недоступны

### Проблема: Медленные ответы

1. Проверьте статус провайдеров
2. Используйте более быстрые модели (например, `gpt-4o-mini`)
3. Система автоматически переключается между провайдерами

### Проблема: Ответы на неправильном языке

1. Проверьте настройки языка в профиле пользователя
2. Сервер автоматически добавляет строгие инструкции по языку
3. Некоторые провайдеры могут игнорировать языковые инструкции

### Проблема: Слишком длинные запросы

1. Сервер автоматически обрезает контекст до 8000 символов
2. Важные части (начало и конец) сохраняются
3. Пользовательский запрос ограничивается 4000 символами

### Проблема: Автоматически открывается браузер

1. Исправлено в новой версии
2. Установлены переменные окружения для отключения GUI
3. Отключены логи доступа для уменьшения вывода

## 📊 Балансировка нагрузки

Сервер автоматически:
- Тестирует всех провайдеров при запуске
- Распределяет запросы между рабочими провайдерами
- Переключается на следующего провайдера при ошибках
- Ведет статистику ошибок и времени ответа

## 🔄 Обновление

Для обновления локальной версии g4f:
1. Замените папку `gpt4free/` на новую версию
2. Перезапустите сервер
3. Запустите перетестирование провайдеров

## 📝 Логирование

Сервер выводит подробную информацию о:
- Инициализации g4f
- Тестировании провайдеров
- Обработке запросов
- Ошибках и переключениях провайдеров
- Обрезке длинных запросов
- Проблемах с языком ответов

## 🔧 Новые возможности

### Обработка длинных запросов
- Автоматическая обрезка контекста до 8000 символов
- Сохранение важных частей (системный промпт, пользовательский запрос)
- Увеличенный таймаут для длинных запросов (20 секунд)

### Принудительное указание языка
- Строгие инструкции в начале и конце промпта
- Проверка ответов на соответствие языку
- Предупреждения о возможных проблемах с языком

### Оптимизация производительности
- Тестирование только 5 лучших провайдеров для скорости
- Отключение автооткрытия браузера
- Уменьшение уровня логирования

## 🔧 Последние исправления (v2.1)

### ✅ Исправлена ошибка "RuntimeError: no running event loop"
- Убрана инициализация asyncio tasks при импорте
- Добавлена ленивая инициализация AI сервиса
- Event loop создается только при необходимости
- Флаг `_initialized` предотвращает повторную инициализацию

### ✅ Улучшен запуск сервера
- Новый скрипт `run_server.py` с проверками
- Автоматическая настройка окружения
- Проверка зависимостей перед запуском
- Отключение reloader для стабильности

### ✅ Расширено тестирование
- Новый скрипт `test_fixes.py` для проверки исправлений
- Тестирование ленивой инициализации
- Проверка обработки длинных сообщений
- Тестирование всех API endpoints

### 🚀 Рекомендуемый порядок запуска:

1. **Проверка и запуск:**
   ```bash
   python run_server.py
   ```

2. **Тестирование (в новом терминале):**
   ```bash
   python test_fixes.py
   ```

3. **Использование:**
   - Откройте http://localhost:1337/admin
   - Зарегистрируйтесь или войдите
   - Начните использовать AI помощника

## 🤝 Поддержка

При возникновении проблем:
1. Проверьте логи сервера
2. Запустите тестовые скрипты (`test_fixes.py` для новых функций)
3. Проверьте статус провайдеров через API
4. Убедитесь, что контекст не слишком длинный
5. Проверьте настройки языка в профиле пользователя
6. Используйте `run_server.py` для запуска с проверками