# 🛡️ Redis8 Spam Guard

Интеллектуальная система классификации спама для постов dev.to с использованием Redis 8 Vector Sets и FastAPI.

## 🎯 Особенности

- **Real-time классификация** - мгновенный анализ постов
- **Redis 8 Vector Sets** - использование новейшей технологии векторного поиска  
- **FastAPI** - современный асинхронный API
- **Машинное обучение** - классификация на основе k-NN с векторными эмбеддингами
- **Web интерфейс** - удобный интерфейс для модераторов
- **Обратная связь** - система улучшения модели на основе отзывов модераторов

## 🏗️ Архитектура

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Dev.to API    │ -> │  FastAPI Server  │ -> │  Redis 8        │
│                 │    │                  │    │  Vector Sets    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              |
                       ┌──────────────────┐
                       │  Web Interface   │
                       │  (Moderators)    │
                       └──────────────────┘
```

### Компоненты:

1. **Data Collector** - сбор постов с dev.to API
2. **Text Preprocessor** - очистка и подготовка текста
3. **Vector Embedder** - преобразование текста в векторы с помощью Sentence Transformers
4. **Redis Vector Store** - хранение и поиск векторов в Redis 8
5. **k-NN Classifier** - классификация на основе ближайших соседей
6. **FastAPI Server** - REST API для классификации
7. **Web Interface** - интерфейс для модераторов

## 🚀 Быстрый старт

### Предварительные требования

- Python 3.11+
- Docker и Docker Compose
- Redis 8 с поддержкой Vector Sets

### Установка

1. **Клонируйте репозиторий**
```bash
git clone https://github.com/your-username/devto-spam-classifier.git
cd devto-spam-classifier
```

2. **Запустите с Docker Compose**
```bash
docker-compose up -d
```

3. **Или установите локально**
```bash
# Создайте виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Установите зависимости
pip install -r requirements.txt

# Запустите Redis 8
docker run -d -p 6379:6379 redis/redis-stack:latest

# Запустите приложение
uvicorn main:app --reload
```

### Первоначальное обучение модели

```bash
# Соберите данные и обучите модель
python train_model.py
```

## 📖 Использование

### REST API

#### Классификация одного поста
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "id": 123,
    "title": "How to Learn Python",
    "description": "A comprehensive guide...",
    "tag_list": ["python", "tutorial"],
    "reading_time_minutes": 10,
    "public_reactions_count": 50,
    "comments_count": 10,
    "user": {"followers_count": 100}
  }'
```

**Ответ:**
```json
{
  "post_id": 123,
  "is_spam": false,
  "confidence": 0.85,
  "recommendation": "approve",
  "reasoning": ["Similar to legitimate posts"],
  "processing_time_ms": 45.2
}
```

#### Пакетная классификация
```bash
curl -X POST "http://localhost:8000/classify-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "posts": [...],
    "threshold": 0.8
  }'
```

#### Обратная связь от модератора
```bash
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "post_id": 123,
    "is_spam": true,
    "moderator_id": "mod_001",
    "notes": "Contains promotional content"
  }'
```

### Web интерфейс

Откройте http://localhost:8000 в браузере для доступа к интерфейсу модератора.

## 🧠 Как работает классификация

### 1. Извлечение признаков
```python
features = {
    'title': "preprocessed title text",
    'description': "preprocessed description", 
    'tags': ["python", "tutorial"],
    'reading_time': 10,
    'reactions_count': 50,
    'comments_count': 10,
    'user_followers': 100
}
```

### 2. Векторизация
- **Текстовые признаки**: Sentence Transformers (384 измерения)
- **Числовые признаки**: нормализованные метрики (5 измерений)
- **Итоговый вектор**: 389 измерений

### 3. Поиск похожих постов
```python
# Используем Redis Vector Sets для поиска k ближайших соседей
similar_posts = redis.vset_search(
    'training_vectors', 
    query_vector, 
    limit=5
)
```

### 4. Классификация
- Голосование большинства среди k ближайших соседей
- Взвешивание по расстоянию в векторном пространстве
- Расчет confidence score

### 5. Эвристические правила
Для новых постов без обучающих данных:
- Спам-слова в заголовке/описании
- Низкая вовлеченность (реакции, комментарии)
- Подозрительные теги
- Профиль автора (количество подписчиков)

## 🔧 Конфигурация

### Environment Variables
```bash
REDIS_URL=redis://localhost:6379
ENVIRONMENT=production
MODEL_CACHE_DIR=/app/cache
LOG_LEVEL=INFO
```

### Redis Configuration
```conf
# redis.conf
loadmodule /opt/redis-stack/lib/redisearch.so
loadmodule /opt/redis-stack/lib/redisjson.so
save 900 1
save 300 10
save 60 10000
```

## 📊 Мониторинг и метрики

### Prometheus метрики
- `classification_requests_total` - общее количество запросов
- `spam_detected_total` - количество обнаруженного спама
- `classification_duration_seconds` - время обработки
- `model_confidence_histogram` - распределение confidence scores

### Grafana дашборд
Включенный дашборд показывает:
- Количество классификаций в реальном времени
- Процент спама
- Производительность системы
- Точность модели

## 🧪 Тестирование

```bash
# Запуск всех тестов
python -m pytest test_api.py -v

# Тесты производительности
python -m pytest test_api.py::TestPerformance -v

# Тесты интеграции с Redis
python -m pytest test_api.py::TestRedisIntegration -v

# Покрытие кода
python -m pytest --cov=main --cov-report=html
```

### Типы тестов
- **Unit тесты** - отдельные компоненты
- **Integration тесты** - взаимодействие с Redis
- **API тесты** - endpoints FastAPI
- **Performance тесты** - нагрузочное тестирование

## 📈 Производительность

### Бенчмарки
- **Классификация одного поста**: ~50ms
- **Пакетная обработка 100 постов**: ~2s  
- **Пропускная способность**: 1000+ запросов/сек
- **Память**: ~200MB (без кэша модели)

### Оптимизации
- Асинхронная обработка с asyncio
- Пулы соединений с Redis
- Кэширование векторов
- Batch processing для обучения

## 🔒 Безопасность

### Аутентификация модераторов
```python
# Добавьте в main.py
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/feedback")
async def moderator_feedback(
    feedback: ModeratorFeedback,
    token: str = Depends(security)
):
    # Проверка токена модератора
    verify_moderator_token(token)
    # ...
```

### Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/classify")
@limiter.limit("100/minute")
async def classify_post(request: Request, post: DevToPost):
    # ...
```

## 🚀 Деплой в продакшн

### Docker Swarm
```bash
docker stack deploy -c docker-compose.prod.yml spam-classifier
```

### Kubernetes
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spam-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: spam-classifier
  template:
    metadata:
      labels:
        app: spam-classifier
    spec:
      containers:
      - name: api
        image: spam-classifier:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
```

### Nginx конфигурация
```nginx
upstream api_servers {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    listen 80;
    server_name spam-classifier.example.com;
    
    location / {
        proxy_pass http://api_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📋 API документация

После запуска приложения, документация доступна по адресам:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 Участие в разработке

### Структура проекта
```
spam-classifier/
├── main.py                 # FastAPI приложение
├── train_model.py          # Скрипт обучения
├── test_api.py            # Тесты
├── requirements.txt       # Python зависимости
├── docker-compose.yml     # Docker конфигурация
├── Dockerfile            # Docker образ
├── redis.conf            # Redis конфигурация
├── nginx.conf            # Nginx конфигурация
├── k8s/                  # Kubernetes манифесты
├── monitoring/           # Prometheus/Grafana конфигурация
└── docs/                 # Дополнительная документация
```

### Workflow разработки
1. Форкните репозиторий
2. Создайте feature branch
3. Добавьте тесты для новой функциональности
4. Убедитесь, что все тесты проходят
5. Создайте Pull Request

### Code Style
Проект использует:
- **Black** для форматирования
- **isort** для сортировки импортов
- **mypy** для проверки типов
- **flake8** для линтинга

```bash
# Проверка стиля кода
black --check .
isort --check-only .
mypy .
flake8 .
```

## 🐛 Известные проблемы

1. **Холодный старт модели** - первая загрузка Sentence Transformers может занять время
2. **Память для векторов** - большие датасеты требуют много RAM
3. **Rate limits dev.to** - ограничения API при сборе данных

## 🗺️ Roadmap

### v2.0
- [ ] Поддержка других источников (Reddit, HackerNews)
- [ ] Трансформеры вместо k-NN
- [ ] Объяснимость предсказаний (LIME/SHAP)
- [ ] A/B тестирование моделей

### v2.1  
- [ ] Поддержка изображений в постах
- [ ] Детекция плагиата
- [ ] Интеграция с Slack/Discord для уведомлений
- [ ] Мультиязычная поддержка

## 📄 Лицензия

MIT License - см. файл [LICENSE](LICENSE)

## 🙋‍♂️ Поддержка

- **Issues**: [GitHub Issues](https://github.com/your-username/devto-spam-classifier/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/devto-spam-classifier/discussions)
- **Email**: support@spam-classifier.com

## 📚 Дополнительные ресурсы

- [Redis Vector Search Documentation](https://redis.io/docs/interact/search-and-query/advanced-concepts/vectors/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Dev.to API Documentation](https://developers.forem.com/api/)

---

**Сделано с ❤️ для сообщества разработчиков**
