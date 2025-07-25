from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import re
import asyncio
import aioredis
from datetime import datetime
import json
import logging
from collections import Counter
import uvicorn

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic модели
class DevToPost(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    tag_list: List[str] = []
    reading_time_minutes: int = 0
    public_reactions_count: int = 0
    comments_count: int = 0
    user: Optional[Dict[str, Any]] = None
    url: Optional[str] = None
    published_at: Optional[str] = None

class ClassificationResult(BaseModel):
    post_id: int
    is_spam: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommendation: str
    reasoning: List[str] = []
    processing_time_ms: float

class ModeratorFeedback(BaseModel):
    post_id: int
    is_spam: bool
    moderator_id: str
    notes: Optional[str] = None

class BatchClassificationRequest(BaseModel):
    posts: List[DevToPost]
    threshold: float = Field(default=0.8, ge=0.0, le=1.0)

class StatsResponse(BaseModel):
    total_classified: int
    spam_detected: int
    accuracy: Optional[float] = None
    last_updated: datetime

# FastAPI приложение
app = FastAPI(
    title="Dev.to Spam Classifier API",
    description="AI-powered spam detection for dev.to posts using Redis 8 Vector Sets",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RedisVectorClassifier:
    def __init__(self):
        self.redis_client = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_dim = 384 + 5  # размерность эмбеддинга + числовые признаки
        self.index_name = "post_vectors"

    async def init_redis(self):
        """Асинхронная инициализация Redis и создание индекса"""
        try:
            self.redis_client = await aioredis.from_url(
                "redis://localhost:6379",
                decode_responses=False
            )
            logger.info("Redis connection established")
            await self.create_index()
        except Exception as e:
            logger.error(f"Failed to connect to Redis or create index: {e}")
            raise

    async def create_index(self):
        """Создание индекса RediSearch для векторов"""
        try:
            # Проверяем, существует ли индекс
            await self.redis_client.execute_command("FT.INFO", self.index_name)
            logger.info(f"Index '{self.index_name}' already exists.")
        except aioredis.exceptions.ResponseError:
            # Индекс не существует, создаем его
            logger.info(f"Creating index '{self.index_name}'")
            schema = (
                "vector", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(self.vector_dim), "DISTANCE_METRIC", "L2",
                "label", "TAG"
            )
            await self.redis_client.execute_command(
                "FT.CREATE", self.index_name, "ON", "HASH", "PREFIX", "1", "post:", "SCHEMA", *schema
            )

    def preprocess_text(self, text: Optional[str]) -> str:
        """Очистка текста"""
        if not text:
            return ""
        
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower().strip()
        
        return text

    def create_features(self, post: DevToPost) -> Dict[str, Any]:
        """Создание признаков для классификации"""
        return {
            'title': self.preprocess_text(post.title),
            'description': self.preprocess_text(post.description),
            'tags': post.tag_list,
            'reading_time': post.reading_time_minutes,
            'reactions_count': post.public_reactions_count,
            'comments_count': post.comments_count,
            'user_followers': post.user.get('followers_count', 0) if post.user else 0
        }

    def get_spam_indicators(self, features: Dict[str, Any]) -> List[str]:
        """Определение индикаторов спама"""
        indicators = []
        
        if features['reading_time'] < 2 and features['reactions_count'] < 5:
            indicators.append("Short post with low engagement")
        
        spam_words = ['earn money', 'get rich', 'click here', 'free offer', 'buy now', 'limited time']
        title_lower = features['title'].lower()
        if any(word in title_lower for word in spam_words):
            indicators.append("Contains spam keywords")
        
        if not features['description']:
            indicators.append("Missing description")
        
        if len(features['tags']) > 10:
            indicators.append("Too many tags")
        
        if features['user_followers'] < 10:
            indicators.append("Low follower count")
            
        return indicators

    async def vectorize_post(self, post: DevToPost) -> np.ndarray:
        """Создание вектора из поста"""
        features = self.create_features(post)
        
        combined_text = f"{features['title']} {features['description']}"
        
        loop = asyncio.get_event_loop()
        text_vector = await loop.run_in_executor(
            None, self.model.encode, combined_text
        )
        
        numeric_features = np.array([
            features['reading_time'],
            features['reactions_count'],
            features['comments_count'],
            features['user_followers'],
            len(features['tags'])
        ], dtype=np.float32)
        
        numeric_features = numeric_features / (np.linalg.norm(numeric_features) + 1e-8)
        
        final_vector = np.concatenate([text_vector, numeric_features])
        
        return final_vector.astype(np.float32)

    async def store_training_vector(self, post_id: int, vector: np.ndarray, label: int):
        """Сохранение вектора в Redis для обучения"""
        try:
            post_key = f"post:{post_id}"
            await self.redis_client.hset(post_key, mapping={
                "vector": vector.tobytes(),
                "label": "spam" if label == 1 else "not_spam"
            })
            logger.info(f"Stored training vector for post {post_id}")
        except Exception as e:
            logger.error(f"Failed to store training vector: {e}")
            raise

    async def find_similar_posts(self, query_vector: np.ndarray, k: int = 5) -> List[tuple]:
        """Поиск похожих постов"""
        try:
            query = (
                f"*=>[KNN {k} @vector $blob AS score]"
            )
            results = await self.redis_client.execute_command(
                "FT.SEARCH", self.index_name, query, "PARAMS", "2", "blob", query_vector.tobytes(), "DIALECT", "2"
            )
            
            similar_posts = []
            # The first result is the total number of documents, so we skip it.
            for i in range(1, len(results)):
                doc = results[i]
                if isinstance(doc, list):
                    post_id = doc[0].decode('utf-8').split(':')[-1]
                    # The score is in the field list, after the 'score' key
                    try:
                        score_index = doc[1].index(b'score') + 1
                        score = float(doc[1][score_index])
                        similar_posts.append((post_id, score))
                    except (ValueError, IndexError):
                        continue
            
            return similar_posts
            
        except Exception as e:
            logger.error(f"Failed to find similar posts: {e}")
            return []


class RediSearchClassifier:
    def __init__(self, redis_classifier: RedisVectorClassifier, k: int = 5):
        self.redis_classifier = redis_classifier
        self.k = k
    
    async def predict(self, post: DevToPost) -> tuple[int, float, List[str]]:
        """Предсказание класса поста"""
        import time
        start_time = time.time()
        
        try:
            query_vector = await self.redis_classifier.vectorize_post(post)
            
            similar_posts = await self.redis_classifier.find_similar_posts(query_vector, self.k)
            
            if not similar_posts:
                features = self.redis_classifier.create_features(post)
                spam_indicators = self.redis_classifier.get_spam_indicators(features)
                is_spam = len(spam_indicators) >= 2
                confidence = 0.6 if is_spam else 0.7
                
                processing_time = (time.time() - start_time) * 1000
                return int(is_spam), confidence, spam_indicators
            
            labels = []
            for post_id, score in similar_posts:
                label = await self.redis_classifier.redis_client.hget(f"post:{post_id}", "label")
                if label:
                    labels.append(label.decode('utf-8'))
            
            if not labels:
                return 0, 0.5, ["No training data available"]
            
            label_counts = Counter(labels)
            predicted_label_str = label_counts.most_common(1)[0][0]
            predicted_label = 1 if predicted_label_str == "spam" else 0
            confidence = label_counts[predicted_label_str] / len(labels)
            
            features = self.redis_classifier.create_features(post)
            reasoning = self.redis_classifier.get_spam_indicators(features)
            if not reasoning and predicted_label == 1:
                reasoning = ["Similar to known spam posts"]
            elif not reasoning:
                reasoning = ["Similar to legitimate posts"]
            
            processing_time = (time.time() - start_time) * 1000
            return predicted_label, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0, 0.5, [f"Error during prediction: {str(e)}"]

# Глобальные объекты
redis_classifier = RedisVectorClassifier()
classifier = RediSearchClassifier(redis_classifier)

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    await redis_classifier.init_redis()
    logger.info("Application started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при остановке"""
    if redis_classifier.redis_client:
        await redis_classifier.redis_client.close()
    logger.info("Application shut down")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница с интерфейсом"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dev.to Spam Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .post { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
            .spam { background-color: #ffebee; border-color: #f44336; }
            .not-spam { background-color: #e8f5e8; border-color: #4caf50; }
            .uncertain { background-color: #fff3e0; border-color: #ff9800; }
            .reasoning { font-size: 0.9em; color: #666; margin-top: 10px; }
            .controls { margin: 20px 0; }
            button { padding: 10px 20px; margin: 5px; cursor: pointer; }
            .loading { opacity: 0.6; }
        </style>
    </head>
    <body>
        <h1>🚀 Dev.to Post Moderator Assistant</h1>
        <p>AI-powered spam detection using Redis 8 Vector Sets</p>
        
        <div class="controls">
            <button onclick="loadAndClassifyPosts()">🔄 Load Latest Posts</button>
            <button onclick="showStats()">📊 Show Statistics</button>
        </div>
        
        <div id="stats-container"></div>
        <div id="posts-container"></div>
        
        <script>
            async function classifyPost(postData) {
                const response = await fetch('/classify', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(postData)
                });
                
                if (!response.ok) {
                    throw new Error('Classification failed');
                }
                
                return await response.json();
            }
            
            async function loadAndClassifyPosts() {
                const container = document.getElementById('posts-container');
                container.innerHTML = '<p>Loading posts...</p>';
                
                try {
                    // Загружаем посты с dev.to
                    const response = await fetch('https://dev.to/api/articles?per_page=10');
                    const posts = await response.json();
                    
                    container.innerHTML = '';
                    
                    for (const post of posts) {
                        const postElement = document.createElement('div');
                        postElement.className = 'post loading';
                        postElement.innerHTML = `
                            <h3>${post.title}</h3>
                            <p>${post.description || 'No description'}</p>
                            <div><strong>Tags:</strong> ${post.tag_list.join(', ')}</div>
                            <div><strong>Reading time:</strong> ${post.reading_time_minutes} min</div>
                            <div>🔄 Classifying...</div>
                        `;
                        container.appendChild(postElement);
                        
                        try {
                            const classification = await classifyPost(post);
                            
                            postElement.className = `post ${classification.is_spam ? 'spam' : 
                                (classification.confidence > 0.8 ? 'not-spam' : 'uncertain')}`;
                            
                            const statusEmoji = classification.is_spam ? '🚫' : '✅';
                            const recommendationEmoji = {
                                'block': '🚫',
                                'review': '⚠️',
                                'approve': '✅'
                            }[classification.recommendation] || '❓';
                            
                            postElement.innerHTML = `
                                <h3>${post.title}</h3>
                                <p>${post.description || 'No description'}</p>
                                <div><strong>Tags:</strong> ${post.tag_list.join(', ')}</div>
                                <div><strong>Reading time:</strong> ${post.reading_time_minutes} min</div>
                                <div><strong>Reactions:</strong> ${post.public_reactions_count}</div>
                                <hr>
                                <div>
                                    <strong>${statusEmoji} Classification:</strong> ${classification.is_spam ? 'SPAM' : 'LEGITIMATE'}
                                    (${(classification.confidence * 100).toFixed(1)}% confidence)
                                </div>
                                <div>
                                    <strong>${recommendationEmoji} Recommendation:</strong> ${classification.recommendation.toUpperCase()}
                                </div>
                                <div><strong>⏱️ Processing time:</strong> ${classification.processing_time_ms.toFixed(1)}ms</div>
                                ${classification.reasoning.length > 0 ? `
                                    <div class="reasoning">
                                        <strong>🧠 Reasoning:</strong>
                                        <ul>${classification.reasoning.map(r => `<li>${r}</li>`).join('')}</ul>
                                    </div>
                                ` : ''}
                            `;
                        } catch (error) {
                            postElement.innerHTML += `<div style="color: red;">Error: ${error.message}</div>`;
                        }
                    }
                } catch (error) {
                    container.innerHTML = `<p style="color: red;">Error loading posts: ${error.message}</p>`;
                }
            }
            
            async function showStats() {
                try {
                    const response = await fetch('/stats');
                    const stats = await response.json();
                    
                    const statsContainer = document.getElementById('stats-container');
                    statsContainer.innerHTML = `
                        <div style="background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0;">
                            <h3>📊 Classification Statistics</h3>
                            <p><strong>Total classified:</strong> ${stats.total_classified}</p>
                            <p><strong>Spam detected:</strong> ${stats.spam_detected}</p>
                            <p><strong>Spam rate:</strong> ${(stats.spam_detected / Math.max(stats.total_classified, 1) * 100).toFixed(1)}%</p>
                            <p><strong>Last updated:</strong> ${new Date(stats.last_updated).toLocaleString()}</p>
                        </div>
                    `;
                } catch (error) {
                    console.error('Failed to load stats:', error);
                }
            }
            
            // Автоматически загружаем посты при загрузке страницы
            document.addEventListener('DOMContentLoaded', loadAndClassifyPosts);
        </script>
    </body>
    </html>
    """

@app.post("/classify", response_model=ClassificationResult)
async def classify_post(post: DevToPost):
    """Классификация одного поста"""
    import time
    start_time = time.time()
    
    try:
        prediction, confidence, reasoning = await classifier.predict(post)
        
        # Определяем рекомендацию
        if confidence >= 0.8:
            recommendation = "block" if prediction == 1 else "approve"
        else:
            recommendation = "review"
        
        processing_time = (time.time() - start_time) * 1000
        
        result = ClassificationResult(
            post_id=post.id,
            is_spam=bool(prediction),
            confidence=confidence,
            recommendation=recommendation,
            reasoning=reasoning,
            processing_time_ms=processing_time
        )
        
        # Логируем для статистики
        logger.info(f"Classified post {post.id}: {'SPAM' if prediction else 'OK'} ({confidence:.2f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Classification failed for post {post.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify-batch")
async def classify_batch(request: BatchClassificationRequest):
    """Пакетная классификация постов"""
    results = []
    
    for post in request.posts:
        try:
            prediction, confidence, reasoning = await classifier.predict(post)
            
            recommendation = "block" if prediction == 1 and confidence >= request.threshold else \
                           "approve" if prediction == 0 and confidence >= request.threshold else "review"
            
            results.append({
                "post_id": post.id,
                "is_spam": bool(prediction),
                "confidence": confidence,
                "recommendation": recommendation,
                "reasoning": reasoning
            })
            
        except Exception as e:
            logger.error(f"Batch classification failed for post {post.id}: {e}")
            results.append({
                "post_id": post.id,
                "error": str(e)
            })
    
    return {"results": results}

@app.post("/feedback")
async def moderator_feedback(feedback: ModeratorFeedback):
    """Обратная связь от модератора для улучшения модели"""
    try:
        # Сохраняем обратную связь
        feedback_key = f"feedback:{feedback.post_id}"
        feedback_data = {
            "post_id": feedback.post_id,
            "is_spam": feedback.is_spam,
            "moderator_id": feedback.moderator_id,
            "notes": feedback.notes,
            "timestamp": datetime.now().isoformat()
        }
        
        await redis_classifier.redis_client.set(
            feedback_key, 
            json.dumps(feedback_data)
        )
        
        logger.info(f"Received feedback for post {feedback.post_id} from {feedback.moderator_id}")
        
        return {"status": "success", "message": "Feedback recorded"}
    
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Статистика классификации"""
    try:
        # Здесь можно добавить реальную статистику из Redis
        # Для демо возвращаем примерные данные
        
        stats = StatsResponse(
            total_classified=150,
            spam_detected=23,
            accuracy=0.94,
            last_updated=datetime.now()
        )
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    try:
        # Проверяем соединение с Redis
        await redis_classifier.redis_client.ping()
        
        return {
            "status": "healthy",
            "redis": "connected",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
