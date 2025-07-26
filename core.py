from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
import aioredis
import requests
from collections import Counter

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

class RedisVectorClassifier:
    def __init__(self):
        self.redis_client = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_dim = 384 + 3  # размерность эмбеддинга + 3 числовых признака
        self.index_name = "post_vectors"

    async def init_redis(self):
        """Асинхронная инициализация Redis и создание индекса"""
        try:
            # Добавляем таймаут, чтобы приложение не висело долго
            self.redis_client = await asyncio.wait_for(
                aioredis.from_url(
                    "redis://localhost:6379",
                    decode_responses=False
                ),
                timeout=2.0  # 2 секунды на подключение
            )
            await self.redis_client.ping() # Проверяем, что соединение живое
            logger.info("Redis connection established")
            await self.create_index()
        except (aioredis.exceptions.RedisError, asyncio.TimeoutError) as e:
            logger.warning(f"Could not connect to Redis: {e}. Application will run in degraded mode (no Redis features).")
            self.redis_client = None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Redis initialization: {e}")
            self.redis_client = None

    async def create_index(self):
        """Создание индекса RediSearch для векторов"""
        if not self.redis_client:
            return
        try:
            # Проверяем, существует ли индекс
            await self.redis_client.execute_command("FT.INFO", self.index_name)
            logger.info(f"Index '{self.index_name}' already exists.")
        except aioredis.exceptions.ResponseError as e:
            # Индекс не существует, создаем его, если ошибка об этом
            if "Unknown Index name" in str(e) or "no such index" in str(e):
                logger.info(f"Index '{self.index_name}' not found. Creating it.")
                schema = (
                    "vector", "VECTOR", "HNSW", "6", "TYPE", "FLOAT32", "DIM", str(self.vector_dim), "DISTANCE_METRIC", "L2",
                    "label", "TAG"
                )
                await self.redis_client.execute_command(
                    "FT.CREATE", self.index_name, "ON", "HASH", "PREFIX", "1", "post:", "SCHEMA", *schema
                )
            else:
                # Если другая ошибка, то пробрасываем ее
                logger.error(f"An unexpected Redis error occurred: {e}")
                raise

    def preprocess_text(self, text: Optional[str]) -> str:
        """Очистка текста"""
        if not text:
            return ""
        
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
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
            # ВАЖНО: API статей не возвращает followers_count, это поле будет обновлено позже
            'user_followers': -1, # Используем -1 как индикатор, что данные не были загружены
            'user_id': post.user.get('id') if post.user else None # Правильное поле - 'id'
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
        
        # Проверяем подписчиков, только если мы смогли их получить
        if features['user_followers'] != -1 and features['user_followers'] < 10:
            indicators.append(f"Low follower count ({features['user_followers']})")
            
        return indicators

    async def vectorize_post(self, post: DevToPost) -> tuple[np.ndarray, Dict[str, Any]]:
        """Создание вектора из поста и возврат обновленных признаков"""
        features = self.create_features(post)
        
        user_followers = features.get('user_followers', -1)
        user_id = features.get('user_id')

        if user_id:
            try:
                loop = asyncio.get_event_loop()
                user_response = await loop.run_in_executor(
                    None, requests.get, f"https://dev.to/api/users/{user_id}"
                )
                if user_response.status_code == 200:
                    user_data = user_response.json()
                    user_followers = user_data.get('followers_count', 0)
                    features['user_followers'] = user_followers # Обновляем словарь!
                    logger.info(f"Successfully fetched followers for user {user_id}: {user_followers}")
                else:
                    logger.warning(f"Failed to fetch user data for user {user_id}")
            except Exception as e:
                logger.error(f"Error fetching user data: {e}")

        combined_text = f"{features['title']} {features['description']}"
        
        loop = asyncio.get_event_loop()
        text_vector = await loop.run_in_executor(
            None, self.model.encode, combined_text
        )
        
        numeric_features = np.array([
            features['reading_time'],
            user_followers if user_followers != -1 else 0, # Используем 0, если не удалось получить
            len(features['tags'])
        ], dtype=np.float32)
        
        numeric_features = numeric_features / (np.linalg.norm(numeric_features) + 1e-8)
        
        final_vector = np.concatenate([text_vector, numeric_features])
        
        return final_vector.astype(np.float32), features

    async def store_training_vector(self, post_id: int, vector: np.ndarray, label: int):
        """Сохранение вектора в Redis для обучения"""
        if not self.redis_client:
            logger.warning("Redis is not available. Skipping vector storage.")
            return
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
        if not self.redis_client:
            return []
        try:
            query = (
                f"*=>[KNN {k} @vector $blob AS score]"
            )
            results = await self.redis_client.execute_command(
                "FT.SEARCH", self.index_name, query, "PARAMS", "2", "blob", query_vector.tobytes(), "DIALECT", "2"
            )
            
            similar_posts = []
            # The response is a list: [count, doc1, [fields1], doc2, [fields2], ...]
            # We iterate through the documents, skipping the count at the beginning.
            for i in range(1, len(results), 2):
                try:
                    # post_key is like b'post:12345'
                    post_key = results[i]
                    post_id = post_key.decode('utf-8').split(':')[-1]
                    
                    # fields is a list like [b'score', b'0.123']
                    fields = results[i+1]
                    score_index = fields.index(b'score') + 1
                    score = float(fields[score_index])
                    
                    similar_posts.append((post_id, score))
                except (ValueError, IndexError, AttributeError) as e:
                    logger.warning(f"Could not parse a result from Redis search: {e}. Result item: {results[i]}")
                    continue
            
            return similar_posts
            
        except Exception as e:
            logger.error(f"Failed to find similar posts: {e}")
            return []


class RediSearchClassifier:
    def __init__(self, redis_classifier: RedisVectorClassifier, k: int = 9):
        self.redis_classifier = redis_classifier
        self.k = k
    
    async def predict(self, post: DevToPost) -> tuple[int, float, List[str]]:
        """Предсказание класса поста"""
        import time
        start_time = time.time()
        
        try:
            # Теперь получаем и вектор, и обновленные признаки
            query_vector, features = await self.redis_classifier.vectorize_post(post)
            
            # Если Redis доступен, ищем похожие посты
            if self.redis_classifier.redis_client:
                similar_posts = await self.redis_classifier.find_similar_posts(query_vector, self.k)
                logger.info(f"Found {len(similar_posts)} similar posts in Redis for post {post.id}")
                
                if similar_posts:
                    labels = []
                    for post_id, score in similar_posts:
                        label_bytes = await self.redis_classifier.redis_client.hget(f"post:{post_id}", "label")
                        if label_bytes:
                            labels.append(label_bytes.decode('utf-8'))
                    
                    if labels:
                        label_counts = Counter(labels)
                        predicted_label_str = label_counts.most_common(1)[0][0]
                        predicted_label = 1 if predicted_label_str == "spam" else 0
                        confidence = label_counts[predicted_label_str] / len(labels)
                        
                        # Логика для формирования развернутого ответа
                        if predicted_label == 1:
                            reasoning = ["Similar to known spam posts (via Redis)"]
                        else:
                            reasoning = ["Similar to legitimate posts (via Redis)"]
                        
                        # Добавляем эвристические индикаторы как дополнительную информацию
                        heuristic_indicators = self.redis_classifier.get_spam_indicators(features)
                        reasoning.extend(heuristic_indicators)
                        
                        return predicted_label, confidence, reasoning

            # Запасной вариант, если Redis недоступен или похожих постов не найдено
            # Используем признаки, полученные из vectorize_post
            spam_indicators = self.redis_classifier.get_spam_indicators(features)
            
            num_indicators = len(spam_indicators)
            
            if num_indicators == 0:
                is_spam = 0
                confidence = 0.8
            elif num_indicators == 1:
                is_spam = 0
                confidence = 0.6
            elif num_indicators == 2:
                is_spam = 1
                confidence = 0.7
            else:  # 3 и более
                is_spam = 1
                confidence = 0.9
            
            reasoning = spam_indicators if spam_indicators else ["Heuristic analysis based on post content."]
            if not self.redis_classifier.redis_client:
                reasoning.append("Redis is not connected, classification is based on heuristics only.")
            
            return int(is_spam), confidence, reasoning
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0, 0.5, [f"Error during prediction: {str(e)}"]
