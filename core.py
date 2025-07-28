import os
import logging
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
from redis import asyncio as redis
import requests
from collections import Counter
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TagField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import aiohttp

# --- Singleton for SentenceTransformer Model ---
class ModelSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            logging.info("Loading SentenceTransformer model for the first time...")
            cls._instance = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._instance

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

class SimilarPostInfo(BaseModel):
    post_id: str
    title: str
    url: str
    score: float

class ClassificationResult(BaseModel):
    """Результат классификации поста."""
    post_id: int
    is_spam: bool
    confidence: float
    recommendation: str
    reasoning: List[str]
    processing_time_ms: float
    similar_posts: List[SimilarPostInfo] = []
    moderator_verdict: Optional[str] = None  # 'spam', 'legit', or None


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
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = None
        self.model = ModelSingleton.get_instance()
        self.vector_dim = 384 + 3
        self.index_name = "post_vectors"

    async def init_redis(self):
        """Инициализация асинхронного клиента Redis и проверка индекса."""
        if self.redis_client:
            return
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info(f"Successfully connected to Redis at {self.redis_url}")
            await self.create_index()
        except Exception as e:
            logger.error(f"Failed to connect to Redis or create index: {e}")
            self.redis_client = None

    async def create_index(self):
        """Создание индекса RediSearch, если он не существует."""
        if not self.redis_client:
            logger.error("Cannot create index, Redis client is not initialized.")
            return
        try:
            await self.redis_client.ft(self.index_name).info()
            logger.info(f"Index '{self.index_name}' already exists.")
        except Exception as e:
            error_message = str(e).lower()
            if "unknown index name" in error_message or "no such index" in error_message:
                logger.info(f"Index '{self.index_name}' not found, creating new one.")
                schema = (
                    VectorField("vector", "HNSW", {"TYPE": "FLOAT32", "DIM": self.vector_dim, "DISTANCE_METRIC": "COSINE"}),
                    TagField("label"),
                    TextField("title"),
                    TagField("url")
                )
                definition = IndexDefinition(prefix=["post:"], index_type=IndexType.HASH)
                await self.redis_client.ft(self.index_name).create_index(fields=schema, definition=definition)
                logger.info(f"Index '{self.index_name}' created successfully.")
            else:
                logger.error(f"An unexpected error occurred while checking for index: {e}")
                raise

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
            'user_followers': -1,
            'user_id': post.user.get('id') if post.user else None
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
                    features['user_followers'] = user_followers
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
            user_followers if user_followers != -1 else 0,
            len(features['tags'])
        ], dtype=np.float32)
        
        numeric_features = numeric_features / (np.linalg.norm(numeric_features) + 1e-8)
        
        final_vector = np.concatenate([text_vector, numeric_features])
        
        return final_vector.astype(np.float32), features

    async def store_training_vector(self, post_id: int, vector: np.ndarray, label: int, title: str, url: str):
        """Сохранение вектора в Redis для обучения"""
        if not self.redis_client:
            logger.warning("Redis is not available. Skipping vector storage.")
            return
        try:
            post_key = f"post:{post_id}"
            await self.redis_client.hset(post_key, mapping={
                "vector": vector.tobytes(),
                "label": "spam" if label == 1 else "not_spam",
                "title": title or "",
                "url": url or ""
            })
            logger.info(f"Stored training vector for post {post_id}")
        except Exception as e:
            logger.error(f"Failed to store training vector: {e}")
            raise

    async def find_similar_posts(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Поиск похожих постов"""
        if not self.redis_client:
            return []
        try:
            query = (
                f"*=>[KNN {k} @vector $blob AS score]"
            )
            results = await self.redis_client.execute_command(
                "FT.SEARCH", self.index_name, query, "PARAMS", "2", "blob", query_vector.tobytes(), "DIALECT", "2", "RETURN", "3", "score", "title", "url"
            )
            
            similar_posts = []
            for i in range(1, len(results), 2):
                try:
                    post_key = results[i].decode('utf-8')
                    post_id = post_key.split(':')[-1]
                    
                    fields = results[i+1]
                    
                    fields_dict = {fields[j].decode('utf-8'): fields[j+1].decode('utf-8') for j in range(0, len(fields), 2)}
                    
                    similar_posts.append({
                        "post_id": post_id,
                        "score": float(fields_dict.get('score', 0.0)),
                        "title": fields_dict.get('title', 'No Title'),
                        "url": fields_dict.get('url', '')
                    })

                except (ValueError, IndexError, AttributeError, UnicodeDecodeError) as e:
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
    
    async def predict(self, post: DevToPost) -> tuple[int, float, List[str], List[SimilarPostInfo]]:
        """Предсказание класса поста"""
        import time
        start_time = time.time()
        
        try:
            query_vector, features = await self.redis_classifier.vectorize_post(post)
            
            similar_posts_info = []

            if self.redis_classifier.redis_client:
                similar_posts = await self.redis_classifier.find_similar_posts(query_vector, self.k)
                logger.info(f"Found {len(similar_posts)} similar posts in Redis for post {post.id}")
                
                if similar_posts:
                    labels = []
                    for post_data in similar_posts:
                        similar_posts_info.append(SimilarPostInfo(**post_data))
                        label_bytes = await self.redis_classifier.redis_client.hget(f"post:{post_data['post_id']}", "label")
                        if label_bytes:
                            labels.append(label_bytes.decode('utf-8'))
                    
                    if labels:
                        label_counts = Counter(labels)
                        predicted_label_str = label_counts.most_common(1)[0][0]
                        predicted_label = 1 if predicted_label_str == "spam" else 0
                        confidence = label_counts[predicted_label_str] / len(labels)
                        
                        if predicted_label == 1:
                            reasoning = ["Similar to known spam posts (via Redis)"]
                        else:
                            reasoning = ["Similar to legitimate posts (via Redis)"]
                        
                        heuristic_indicators = self.redis_classifier.get_spam_indicators(features)
                        reasoning.extend(heuristic_indicators)
                        
                        return predicted_label, confidence, reasoning, similar_posts_info

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
            else:
                is_spam = 1
                confidence = 0.9
            
            reasoning = spam_indicators if spam_indicators else ["Heuristic analysis based on post content."]
            if not self.redis_classifier.redis_client:
                reasoning.append("Redis is not connected, classification is based on heuristics only.")
            
            return int(is_spam), confidence, reasoning, []
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0, 0.5, [f"Error during prediction: {str(e)}"], []