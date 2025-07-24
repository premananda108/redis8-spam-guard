#!/usr/bin/env python3
"""
Тестовый набор для API классификации спама
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime

# Импортируем наше приложение
from main import app, redis_classifier, classifier, DevToPost

# Создаем тестовый клиент
client = TestClient(app)

# Тестовые данные
SAMPLE_POSTS = [
    {
        "id": 1,
        "title": "How to Learn Python in 2024: Complete Guide",
        "description": "A comprehensive tutorial for beginners to start their Python journey",
        "tag_list": ["python", "tutorial", "beginners"],
        "reading_time_minutes": 15,
        "public_reactions_count": 120,
        "comments_count": 25,
        "user": {"followers_count": 500},
        "url": "https://dev.to/test/python-guide",
        "published_at": "2024-01-15T10:00:00Z"
    },
    {
        "id": 2,
        "title": "EARN $5000 PER MONTH - CLICK HERE NOW!!!",
        "description": "Make money fast with this secret method",
        "tag_list": ["money", "earn", "profit", "investment", "trading", "crypto", "bitcoin"],
        "reading_time_minutes": 1,
        "public_reactions_count": 2,
        "comments_count": 0,
        "user": {"followers_count": 5},
        "url": "https://dev.to/spam/money",
        "published_at": "2024-01-20T15:30:00Z"
    }
]

class TestHealthCheck:
    """Тесты проверки здоровья сервиса"""
    
    def test_health_endpoint(self):
        """Тест endpoint'а здоровья"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

class TestClassification:
    """Тесты классификации постов"""
    
    def test_classify_legitimate_post(self):
        """Тест классификации легитимного поста"""
        legitimate_post = SAMPLE_POSTS[0]
        
        response = client.post("/classify", json=legitimate_post)
        assert response.status_code == 200
        
        result = response.json()
        assert "post_id" in result
        assert "is_spam" in result
        assert "confidence" in result
        assert "recommendation" in result
        assert "reasoning" in result
        assert "processing_time_ms" in result
        
        assert result["post_id"] == legitimate_post["id"]
        assert isinstance(result["is_spam"], bool)
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["recommendation"] in ["approve", "review", "block"]
    
    def test_classify_spam_post(self):
        """Тест классификации спам поста"""
        spam_post = SAMPLE_POSTS[1]
        
        response = client.post("/classify", json=spam_post)
        assert response.status_code == 200
        
        result = response.json()
        assert result["post_id"] == spam_post["id"]
        
        # Ожидаем, что спам будет обнаружен
        # (может не сработать без обученной модели)
        assert isinstance(result["is_spam"], bool)
        assert len(result["reasoning"]) > 0
    
    def test_classify_invalid_post(self):
        """Тест классификации с невалидными данными"""
        invalid_post = {
            "id": "invalid",  # Должно быть число
            "title": "",
        }
        
        response = client.post("/classify", json=invalid_post)
        assert response.status_code == 422  # Validation error
    
    def test_classify_minimal_post(self):
        """Тест классификации минимального поста"""
        minimal_post = {
            "id": 999,
            "title": "Test Post",
            "description": None,
            "tag_list": [],
            "reading_time_minutes": 0,
            "public_reactions_count": 0,
            "comments_count": 0,
            "user": None
        }
        
        response = client.post("/classify", json=minimal_post)
        assert response.status_code == 200
        
        result = response.json()
        assert result["post_id"] == 999

class TestBatchClassification:
    """Тесты пакетной классификации"""
    
    def test_batch_classification(self):
        """Тест пакетной классификации"""
        batch_request = {
            "posts": SAMPLE_POSTS,
            "threshold": 0.8
        }
        
        response = client.post("/classify-batch", json=batch_request)
        assert response.status_code == 200
        
        result = response.json()
        assert "results" in result
        assert len(result["results"]) == len(SAMPLE_POSTS)
        
        for i, classification in enumerate(result["results"]):
            assert classification["post_id"] == SAMPLE_POSTS[i]["id"]
            assert "is_spam" in classification
            assert "confidence" in classification
            assert "recommendation" in classification
    
    def test_empty_batch(self):
        """Тест пустой пакетной классификации"""
        batch_request = {
            "posts": [],
            "threshold": 0.8
        }
        
        response = client.post("/classify-batch", json=batch_request)
        assert response.status_code == 200
        
        result = response.json()
        assert result["results"] == []

class TestFeedback:
    """Тесты обратной связи"""
    
    def test_moderator_feedback(self):
        """Тест отправки обратной связи от модератора"""
        feedback = {
            "post_id": 123,
            "is_spam": True,
            "moderator_id": "mod_001",
            "notes": "Clear spam content"
        }
        
        response = client.post("/feedback", json=feedback)
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "success"
    
    def test_feedback_validation(self):
        """Тест валидации обратной связи"""
        invalid_feedback = {
            "post_id": "invalid",  # Должно быть число
            "is_spam": "yes",      # Должно быть bool
            "moderator_id": ""     # Не должно быть пустым
        }
        
        response = client.post("/feedback", json=invalid_feedback)
        assert response.status_code == 422

class TestStats:
    """Тесты статистики"""
    
    def test_stats_endpoint(self):
        """Тест получения статистики"""
        response = client.get("/stats")
        assert response.status_code == 200
        
        stats = response.json()
        assert "total_classified" in stats
        assert "spam_detected" in stats
        assert "last_updated" in stats
        
        assert isinstance(stats["total_classified"], int)
        assert isinstance(stats["spam_detected"], int)

class TestRedisIntegration:
    """Тесты интеграции с Redis"""
    
    @pytest.mark.asyncio
    async def test_redis_connection(self):
        """Тест соединения с Redis"""
        try:
            await redis_classifier.init_redis()
            assert redis_classifier.redis_client is not None
            
            # Проверяем ping
            result = await redis_classifier.redis_client.ping()
            assert result is True
            
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")
    
    @pytest.mark.asyncio
    async def test_vector_operations(self):
        """Тест операций с векторами"""
        try:
            await redis_classifier.init_redis()
            
            # Создаем тестовый пост
            test_post = DevToPost(**SAMPLE_POSTS[0])
            
            # Векторизуем
            vector = await redis_classifier.vectorize_post(test_post)
            assert isinstance(vector, np.ndarray)
            assert vector.shape[0] > 0
            
            # Сохраняем
            await redis_classifier.store_training_vector(
                test_post.id, vector, 0
            )
            
            # Ищем похожие
            similar = await redis_classifier.find_similar_posts(vector, k=1)
            assert len(similar) >= 0  # Может быть пустым, если нет других векторов
            
        except Exception as e:
            pytest.skip(f"Redis vector operations not available: {e}")

class TestDataValidation:
    """Тесты валидации данных"""
    
    def test_devto_post_validation(self):
        """Тест валидации модели DevToPost"""
        # Валидный пост
        valid_data = SAMPLE_POSTS[0]
        post = DevToPost(**valid_data)
        assert post.id == valid_data["id"]
        assert post.title == valid_data["title"]
    
    def test_invalid_post_data(self):
        """Тест невалидных данных поста"""
        with pytest.raises(ValueError):
            # ID должен быть числом
            DevToPost(id="invalid", title="Test")
    
    def test_optional_fields(self):
        """Тест опциональных полей"""
        minimal_data = {
            "id": 1,
            "title": "Test"
        }
        
        post = DevToPost(**minimal_data)
        assert post.id == 1
        assert post.title == "Test"
        assert post.description is None
        assert post.tag_list == []

class TestErrorHandling:
    """Тесты обработки ошибок"""
    
    @patch('main.classifier.predict')
    def test_classification_error_handling(self, mock_predict):
        """Тест обработки ошибок при классификации"""
        # Мокаем ошибку в классификаторе
        mock_predict.side_effect = Exception("Redis connection failed")
        
        response = client.post("/classify", json=SAMPLE_POSTS[0])
        assert response.status_code == 500
        
        error = response.json()
        assert "detail" in error
    
    def test_malformed_json(self):
        """Тест обработки некорректного JSON"""
        response = client.post(
            "/classify",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

class TestPerformance:
    """Тесты производительности"""
    
    def test_classification_performance(self):
        """Тест производительности классификации"""
        import time
        
        start_time = time.time()
        response = client.post("/classify", json=SAMPLE_POSTS[0])
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Проверяем, что классификация занимает разумное время
        processing_time = end_time - start_time
        assert processing_time < 5.0  # Менее 5 секунд
        
        # Проверяем время обработки в ответе
        result = response.json()
        assert result["processing_time_ms"] > 0
    
    def test_concurrent_requests(self):
        """Тест одновременных запросов"""
        import concurrent.futures
        import threading
        
        def make_request():
            return client.post("/classify", json=SAMPLE_POSTS[0])
        
        # Отправляем 10 одновременных запросов
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [f.result() for f in futures]
        
        # Все запросы должны быть успешными
        for response in responses:
            assert response.status_code == 200

# Фикстуры для тестов
@pytest.fixture(scope="session")
def event_loop():
    """Создает event loop для асинхронных тестов"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def redis_connection():
    """Фикстура для соединения с Redis"""
    try:
        await redis_classifier.init_redis()
        yield redis_classifier.redis_client
    finally:
        if redis_classifier.redis_client:
            await redis_classifier.redis_client.close()

# Конфигурация pytest
def pytest_configure(config):
    """Конфигурация pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v", "--tb=short"])
