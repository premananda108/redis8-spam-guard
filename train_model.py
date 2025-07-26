#!/usr/bin/env python3
"""
Скрипт для обучения модели классификации спама
Собирает данные с dev.to и создает обучающий датасет
"""

import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
import numpy as np
from core import DevToPost, RedisVectorClassifier, RediSearchClassifier
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DevToDataCollector:
    def __init__(self):
        self.base_url = "https://dev.to/api"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_articles(self, page: int = 1, per_page: int = 30, tag: str = None) -> List[Dict]:
        """Получение статей с dev.to"""
        url = f"{self.base_url}/articles"
        params = {
            'page': page,
            'per_page': per_page
        }
        if tag:
            params['tag'] = tag
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch articles: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
            return []
    
    async def fetch_article_details(self, article_id: int) -> Dict:
        """Получение детальной информации о статье"""
        url = f"{self.base_url}/articles/{article_id}"
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch article {article_id}: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching article {article_id}: {e}")
            return {}
    
    async def collect_training_data(self, num_pages: int = 50) -> List[Dict]:
        """Сбор данных для обучения"""
        all_articles = []
        
        # Собираем статьи по популярным тегам
        tags = ['python', 'javascript', 'react', 'nodejs', 'webdev', 'tutorial', 
                'beginners', 'programming', 'ai', 'career']
        
        for tag in tags:
            logger.info(f"Collecting articles for tag: {tag}")
            for page in range(1, min(num_pages // len(tags) + 1, 10)):
                articles = await self.fetch_articles(page=page, tag=tag)
                if not articles:
                    break
                
                all_articles.extend(articles)
                await asyncio.sleep(0.5)  # Rate limiting
        
        # Собираем также общие статьи
        logger.info("Collecting general articles")
        for page in range(1, 20):
            articles = await self.fetch_articles(page=page)
            if not articles:
                break
            
            all_articles.extend(articles)
            await asyncio.sleep(0.5)
        
        logger.info(f"Collected {len(all_articles)} articles")
        return all_articles

class SpamLabelGenerator:
    """Генератор меток для обучения на основе эвристик"""
    
    def __init__(self):
        self.spam_keywords = [
            'earn money', 'make money', 'get rich', 'free money', 'easy money',
            'click here', 'buy now', 'limited time', 'act now', 'urgent',
            'guaranteed', 'no risk', 'work from home', 'side hustle',
            'crypto trading', 'bitcoin profit', 'investment opportunity',
            'affiliate marketing', 'dropshipping course', 'make $1000'
        ]
        
        self.quality_indicators = [
            'tutorial', 'guide', 'how to', 'best practices', 'tips',
            'introduction to', 'getting started', 'deep dive',
            'comprehensive', 'step by step', 'beginner', 'advanced'
        ]
    
    def calculate_spam_score(self, article: Dict) -> float:
        """Расчет вероятности спама (0-1)"""
        score = 0.0
        
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        tags = [tag.lower() for tag in article.get('tag_list', [])]
        
        # Проверяем спам-слова в заголовке (высокий вес)
        for keyword in self.spam_keywords:
            if keyword in title:
                score += 0.3
            if keyword in description:
                score += 0.2
        
        # Проверяем качественные индикаторы (снижают вероятность спама)
        for indicator in self.quality_indicators:
            if indicator in title:
                score -= 0.2
            if indicator in description:
                score -= 0.1
        
        # Анализ метрик вовлеченности
        reading_time = article.get('reading_time_minutes', 0)
        reactions = article.get('public_reactions_count', 0)
        comments = article.get('comments_count', 0)
        
        # Очень короткие посты с низким вовлечением
        if reading_time < 2 and reactions < 5:
            score += 0.3
        
        # Хорошее вовлечение снижает вероятность спама
        if reactions > 50 or comments > 10:
            score -= 0.2
        
        # Анализ тегов
        if len(tags) > 10:  # Слишком много тегов
            score += 0.2
        
        suspicious_tags = ['money', 'earn', 'profit', 'investment', 'trading']
        for tag in tags:
            if any(sus_tag in tag for sus_tag in suspicious_tags):
                score += 0.1
        
        # Анализ автора
        user = article.get('user', {})
        if user:
            followers = user.get('followers_count', 0)
            if followers < 10:  # Новый пользователь
                score += 0.1
        
        # Анализ даты публикации
        try:
            published_at = datetime.fromisoformat(article.get('published_at', '').replace('Z', '+00:00'))
            days_old = (datetime.now(published_at.tzinfo) - published_at).days
            if days_old < 1:  # Очень свежий пост
                score += 0.1
        except:
            pass
        
        return min(max(score, 0.0), 1.0)  # Ограничиваем 0-1
    
    def generate_label(self, article: Dict) -> int:
        """Генерация метки (0 = не спам, 1 = спам)"""
        spam_score = self.calculate_spam_score(article)
        
        # Добавляем немного случайности для разнообразия
        noise = random.uniform(-0.1, 0.1)
        final_score = spam_score + noise
        
        return 1 if final_score > 0.5 else 0

class ModelTrainer:
    def __init__(self, classifier: RedisVectorClassifier):
        self.redis_classifier = classifier
        self.label_generator = SpamLabelGenerator()
    
    async def prepare_training_data(self, articles: List[Dict]) -> List[tuple]:
        """Подготовка данных для обучения"""
        training_data = []
        
        for article in articles:
            try:
                # Преобразуем в DevToPost
                post = DevToPost(**{
                    'id': article.get('id'),
                    'title': article.get('title', ''),
                    'description': article.get('description'),
                    'tag_list': article.get('tag_list', []),
                    'reading_time_minutes': article.get('reading_time_minutes', 0),
                    'public_reactions_count': article.get('public_reactions_count', 0),
                    'comments_count': article.get('comments_count', 0),
                    'user': article.get('user'),
                    'url': article.get('url'),
                    'published_at': article.get('published_at')
                })
                
                # Генерируем метку
                label = self.label_generator.generate_label(article)
                
                training_data.append((post, label))
                
            except Exception as e:
                logger.error(f"Error preparing training data for article {article.get('id')}: {e}")
                continue
        
        return training_data
    
    async def train_model(self, training_data: List[tuple]):
        """Обучение модели"""
        await self.redis_classifier.init_redis()
        
        logger.info(f"Training model with {len(training_data)} samples")
        
        spam_count = 0
        total_count = 0
        
        for post, label in training_data:
            try:
                # Векторизуем пост
                vector, _ = await self.redis_classifier.vectorize_post(post)
                
                # Сохраняем в Redis
                await self.redis_classifier.store_training_vector(post.id, vector, label)
                
                if label == 1:
                    spam_count += 1
                total_count += 1
                
                if total_count % 100 == 0:
                    logger.info(f"Processed {total_count} samples")
                
                # Небольшая задержка для предотвращения перегрузки
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error training on post {post.id}: {e}")
                continue
        
        logger.info(f"Training completed!")
        logger.info(f"Total samples: {total_count}")
        logger.info(f"Spam samples: {spam_count} ({spam_count/total_count*100:.1f}%)")
        logger.info(f"Non-spam samples: {total_count-spam_count} ({(total_count-spam_count)/total_count*100:.1f}%)")
    
    async def evaluate_model(self, test_data: List[tuple]) -> Dict[str, float]:
        """Оценка качества модели"""
        classifier = RediSearchClassifier(self.redis_classifier)
        
        true_positives = false_positives = true_negatives = false_negatives = 0
        
        for post, true_label in test_data:
            try:
                predicted_label, confidence, _ = await classifier.predict(post)
                
                if true_label == 1 and predicted_label == 1:
                    true_positives += 1
                elif true_label == 0 and predicted_label == 1:
                    false_positives += 1
                elif true_label == 0 and predicted_label == 0:
                    true_negatives += 1
                elif true_label == 1 and predicted_label == 0:
                    false_negatives += 1
                    
            except Exception as e:
                logger.error(f"Error evaluating post {post.id}: {e}")
                continue
        
        # Расчет метрик
        total = true_positives + false_positives + true_negatives + false_negatives
        accuracy = (true_positives + true_negatives) / max(total, 1)
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 1)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }

async def main(classifier: RedisVectorClassifier):
    """Основная функция обучения"""
    logger.info("Starting model training process")

    trainer = ModelTrainer(classifier)
    await trainer.redis_classifier.init_redis()

    if not trainer.redis_classifier.redis_client:
        logger.error("Redis is not available. The training process cannot continue without a Redis connection.")
        return
    
    # Сбор данных
    async with DevToDataCollector() as collector:
        logger.info("Collecting training data from dev.to")
        articles = await collector.collect_training_data(num_pages=30)
    
    if not articles:
        logger.error("No articles collected, exiting")
        return
    
    # Удаляем дубликаты
    unique_articles = {article['id']: article for article in articles}.values()
    articles = list(unique_articles)
    logger.info(f"Using {len(articles)} unique articles")
    
    # Подготовка данных
    training_data = await trainer.prepare_training_data(articles)
    
    if not training_data:
        logger.error("No training data prepared, exiting")
        return
    
    # Разделение на обучающую и тестовую выборки
    random.shuffle(training_data)
    split_index = int(len(training_data) * 0.8)
    train_data = training_data[:split_index]
    test_data = training_data[split_index:]
    
    logger.info(f"Training set: {len(train_data)} samples")
    logger.info(f"Test set: {len(test_data)} samples")
    
    # Обучение
    await trainer.train_model(train_data)
    
    # Оценка
    logger.info("Evaluating model performance")
    metrics = await trainer.evaluate_model(test_data)
    
    logger.info("Model evaluation results:")
    logger.info(f"Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"Precision: {metrics['precision']:.3f}")
    logger.info(f"Recall: {metrics['recall']:.3f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.3f}")
    
    # Сохранение результатов
    results = {
        'timestamp': datetime.now().isoformat(),
        'training_samples': len(train_data),
        'test_samples': len(test_data),
        'metrics': metrics
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Training completed successfully!")
    logger.info("Results saved to training_results.json")

if __name__ == "__main__":
    # Этот блок больше не будет выполняться при импорте
    # Для запуска из командной строки потребуется отдельный скрипт
    # или изменение логики в main.py для передачи классификатора
    pass

