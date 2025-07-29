#!/usr/bin/env python3
"""
Script to train the spam classification model.
It combines three data sources:
1. Live data from dev.to, heuristically labeled.
2. A local dataset of known spam (`spam_dataset.json`).
3. Moderator feedback stored in Redis.
"""

import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from core import DevToPost, RedisVectorClassifier, RediSearchClassifier
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# SECTION 1: DATA COLLECTION AND HEURISTIC LABELING
# ==============================================================================

class DevToDataCollector:
    """Fetches recent articles from the dev.to API."""
    def __init__(self):
        self.base_url = "https://dev.to/api"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_articles(self, page: int = 1, per_page: int = 30) -> List[Dict]:
        """Fetches a single page of articles."""
        url = f"{self.base_url}/articles/latest"
        params = {'page': page, 'per_page': per_page}
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch articles (page {page}): {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching articles (page {page}): {e}")
            return []

    async def collect_training_data(self, num_articles: int = 1000) -> List[Dict]:
        """Collects a target number of articles from dev.to."""
        all_articles = []
        per_page = 100 # Fetch more per page to be efficient
        num_pages = (num_articles + per_page - 1) // per_page
        
        logger.info(f"Starting data collection from dev.to, aiming for ~{num_articles} articles.")
        for page in range(1, num_pages + 1):
            logger.info(f"Fetching page {page}/{num_pages}...")
            articles = await self.fetch_articles(page=page, per_page=per_page)
            if not articles:
                logger.warning("No more articles returned from API. Stopping collection.")
                break
            all_articles.extend(articles)
            await asyncio.sleep(0.5)  # Basic rate limiting

        logger.info(f"Collected {len(all_articles)} articles from dev.to.")
        return all_articles

class SpamLabelGenerator:
    """
    Generates spam labels for articles based on a set of heuristics.
    This creates the "dirty" dataset for broad training.
    """
    def __init__(self):
        self.spam_keywords = [
            'earn money', 'make money', 'get rich', 'free money', 'click here', 
            'buy now', 'limited time', 'guaranteed', 'no risk', 'work from home', 
            'crypto trading', 'bitcoin profit', 'investment opportunity', 'urgent'
        ]
        self.quality_indicators = [
            'tutorial', 'guide', 'how to', 'best practices', 'deep dive', 
            'introduction', 'getting started', 'step-by-step'
        ]

    def calculate_spam_score(self, article: Dict) -> float:
        """Calculates a spam score based on heuristics."""
        score = 0.0
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        
        for keyword in self.spam_keywords:
            if keyword in title or keyword in description:
                score += 0.4
        
        for indicator in self.quality_indicators:
            if indicator in title:
                score -= 0.2
        
        if article.get('reading_time_minutes', 0) < 2 and article.get('public_reactions_count', 0) < 3:
            score += 0.3
            
        if not article.get('description'):
             score += 0.2

        user_followers = article.get('user', {}).get('followers_count', 0)
        if user_followers < 5:
            score += 0.15

        return min(max(score, 0.0), 1.0)

    def generate_label(self, article: Dict) -> int:
        """Generates a final label (0 or 1) based on the score."""
        spam_score = self.calculate_spam_score(article)
        return 1 if spam_score > 0.5 else 0

async def prepare_data_from_articles(articles: List[Dict], label_generator: SpamLabelGenerator) -> List[Tuple[DevToPost, int]]:
    """Converts raw article dicts into labeled training data."""
    prepared_data = []
    for article in articles:
        try:
            post = DevToPost(**article)
            label = label_generator.generate_label(article)
            prepared_data.append((post, label))
        except Exception as e:
            logger.error(f"Skipping invalid article from dev.to API (ID: {article.get('id')}): {e}")
    return prepared_data

# ==============================================================================
# SECTION 2: MODERATOR DATA AND MODEL TRAINING
# ==============================================================================

async def get_moderator_labeled_data(redis_client) -> List[Tuple[DevToPost, int]]:
    """Fetches posts labeled by moderators from Redis."""
    labeled_data = []
    logger.info("Searching for moderator-labeled posts in Redis...")
    try:
        # The client already decodes responses, so we work with strings
        feedback_keys = [key async for key in redis_client.scan_iter("feedback:*")]
        if not feedback_keys:
            logger.info("No moderator feedback found in Redis.")
            return []

        logger.info(f"Found {len(feedback_keys)} feedback entries. Fetching full post data...")
        
        # Create keys for post data
        post_data_keys = [f"post_data:{key.split(':')[1]}" for key in feedback_keys]
        
        # Fetch all data in batches
        all_feedback_json = await redis_client.mget(feedback_keys)
        all_post_json = await redis_client.mget(post_data_keys)

        for i, key in enumerate(feedback_keys):
            post_id = key.split(":")[1]
            post_json = all_post_json[i]
            feedback_json = all_feedback_json[i]

            if post_json and feedback_json:
                try:
                    post = DevToPost(**json.loads(post_json))
                    label = 1 if json.loads(feedback_json).get("is_spam") else 0
                    labeled_data.append((post, label))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Could not parse data for post {post_id}: {e}")
            else:
                 logger.warning(f"Missing feedback or post data for post ID {post_id}. Skipping.")

    except Exception as e:
        logger.error(f"An error occurred while fetching moderator data from Redis: {e}", exc_info=True)
    
    logger.info(f"Successfully loaded {len(labeled_data)} posts from moderator feedback.")
    return labeled_data

class ModelTrainer:
    def __init__(self, classifier: RedisVectorClassifier):
        self.redis_classifier = classifier
    
    async def train_model(self, training_data: List[Tuple[DevToPost, int]]):
        """Trains the model by vectorizing posts and storing them in Redis."""
        await self.redis_classifier.init_redis()
        logger.info(f"Starting training with {len(training_data)} samples...")
        
        spam_count = sum(1 for _, label in training_data if label == 1)
        total_count = len(training_data)
        logger.info(f"Dataset composition: {spam_count} SPAM ({spam_count/total_count*100:.1f}%), {total_count-spam_count} LEGIT ({(total_count-spam_count)/total_count*100:.1f}%)")

        processed_count = 0
        for post, label in training_data:
            try:
                vector, _ = await self.redis_classifier.vectorize_post(post)
                await self.redis_classifier.store_training_vector(post.id, vector, label, post.title, post.url)
                processed_count += 1
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count}/{total_count} samples...")
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error training on post {post.id}: {e}")
        
        logger.info("Training completed!")
    
    async def evaluate_model(self, test_data: List[Tuple[DevToPost, int]]) -> Dict[str, float]:
        """Evaluates the model's performance against a test set."""
        classifier = RediSearchClassifier(self.redis_classifier)
        true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0
        
        for post, true_label in test_data:
            try:
                predicted_label, _, _, _ = await classifier.predict(post)
                if true_label == 1 and predicted_label == 1: true_positives += 1
                elif true_label == 0 and predicted_label == 1: false_positives += 1
                elif true_label == 0 and predicted_label == 0: true_negatives += 1
                elif true_label == 1 and predicted_label == 0: false_negatives += 1
            except Exception as e:
                logger.error(f"Error evaluating post {post.id}: {e}")
        
        total = len(test_data)
        accuracy = (true_positives + true_negatives) / max(total, 1)
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 1)
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score,
                'true_positives': true_positives, 'false_positives': false_positives,
                'true_negatives': true_negatives, 'false_negatives': false_negatives}

# ==============================================================================
# SECTION 3: MAIN EXECUTION
# ==============================================================================

async def main(classifier: Optional[RedisVectorClassifier] = None):
    """Main training function."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler("training.log", mode='w')
    file_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.handlers = [file_handler]
    root_logger.setLevel(logging.INFO)

    logger.info("Starting model training process...")

    if classifier is None:
        classifier = RedisVectorClassifier()
    
    await classifier.init_redis()
    if not classifier.redis_client:
        logger.error("Redis is not available. Training process cannot continue.")
        return

    # --- DATA GATHERING ---
    # 1. Fetch live data and apply heuristics
    async with DevToDataCollector() as collector:
        live_articles = await collector.collect_training_data(num_articles=1000)
    label_generator = SpamLabelGenerator()
    heuristic_data = await prepare_data_from_articles(live_articles, label_generator)

    # 2. Load base spam data from JSON file
    base_spam_data = []
    try:
        with open('spam_dataset.json', 'r', encoding='utf-8') as f:
            for article in json.load(f):
                try:
                    base_spam_data.append((DevToPost(**article), 1))
                except Exception as e:
                    logger.error(f"Skipping invalid article in spam_dataset.json (ID: {article.get('id')}): {e}")
        logger.info(f"Loaded {len(base_spam_data)} articles from local spam dataset.")
    except FileNotFoundError:
        logger.warning("spam_dataset.json not found.")
    
    # 3. Load high-quality labeled data from Redis
    moderator_data = await get_moderator_labeled_data(classifier.redis_client)
    
    # --- DATA COMBINATION (with priorities) ---
    combined_data = {}
    logger.info("Combining data sources...")
    # Priority 1: Heuristic data (lowest priority)
    for post, label in heuristic_data:
        combined_data[post.id] = (post, label)
    logger.info(f"Size after adding heuristic data: {len(combined_data)}")
    
    # Priority 2: Base spam file
    for post, label in base_spam_data:
        combined_data[post.id] = (post, label)
    logger.info(f"Size after adding base spam data: {len(combined_data)}")

    # Priority 3: Moderator feedback (highest priority)
    for post, label in moderator_data:
        combined_data[post.id] = (post, label)
    logger.info(f"Final size after adding moderator feedback: {len(combined_data)}")

    final_training_data = list(combined_data.values())
    
    if not final_training_data:
        logger.error("No training data available from any source. Exiting.")
        return
    
    # --- TRAINING AND EVALUATION ---
    random.shuffle(final_training_data)
    split_index = int(len(final_training_data) * 0.85)
    train_set, test_set = final_training_data[:split_index], final_training_data[split_index:]
    
    if not test_set and train_set:
        test_set.append(train_set.pop())
    elif not train_set:
        logger.error("Not enough data to create a training set. Aborting.")
        return

    logger.info(f"Training set size: {len(train_set)}, Test set size: {len(test_set)}")
    
    trainer = ModelTrainer(classifier)
    await trainer.train_model(train_set)
    
    logger.info("Evaluating model performance on the test set...")
    metrics = await trainer.evaluate_model(test_set)
    
    logger.info("--- Model Evaluation Results ---")
    for key, value in metrics.items():
        logger.info(f"{key.replace('_', ' ').title():<18}: {value:.3f}" if isinstance(value, float) else f"{key.replace('_', ' ').title():<18}: {value}")
    logger.info("---------------------------------")
    
    results = {'timestamp': datetime.now().isoformat(), 'training_samples': len(train_set), 'test_samples': len(test_set), 'metrics': metrics}
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Training process completed successfully! Results saved to training_results.json")

if __name__ == "__main__":
    redis_classifier_instance = RedisVectorClassifier()
    asyncio.run(main(redis_classifier_instance))
