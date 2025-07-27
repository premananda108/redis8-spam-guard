import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from datetime import datetime
import json
import logging
import uvicorn
from train_model import main as run_training
from core import (
    DevToPost, ClassificationResult, ModeratorFeedback, 
    BatchClassificationRequest, StatsResponse,
    RedisVectorClassifier, RediSearchClassifier
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            body { font-family: Arial, sans-serif; margin: 20px; display: flex; gap: 40px; }
            .main-content { flex: 2; }
            .sidebar { flex: 1; max-width: 400px; }
            .post { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
            .spam { background-color: #ffebee; border-color: #f44336; }
            .not-spam { background-color: #e8f5e8; border-color: #4caf50; }
            .uncertain { background-color: #fff3e0; border-color: #ff9800; }
            .reasoning { font-size: 0.9em; color: #666; margin-top: 10px; }
            .controls { margin: 20px 0; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
            button { padding: 10px 20px; margin: 5px; cursor: pointer; border-radius: 5px; border: 1px solid #ccc; background-color: #f0f0f0; }
            button:disabled { cursor: not-allowed; opacity: 0.5; }
            .loading { opacity: 0.6; }
            #page-indicator { font-weight: bold; }
            .manual-check-form { background-color: #f9f9f9; padding: 20px; border-radius: 8px; border: 1px solid #eee; }
            .manual-check-form h2 { margin-top: 0; }
            .manual-check-form label { display: block; margin-top: 10px; font-weight: bold; }
            .manual-check-form input, .manual-check-form textarea { width: 100%; padding: 8px; margin-top: 5px; border-radius: 4px; border: 1px solid #ccc; box-sizing: border-box; }
            #log-container { background-color: #222; color: #0f0; font-family: monospace; padding: 15px; height: 400px; overflow-y: scroll; border-radius: 5px; margin-top: 20px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <div class="main-content">
            <h1>🚀 Dev.to Post Moderator Assistant</h1>
            <p>AI-powered spam detection using Redis 8 Vector Sets</p>
            
            <div class="controls">
                <button onclick="resetAndLoad()">🔄 Load Latest Posts</button>
                <button onclick="showStats()">📊 Show Statistics</button>
                <button id="train-button" onclick="startTraining()">🚀 Train Model</button>
            </div>
            
            <div id="stats-container"></div>
            <div id="redis-info-container"></div>
            <div id="log-container-wrapper" style="display: none;">
                <h3>🎓 Training Logs</h3>
                <button onclick="refreshLogs()">🔄 Refresh Logs</button>
                <pre id="log-container"></pre>
            </div>
            <div id="posts-container"></div>
            
            <div id="pagination-controls" class="controls">
                <button id="prev-button" onclick="prevPage()">⬅️ Previous</button>
                <span id="page-indicator">Page 1</span>
                <button id="next-button" onclick="nextPage()">Next ➡️</button>
            </div>
        </div>

        <div class="sidebar">
            <div class="manual-check-form">
                <h2>Manual Post Check</h2>
                <p>Enter post details to get an instant diagnosis.</p>
                <label for="manual-title">Title:</label>
                <input type="text" id="manual-title" value="How to earn $1000 in a week!">
                
                <label for="manual-description">Description:</label>
                <textarea id="manual-description" rows="4">Click this link now for a limited time offer. You won't regret it. Best crypto strategy.</textarea>
                
                <label for="manual-tags">Tags (comma-separated):</label>
                <input type="text" id="manual-tags" value="money, crypto, investment, quick">
                
                <label for="manual-reactions">Reactions:</label>
                <input type="number" id="manual-reactions" value="1">
                
                <label for="manual-comments">Comments:</label>
                <input type="number" id="manual-comments" value="0">

                <label for="manual-reading-time">Reading Time (minutes):</label>
                <input type="number" id="manual-reading-time" value="1">

                <label for="manual-followers">Author's Followers:</label>
                <input type="number" id="manual-followers" value="5">
                
                <div class="controls">
                    <button id="manual-check-button" onclick="performManualCheck()">🔬 Check Post</button>
                </div>
                
                <div id="manual-check-result"></div>
            </div>
        </div>
        
        <script>
            let currentPage = 1;
            const postsPerPage = 10;

            async function classifyPost(postData) {
                const response = await fetch('/classify', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(postData)
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Classification failed');
                }
                
                return await response.json();
            }

            async function performManualCheck() {
                const button = document.getElementById('manual-check-button');
                const resultContainer = document.getElementById('manual-check-result');
                
                button.disabled = true;
                button.textContent = 'Checking...';
                resultContainer.innerHTML = '';

                const postData = {
                    id: Math.floor(Math.random() * 1000000), // Fake ID
                    title: document.getElementById('manual-title').value,
                    description: document.getElementById('manual-description').value,
                    tag_list: document.getElementById('manual-tags').value.split(',').map(t => t.trim()),
                    public_reactions_count: parseInt(document.getElementById('manual-reactions').value) || 0,
                    comments_count: parseInt(document.getElementById('manual-comments').value) || 0,
                    reading_time_minutes: parseInt(document.getElementById('manual-reading-time').value) || 0,
                    user: {
                        // We need to simulate the user object for the backend
                        id: Math.floor(Math.random() * 100000), // Fake user ID
                        followers_count: parseInt(document.getElementById('manual-followers').value) || 0
                    }
                };

                try {
                    const classification = await classifyPost(postData);
                    
                    const postElement = document.createElement('div');
                    postElement.className = `post ${classification.is_spam ? 'spam' : 
                        (classification.confidence > 0.8 ? 'not-spam' : 'uncertain')}`;
                    
                    const statusEmoji = classification.is_spam ? '🚫' : '✅';
                    const recommendationEmoji = {
                        'block': '🚫',
                        'review': '⚠️',
                        'approve': '✅'
                    }[classification.recommendation] || '❓';
                    
                    const similarPostsHtml = classification.similar_post_ids && classification.similar_post_ids.length > 0
                        ? `
                        <div class="reasoning">
                            <strong>🔗 Similar to posts:</strong>
                            <ul>
                                ${classification.similar_post_ids.map(id => `<li><a href="https://dev.to/api/articles/${id}" target="_blank">${id}</a></li>`).join('')}
                            </ul>
                        </div>
                        `
                        : '';

                    postElement.innerHTML = `
                        <h3>Diagnosis Result</h3>
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
                        ${similarPostsHtml}
                    `;
                    resultContainer.appendChild(postElement);

                } catch (error) {
                    resultContainer.innerHTML = `<div style="color: red;">Error: ${error.message}</div>`;
                } finally {
                    button.disabled = false;
                    button.textContent = '🔬 Check Post';
                }
            }
            
            async function loadAndClassifyPosts(page) {
                const container = document.getElementById('posts-container');
                const pageIndicator = document.getElementById('page-indicator');
                const prevButton = document.getElementById('prev-button');
                const nextButton = document.getElementById('next-button');

                container.innerHTML = `<p>Loading page ${page}...</p>`;
                pageIndicator.textContent = `Page ${page}`;
                prevButton.disabled = page <= 1;
                
                try {
                    const response = await fetch(`https://dev.to/api/articles?state=fresh&per_page=${postsPerPage}&page=${page}`);
                    const posts = await response.json();
                    
                    if (posts.length === 0) {
                        container.innerHTML = '<p>No more posts found.</p>';
                        nextButton.disabled = true;
                        return;
                    }
                    nextButton.disabled = false;

                    container.innerHTML = '';
                    
                    for (const post of posts) {
                        const postElement = document.createElement('div');
                        postElement.className = 'post loading';
                        postElement.innerHTML = `
                            <h3><a href="${post.url}" target="_blank">${post.title}</a></h3>
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
                                <h3><a href="${post.url}" target="_blank">${post.title}</a></h3>
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

            function nextPage() {
                currentPage++;
                loadAndClassifyPosts(currentPage);
            }

            function prevPage() {
                if (currentPage > 1) {
                    currentPage--;
                    loadAndClassifyPosts(currentPage);
                }
            }

            function resetAndLoad() {
                document.getElementById('stats-container').innerHTML = ''; // Скрываем статистику
                switchView('moderation');
                currentPage = 1;
                loadAndClassifyPosts(currentPage);
            }
            
            async function showStats() {
                try {
                    const response = await fetch('/stats');
                    const stats = await response.json();
                    
                    const statsContainer = document.getElementById('stats-container');
                    const accuracy_text = stats.accuracy ? `${(stats.accuracy * 100).toFixed(2)}%` : 'N/A';

                    statsContainer.innerHTML = `
                        <div style="background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0;">
                            <h3>📊 Classification Statistics</h3>
                            <p><strong>Total classified (since last start):</strong> ${stats.total_classified}</p>
                            <p><strong>Spam detected (since last start):</strong> ${stats.spam_detected}</p>
                            <p><strong>Spam rate:</strong> ${(stats.spam_detected / Math.max(stats.total_classified, 1) * 100).toFixed(1)}%</p>
                            <p><strong>Model Accuracy (from training):</strong> ${accuracy_text}</p>
                            <p><strong>Last updated:</strong> ${new Date(stats.last_updated).toLocaleString()}</p>
                        </div>
                    `;
                } catch (error) {
                    console.error('Failed to load stats:', error);
                }
            }

            async function showRedisInfo() {
                try {
                    const response = await fetch('/redis-info');
                    const redisInfo = await response.json();
                    
                    const redisInfoContainer = document.getElementById('redis-info-container');
                    let content = '';
                    if (redisInfo.status === 'disconnected') {
                        content = `
                        <div style="background: #fff3e0; padding: 15px; border-radius: 5px; margin: 10px 0; border: 1px solid #ff9800;">
                            <h3>⚠️ Redis Info</h3>
                            <p><strong>Status:</strong> Disconnected</p>
                            <p>Advanced features like vector search are disabled.</p>
                        </div>
                        `;
                    } else {
                        content = `
                        <div style="background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; border: 1px solid #4caf50;">
                            <h3>ℹ️ Redis Info</h3>
                            <p><strong>Redis Version:</strong> ${redisInfo.redis_version}</p>
                            <p><strong>Vectors in DB:</strong> ${redisInfo.num_vectors}</p>
                        </div>
                        `;
                    }
                    redisInfoContainer.innerHTML = content;
                } catch (error) {
                    console.error('Failed to load Redis info:', error);
                    const redisInfoContainer = document.getElementById('redis-info-container');
                    redisInfoContainer.innerHTML = `<div style="color: red;">Error loading Redis info.</div>`;
                }
            }

            async function startTraining() {
                document.getElementById('stats-container').innerHTML = ''; // Скрываем статистику
                switchView('training');
                const trainButton = document.getElementById('train-button');
                
                trainButton.disabled = true;
                trainButton.textContent = '👨‍🏫 Training in progress...';
                await refreshLogs();

                try {
                    const response = await fetch('/train', { method: 'POST' });
                    const result = await response.json();
                    alert(result.message);
                } catch (error) {
                    alert(`Training failed: ${error.message}`);
                } finally {
                    trainButton.disabled = false;
                    trainButton.textContent = '🚀 Train Model';
                    // Обновляем статистику и логи после завершения
                    showStats();
                    showRedisInfo();
                    refreshLogs();
                }
            }

            async function refreshLogs() {
                const logContainer = document.getElementById('log-container');
                try {
                    const response = await fetch('/get-logs');
                    const logs = await response.text();
                    logContainer.textContent = logs || 'No logs yet. Training may be in progress.';
                    logContainer.scrollTop = logContainer.scrollHeight;
                } catch (error) {
                    console.error("Failed to fetch logs:", error);
                    logContainer.textContent = "Error loading logs.";
                }
            }

            function switchView(viewName) {
                const logWrapper = document.getElementById('log-container-wrapper');
                const postsContainer = document.getElementById('posts-container');
                const paginationControls = document.getElementById('pagination-controls');

                if (viewName === 'training') {
                    postsContainer.style.display = 'none';
                    paginationControls.style.display = 'none';
                    logWrapper.style.display = 'block';
                } else {
                    postsContainer.style.display = 'block';
                    paginationControls.style.display = 'flex';
                    logWrapper.style.display = 'none';
                }
            }
            
            // Автоматически загружаем посты при загрузке страницы
            document.addEventListener('DOMContentLoaded', () => {
                switchView('moderation'); // Устанавливаем вид по умолчанию
                loadAndClassifyPosts(currentPage);
                showRedisInfo();
                // Set default values for manual check
                document.getElementById('manual-title').value = "How to earn $1000 in a week!";
                document.getElementById('manual-description').value = "Click this link now for a limited time offer. You won't regret it. Best crypto strategy.";
                document.getElementById('manual-tags').value = "money, crypto, investment, quick";
                document.getElementById('manual-reactions').value = "1";
                document.getElementById('manual-comments').value = "0";
                document.getElementById('manual-reading-time').value = "1";
                document.getElementById('manual-followers').value = "5";
            });
        </script>
    </body>
    </html>
    """

@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    """Запуск процесса обучения модели в фоновом режиме"""
    logger.info("Received request to start model training.")
    
    # Запускаем обучение в фоновой задаче, передавая существующий классификатор
    background_tasks.add_task(run_training, classifier=redis_classifier)
    
    return {"message": "Model training started in the background. Check logs for progress."}

@app.get("/get-logs")
async def get_logs():
    """Чтение и возврат файла с логами"""
    try:
        with open("training.log", "r") as f:
            return HTMLResponse(content=f.read(), media_type="text/plain")
    except FileNotFoundError:
        return HTMLResponse(content="Log file not found. Training has not been run yet.", media_type="text/plain")

@app.post("/classify", response_model=ClassificationResult)
async def classify_post(post: DevToPost):
    """Классификация одного поста"""
    import time
    start_time = time.time()
    
    try:
        prediction, confidence, reasoning, similar_post_ids = await classifier.predict(post)
        
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
            processing_time_ms=processing_time,
            similar_post_ids=similar_post_ids
        )
        
        # Асинхронно обновляем статистику в Redis
        if redis_classifier.redis_client:
            async def update_stats():
                try:
                    p = redis_classifier.redis_client.pipeline()
                    p.incr("stats:total_classified")
                    if prediction == 1:
                        p.incr("stats:spam_detected")
                    await p.execute()
                    logger.info("Stats updated in Redis.")
                except Exception as e:
                    logger.error(f"Failed to update stats in Redis: {e}")
            
            background_tasks = BackgroundTasks()
            background_tasks.add_task(update_stats)
            # Это нужно для FastAPI, чтобы задача выполнилась в фоне
            # В данном контексте вызов будет выглядеть так:
            # response = JSONResponse(result.dict())
            # response.background = background_tasks
            # Но для простоты мы просто вызовем ее напрямую
            await update_stats() # Упрощенный вызов для данного примера

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
    if not redis_classifier.redis_client:
        raise HTTPException(status_code=503, detail="Redis is not available. Cannot record feedback.")
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
    total_classified = 0
    spam_detected = 0
    accuracy = None

    # Пытаемся получить статистику из Redis
    if redis_classifier.redis_client:
        try:
            total_classified_bytes, spam_detected_bytes = await redis_classifier.redis_client.mget(
                "stats:total_classified", "stats:spam_detected"
            )
            total_classified = int(total_classified_bytes) if total_classified_bytes else 0
            spam_detected = int(spam_detected_bytes) if spam_detected_bytes else 0
        except Exception as e:
            logger.error(f"Failed to get stats from Redis: {e}")
            # Не прерываем выполнение, просто вернем нули

    # Пытаемся прочитать точность из файла результатов обучения
    try:
        with open("training_results.json", "r") as f:
            results = json.load(f)
            accuracy = results.get("metrics", {}).get("accuracy")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Could not read or parse training_results.json: {e}")

    return StatsResponse(
        total_classified=total_classified,
        spam_detected=spam_detected,
        accuracy=accuracy,
        last_updated=datetime.now()
    )

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    if not redis_classifier.redis_client:
        return {
            "status": "healthy", # Приложение работает, но без Redis
            "redis": "disconnected",
            "timestamp": datetime.now().isoformat()
        }
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
            "redis": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/redis-info")
async def get_redis_info():
    """Получение информации о Redis"""
    if not redis_classifier.redis_client:
        return {
            "redis_version": "N/A",
            "num_vectors": "N/A",
            "status": "disconnected"
        }
    try:
        info = await redis_classifier.redis_client.info()
        num_vectors = 0
        try:
            index_info = await redis_classifier.redis_client.execute_command("FT.INFO", redis_classifier.index_name)
            # Преобразуем список в словарь для удобного доступа
            index_info_dict = {index_info[i]: index_info[i+1] for i in range(0, len(index_info), 2)}
            num_vectors = index_info_dict.get(b'num_docs', 0)
        except aioredis.exceptions.ResponseError as e:
            if "Unknown Index name" not in str(e):
                raise # Пробрасываем неожиданные ошибки
            # Если индекс не найден, количество векторов равно 0

        return {
            "redis_version": info.get("redis_version"),
            "num_vectors": num_vectors,
            "status": "connected"
        }
    except Exception as e:
        logger.error(f"Failed to get Redis info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get Redis info")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="127.0.0.1",
        port=8000, 
        reload=False,
        log_level="info"
    )
