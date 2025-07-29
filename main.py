import json
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from datetime import datetime
from redis.exceptions import ResponseError

from train_model import main as run_training
from core import (
    DevToPost, ClassificationResult, ModeratorFeedback,
    BatchClassificationRequest, StatsResponse,
    RedisVectorClassifier, RediSearchClassifier, SimilarPostInfo
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Dev.to Spam Classifier API",
    description="AI-powered spam detection for dev.to posts using Redis 8 Vector Sets",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global objects
redis_classifier = RedisVectorClassifier()
classifier = RediSearchClassifier(redis_classifier)
POST_CACHE = {} # Cache for storing full post data for feedback

@app.on_event("startup")
async def startup_event():
    """Initialization on startup"""
    await redis_classifier.init_redis()
    logger.info("Application started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if redis_classifier.redis_client:
        await redis_classifier.redis_client.close()
    logger.info("Application shut down")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Main page with the interface"""
    with open("index.html", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    """Starts the model training process in the background"""
    logger.info("Received request to start model training.")
    
    # Start training in a background task, passing the existing classifier
    background_tasks.add_task(run_training, classifier=redis_classifier)
    
    return {"message": "Model training started in the background. Check logs for progress."}

@app.get("/get-logs")
async def get_logs():
    """Reads and returns the log file"""
    try:
        with open("training.log", "r") as f:
            return HTMLResponse(content=f.read(), media_type="text/plain")
    except FileNotFoundError:
        return HTMLResponse(content="Log file not found. Training has not been run yet.", media_type="text/plain")

@app.post("/classify", response_model=ClassificationResult)
async def classify_post(post: DevToPost):
    """Classifies a single post"""
    import time
    start_time = time.time()
    
    try:
        # Store post data in cache for potential feedback
        POST_CACHE[str(post.id)] = post

        prediction, confidence, reasoning, similar_posts_data = await classifier.predict(post)
        
        # Determine recommendation
        if confidence >= 0.8:
            recommendation = "block" if prediction == 1 else "approve"
        else:
            recommendation = "review"
        
        processing_time = (time.time() - start_time) * 1000
        
        # The data from `predict` is already a list of SimilarPostInfo objects
        similar_posts = similar_posts_data

        # Check for moderator feedback
        moderator_verdict = None
        if redis_classifier.redis_client:
            feedback_key = f"feedback:{post.id}"
            feedback_data = await redis_classifier.redis_client.get(feedback_key)
            if feedback_data:
                feedback_json = json.loads(feedback_data)
                moderator_verdict = "spam" if feedback_json.get("is_spam") else "legit"

        result = ClassificationResult(
            post_id=post.id,
            is_spam=bool(prediction),
            confidence=confidence,
            recommendation=recommendation,
            reasoning=reasoning,
            processing_time_ms=processing_time,
            similar_posts=similar_posts,
            moderator_verdict=moderator_verdict
        )
        
        # Asynchronously update stats in Redis
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
            
            await update_stats()

        logger.info(f"Classified post {post.id}: {'SPAM' if prediction else 'OK'} ({confidence:.2f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Classification failed for post {post.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify-batch")
async def classify_batch(request: BatchClassificationRequest):
    """Batch classification of posts"""
    results = []
    
    for post in request.posts:
        try:
            prediction, confidence, reasoning, _ = await classifier.predict(post) # similar_posts are not needed here
            
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
    """Moderator feedback to improve the model"""
    if not redis_classifier.redis_client:
        raise HTTPException(status_code=503, detail="Redis is not available. Cannot record feedback.")
    
    post_id_str = str(feedback.post_id)
    post_data = POST_CACHE.get(post_id_str)

    if not post_data:
        raise HTTPException(status_code=404, detail=f"Post data for ID {post_id_str} not found in cache. Cannot record feedback for training.")

    try:
        # Save feedback verdict and full post data for retraining
        feedback_key = f"feedback:{post_id_str}"
        post_data_key = f"post_data:{post_id_str}"

        feedback_data = {
            "post_id": feedback.post_id,
            "is_spam": feedback.is_spam,
            "moderator_id": feedback.moderator_id,
            "notes": feedback.notes,
            "timestamp": datetime.now().isoformat()
        }
        
        # Use a pipeline for atomic operations
        p = redis_classifier.redis_client.pipeline()
        p.set(feedback_key, json.dumps(feedback_data))
        p.set(post_data_key, post_data.model_dump_json()) # Use model_dump_json for Pydantic v2
        await p.execute()
        
        logger.info(f"Received and stored full feedback for post {feedback.post_id} from {feedback.moderator_id}")
        
        return {"status": "success", "message": "Feedback recorded for retraining"}
    
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Classification statistics"""
    total_classified = 0
    spam_detected = 0
    accuracy = None

    # Try to get stats from Redis
    if redis_classifier.redis_client:
        try:
            total_classified_bytes, spam_detected_bytes = await redis_classifier.redis_client.mget(
                "stats:total_classified", "stats:spam_detected"
            )
            total_classified = int(total_classified_bytes) if total_classified_bytes else 0
            spam_detected = int(spam_detected_bytes) if spam_detected_bytes else 0
        except Exception as e:
            logger.error(f"Failed to get stats from Redis: {e}")
            # Don't interrupt execution, just return zeros

    # Try to read accuracy from the training results file
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
    """Service health check"""
    if not redis_classifier.redis_client:
        return {
            "status": "healthy", # App is running, but without Redis
            "redis": "disconnected",
            "timestamp": datetime.now().isoformat()
        }
    try:
        # Check connection to Redis
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
    """Get information about Redis"""
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
            # FT.INFO response is a list of key-value pairs
            raw_index_info = await redis_classifier.redis_client.execute_command("FT.INFO", redis_classifier.index_name)
            # Convert the list to a dictionary for easier access
            index_info_dict = {raw_index_info[i]: raw_index_info[i+1] for i in range(0, len(raw_index_info), 2)}
            num_vectors = int(index_info_dict.get('num_docs', 0))
        except ResponseError as e:
            # This is expected if the index doesn't exist yet
            if "Unknown Index name" in str(e) or "no such index" in str(e):
                num_vectors = 0
            else:
                raise  # Re-raise other unexpected errors

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