# 🛡️ Redis8 Spam Guard

An intelligent spam classification system for dev.to posts using Redis 8 Vector Sets and FastAPI.

## 🎯 Features

- **Real-time classification** - instant analysis of posts.
- **Redis 8 Vector Sets** - leveraging the latest vector search technology.
- **FastAPI** - a modern asynchronous API.
- **Machine Learning** - classification based on k-NN with vector embeddings.
- **Interactive Web UI** - a dashboard for moderators with real-time post classification, manual checking, statistics, and training logs.
- **Dynamic Data Enrichment** - automatic loading of additional data, such as the author's follower count, for more accurate classification.
- **Feedback Loop** - a system for improving the model based on moderator feedback.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Dev.to API    │ -> │  FastAPI Server  │ -> │  Redis 8        │
│ (Posts, Users)  │    │ (API, Web UI)    │    │  (Vector Sets)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Components:

1. **Data Collector** - collects posts and user data from the dev.to API.
2. **Text Preprocessor** - cleans and prepares text.
3. **Vector Embedder** - converts text and numerical features into vectors using Sentence Transformers.
4. **Redis Vector Store** - stores and searches for vectors in Redis 8.
5. **k-NN Classifier** - classifies based on the nearest neighbors.
6. **FastAPI Server** - provides a REST API for classification and serves the Web UI.
7. **Web Interface** - an interactive control panel for moderators.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/premananda108/redis8-spam-guard.git
cd redis8-spam-guard
```

2. **Set up**
```bash
# Create a virtual environment
python -m venv venv
# source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run Redis 8 in Docker
docker run -d --name redis8-spam-guard -p 6379:6379 redis:8.0.3-bookworm

# Run the application
uvicorn main:app --reload
```

### Initial Model Training

Training can be started directly from the web interface (the "Train Model" button) or with a command in the terminal:
```bash
# Collect data and train the model
python train_model.py
```

## 📖 Usage

### Web Interface

Open **http://localhost:8000** in your browser to access the "Moderator Assistant".

Interface features:
- **Post Feed**: Loads the latest posts from dev.to and classifies them immediately.
- **Manual Check**: A form to check any post based on its parameters (title, description, tags, etc.).
- **Statistics**: Displays classifier performance statistics and Redis connection information.
- **Model Training**: Allows you to start the training process and monitor its progress through real-time logs.

### REST API

#### Classification
- `POST /classify`: Classify a single post.
- `POST /classify-batch`: Batch classification of multiple posts.

**Example request:**
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
    "user": {"id": 456}
  }'
```

**Example response:**
```json
{
  "post_id": 123,
  "is_spam": false,
  "confidence": 0.95,
  "recommendation": "approve",
  "reasoning": ["Similar to legitimate posts (via Redis)"],
  "processing_time_ms": 52.1
}
```

#### Feedback
- `POST /feedback`: Send moderator feedback for future model retraining.

#### Monitoring and Management
- `GET /`: Main page with the web interface.
- `GET /stats`: Get classification statistics.
- `GET /health`: Check the service status and Redis connection.
- `GET /redis-info`: Information about the Redis version and the number of vectors in the database.
- `POST /train`: Start the model training process in the background.
- `GET /get-logs`: Get the logs of the training process.

## 🧠 How Classification Works

### 1. Feature Extraction and Enrichment
The system extracts basic features from the post and dynamically enriches them by requesting additional data (e.g., the author's follower count) via the dev.to API.

### 2. Vectorization
- **Text features** (title, description) are converted into a vector using `Sentence Transformers` (384 dimensions).
- **Numerical features** (reading time, followers, tags) are normalized and added to the vector.
- **Final vector**: ~387 dimensions.

### 3. Similar Post Search (k-NN)
- `Redis Vector Search` is used for ultra-fast search of `k` nearest neighbors (posts with similar vectors) in the database of trained examples.

### 4. Classification
- **Primary method**: Majority vote among the `k` found neighbors. If the majority are spam, the post is classified as spam.
- **Fallback method (heuristics)**: If Redis is unavailable or no similar posts are found, the system uses a set of rules (presence of spam words, low engagement, suspicious tags, etc.) to make a verdict.
- A `confidence score` is calculated.

## 📊 Monitoring and Statistics

Instead of external systems like Prometheus/Grafana, the application has simple and effective built-in monitoring tools:
- **Web interface**: Displays key metrics in real-time.
- **`/stats` endpoint**: Returns JSON with general statistics (number of processed posts, spam found, accuracy of the last trained model).
- **`/health` endpoint**: Allows checking the service's health and the status of the Redis connection.

## 🤝 Contributing

### Project Structure
```
redis8-spam-guard/
├── main.py                 # FastAPI application and Web UI
├── core.py                 # Core classification logic
├── train_model.py          # Script for data collection and training
├── test_api.py             # API tests
├── requirements.txt        # Python dependencies
├── spam_dataset.json       # Sample dataset for training
├── docker-compose.yml      # Docker configuration
├── Dockerfile              # Application Docker image
└── redis.conf              # Redis configuration
```

### Code Style
The project uses **Black** for formatting and **flake8** for linting.

## 🐛 Known Issues

1. **Model cold start** - the first load of Sentence Transformers can take time.
2. **Memory for vectors** - large datasets require a lot of RAM in Redis.
3. **dev.to rate limits** - possible limitations when collecting a large amount of data.

## 🗺️ Roadmap

### v2.0
- [ ] Support for other sources (Reddit, HackerNews)
- [ ] Transformers instead of k-NN
- [ ] Prediction explainability (LIME/SHAP)
- [ ] A/B testing of models

### v2.1  
- [ ] Support for images in posts
- [ ] Plagiarism detection
- [ ] Integration with Slack/Discord for notifications
- [ ] Multilingual support

## 📄 License

MIT License - see the [LICENSE](LICENSE) file.