// server.js
const express = require('express');
const redis = require('redis');
const cors = require('cors');

const app = express();
app.use(express.json()); // Allows reading JSON from requests
app.use(cors()); // Allows the frontend to communicate with the backend

// Connect to Redis
const redisClient = redis.createClient();
redisClient.on('error', (err) => console.log('Redis Client Error', err));
(async () => {
    await redisClient.connect();
})();

// --- API Endpoints ---

// 1. Get all posts
app.get('/posts', async (req, res) => {
    try {
        const postKeys = await redisClient.keys('post:*');
        if (postKeys.length === 0) {
            return res.json([]);
        }
        const posts = await redisClient.mGet(postKeys);
        res.json(posts.map(JSON.parse));
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

// 2. Create a new post
app.post('/posts', async (req, res) => {
    try {
        const { title, content, author } = req.body;
        
        // --- TODO: SPAM GUARD INTEGRATION GOES HERE ---
        /*
        // 1. Send the post content to your Spam Guard API
        const spamGuardResponse = await fetch('http://localhost:8000/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                // We create a mock post object for classification
                id: Date.now(),
                title: title,
                description: content,
                tag_list: [], // You can add tag extraction logic if needed
                user: { id: author } // Using author name as a simple user ID
            })
        });

        const classification = await spamGuardResponse.json();

        // 2. Check the recommendation
        if (classification.recommendation === 'block') {
            // If it's spam, reject the post
            return res.status(403).json({ error: 'Post was classified as spam and rejected.' });
        }
        */
        // --- END OF SPAM GUARD INTEGRATION ---


        const postId = `post:${Date.now()}`; // Unique ID
        const postData = JSON.stringify({ id: postId, title, content, author });
        await redisClient.set(postId, postData);
        
        res.status(201).json(JSON.parse(postData));
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

// Start the server
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Forum backend server is running on http://localhost:${PORT}`);
});
