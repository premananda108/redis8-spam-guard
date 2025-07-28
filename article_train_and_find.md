# How the Spam Classification System Works

This document describes the architecture and logic of the spam detection system, built using the `all-MiniLM-L6-v2` model and Redis.

## Key Components

1.  **Vectorization Model:** `all-MiniLM-L6-v2`
2.  **Vector Database:** Redis (with the RediSearch module)
3.  **Training Data Source:** The `dev.to` API
4.  **Labeling Mechanism:** Automatic, based on heuristics.

---

## The Role of the `all-MiniLM-L6-v2` Model

The key point is: **we do not train this model**. We use it as a ready-made, pre-trained tool.

`all-MiniLM-L6-v2` is a compact and fast transformer model that solves one task: converting text (e.g., a post's title and description) into a numerical vector (in our case, an array of 384 numbers). This process is called **vectorization** or **creating embeddings**.

This vector is a mathematical "embodiment" of the text's meaning. Texts with similar meanings will have vectors that are close to each other.

The model was chosen for its ideal balance between:
*   **Quality:** It understands context and language nuances well.
*   **Speed and Size:** It is lightweight enough to work in real-time without specialized hardware (GPU).

---

## The "Training" Process (Creating a Knowledge Base)

Our process, which we call "training," is actually the **creation and population of a knowledge base** in Redis. Here's how it's structured:

1.  **Data Collection:** The `train_model.py` script accesses the public `dev.to` API and downloads hundreds of real posts.

2.  **Automatic Labeling:** For each downloaded post, the `SpamLabelGenerator` is run. This class, based on a set of **heuristics (rules)**, assigns a label to the post: `spam` or `not_spam`.
    *   **Examples of heuristics:** presence of spam keywords ("earn money," "buy now"), low number of reactions, suspicious tags, low follower count for the author, etc.

3.  **Vectorization:** A combined vector is created for each post, consisting of two parts:
    *   **Text Vector:** The post's title and description are passed to the `all-MiniLM-L6-v2` model, which converts them into a numerical vector (an embedding of 384 numbers) that reflects the meaning of the text.
    *   **Numerical Features:** Normalized numerical values are added to the text vector: reading time, author's follower count, and the number of tags. This allows finding posts that are similar not only in meaning but also in structure.

4.  **Saving to Redis:** A set of data is saved in Redis for each post, including:
    *   The generated **vector**.
    *   The assigned **label** (`spam`/`not_spam`).
    *   The **title** and **URL** of the post for quick display in the interface.

**The bottom line:** We do not load the model itself into Redis. We load **the results of its work** into it—vectors to which we have assigned labels ourselves.

---

## The Role of Redis in Classification

When a new, unknown post is submitted for review, the system performs the following steps:

1.  **Vectorization:** The text of the new post is also passed through the `all-MiniLM-L6-v2` model to obtain its vector.

2.  **k-Nearest Neighbors Search (k-NN Search):** The system queries Redis with a request: "Find the `k` vectors in the database that are closest to the vector of the new post." (In our configuration, `k=9`).

3.  **Voting:** The system looks at the labels (`spam`/`not_spam`) stored with the found neighbors.
    *   If the majority of the neighbors have the `spam` label, the new post is classified as spam.
    *   Otherwise, it is considered legitimate.

Thus, Redis acts as a fast "card catalog" that allows, based on the "semantic fingerprint" (vector) of a new post, to instantly find similar posts from the past and draw a conclusion based on their labels.
