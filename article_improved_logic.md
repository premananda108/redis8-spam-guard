# The Evolution of Logic: How We Improved the Vector Model to Combat Spam

In the first version of our `Redis8 Spam Guard` system, we used a hybrid approach to post vectorization. The vector representing each post consisted of two parts:

1.  **Text Embedding (384 dimensions):** A semantic representation of the title and description.
2.  **Numerical Features (5 dimensions):** Post metadata, including `reading_time`, `reactions_count`, `comments_count`, `user_followers`, and `tags_count`.

The resulting vector had a dimensionality of 389. During the training phase, this model performed well because it was learning from historical data where posts had already accumulated engagement statistics.

## The Problem: A Logical Paradox in Real-Time Operation

During our analysis, we discovered a fundamental problem with this approach that became apparent when classifying **new** posts.

**The Core Issue:** Our system is designed to react to spam **instantly**, at the moment it appears. A new post, by definition, has no reactions or comments yet. Its `reactions_count` and `comments_count` are always zero.

When we created a vector for such a post, the zero values in the engagement components made it mathematically "closer" to the vectors from the training set that also had low engagement. And those, as a rule, were the spam posts.

This created a paradox: the system was overly suspicious of **any** new post because its "engagement profile" looked like spam. We were comparing "apples" (new posts without history) to "oranges" (old posts with history).

## The Solution: Separating Labeling and Vectorization Logic

The key idea for improving the system is to separate the process of deciding on a "spam/not spam" label from the process of creating the post's vector representation.

### 1. Training Phase: Intelligent Labeling

We still use heuristics that analyze **all** available data about a post, including engagement. This allows us to assign the `is_spam = True` or `is_spam = False` label to historical data with high accuracy. This phase is the "teacher's work," using the full scope of information to create a high-quality training dataset.

### 2. Vectorization Phase: A "Clean" Vector

However, in the vector that we now save to Redis, we include **only those features that are known at the moment the post is created**. We have **excluded** `reactions_count` and `comments_count` from it.

**Old Vector (389 dimensions):**
`[Text (384), Reading Time, **Reactions**, **Comments**, Followers, Tag Count]`

**New, Improved Vector (387 dimensions):**
`[Text (384), Reading Time, Followers, Tag Count]`

## Advantages of the New Approach

1.  **Correct Comparison:** Now, the vector of a new post is compared with vectors from the training set using the same, "fair" parameters. The model focuses on the **content** of the post and the **author's reputation**, not on engagement that cannot yet exist.
2.  **Elimination of Bias:** We have removed the system's bias against new posts. Now, a good post from a new author will not be mistakenly classified as spam just because it has 0 likes.
3.  **More Robust Model:** The system is forced to learn to identify spam by its essence (keywords, text structure, suspicious tags), rather than by indirect signs. This makes the model more stable and accurate.

This refactoring of logic is an excellent example of how a critical analysis of a system's operation can lead to a significant improvement in its architecture and reliability without requiring complex code changes.

## The Final Push: Creating a Synthetic Dataset

After implementing the improved logic, we faced a new problem: data imbalance. The data from dev.to contained very little real spam. As a result, the model, even with a correct architecture, could not train effectively—it simply lacked examples of "bad" behavior. The `precision` and `recall` metrics remained at zero.

The solution was obvious: if there is no spam in the real data, we must create it ourselves.

We generated a `spam_dataset.json` file containing 50 diverse examples of blatant spam:

-   Offers of quick earnings and crypto schemes.
-   Phishing links and fake security notifications.
-   Sales of SEO services, follower boosting.
-   Dubious courses, miracle products, and much more.

We then enhanced the training script to combine the "clean" data from dev.to with our "dirty" spam dataset. This allowed us to create a balanced sample on which the model could finally unleash its potential.

## The Result: A Working and Reliable Model

After training on the new, balanced dataset, we obtained the following metrics:

-   **Accuracy:** ~94%
-   **Precision:** **1.0** — a perfect result! This means the model made **zero false positives**, labeling a good post as spam.
-   **Recall:** **~30%** — the model successfully detected and classified one-third of all spam in the test set. This is a huge leap from zero and an excellent starting point for further improvements.

The resulting model is "cautious": it prefers to miss spam rather than block legitimate content. This is a critically important property for any moderation system.

In the end, by going from analyzing and fixing architectural flaws to enriching the data, we have created a truly working and reliable system for combating spam.

## Final Touches: Improving the Interface and User Experience

In the final stage, we focused on refining the web interface to make it as convenient and informative as possible for the end-user (the moderator).

### 1. Visualizing the Training Process

Initially, the training process was launched in a "blind" background mode. To make it transparent, we implemented a simple but effective logging system:

-   The training script (`train_model.py`) now writes all its progress to a text file, `training.log`.
-   A special endpoint (`/get-logs`) and a "Refresh Logs" button were added to the web interface.
-   When training is started, the interface now automatically switches to "training mode," hiding the post feed and showing a special window where progress can be tracked by refreshing the logs with a button.

### 2. Prioritizing New Posts

We realized that for moderation, "fresh" posts are more important than "popular" ones. We changed the request to the dev.to API by adding the `state=fresh` parameter. Now, the moderator panel defaults to showing the most recently published posts, allowing for the most rapid response to potential spam.

### 3. Separating Interface Modes

To avoid confusion and make the interface more focused, we introduced two operating modes:

-   **Moderation Mode:** Shows the post feed and pagination. Activated by default and when clicking "Load Latest Posts".
-   **Training Mode:** Shows the log window. Activated when clicking "Train Model".

When switching between modes, secondary blocks (like the statistics block) are automatically hidden to avoid cluttering the interface.

### Important Note on Data

It is worth noting that our application uses the **public API** of dev.to. It analyzes posts that have already passed initial moderation and have been published. Therefore, the list of posts in our tool **will not match** the list in the official, internal dev.to moderation panel, which works with posts before they are published. Our tool should be considered a second line of defense and a powerful means for analyzing spam trends among already published content.