# Import Libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("dataset.csv")

# Prepare Features and Target
X = df.drop(['user_id', 'song_id', 'timestamp', 'target'], axis=1)
y = df['target']

# Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make Predictions and Evaluate
y_pred = model.predict(X_test)
print("🎯 Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))


# Song Recommendation Function
def recommend_top_songs_for_user(user_id, df, model):
    user_data = df[df['user_id'] == user_id]

    if user_data.empty:
        print(f"No data found for user: {user_id}")
        return

    # Prepare user features
    X_user = user_data.drop(['user_id', 'song_id', 'timestamp', 'target'], axis=1)
    song_ids = user_data['song_id'].values

    # Predict probability of replaying each song
    probabilities = model.predict_proba(X_user)[:, 1]

    # Combine song IDs with their predicted probabilities
    recommendations = list(zip(song_ids, probabilities))

    # Sort by highest probability
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Display Top 3 Recommendations
    print(f"\n🎧 Top Recommended Songs for {user_id}:")
    for song, prob in recommendations[:3]:
        print(f"  {song} — Replay Probability: {prob:.2f}")


#  Test Recommendation for One User
recommend_top_songs_for_user('u1', df, model)
