import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

RAW_IMAGE_DIR = "data/raw/"
CSV_PATH = "data/metadata.csv"

# Load metadata
df = pd.read_csv(CSV_PATH)

# Filter to images that actually exist in data/raw/
available_images = set(os.listdir(RAW_IMAGE_DIR))
df = df[df["Image Index"].isin(available_images)]

# Parse multi-labels
df["Finding Labels"] = df["Finding Labels"].str.split("|")

# Remove rare label combinations (must appear at least twice)
df["label_combo"] = df["Finding Labels"].apply(lambda labels: "|".join(sorted(labels)))
label_combo_counts = df["label_combo"].value_counts()
df = df[df["label_combo"].isin(label_combo_counts[label_combo_counts >= 2].index)]
df = df.drop(columns=["label_combo"])

# Create stratification labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["Finding Labels"])

# Train/test split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Save to CSV
os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print(f"✅ Split complete. Saved {len(train_df)} train and {len(test_df)} test samples.")

