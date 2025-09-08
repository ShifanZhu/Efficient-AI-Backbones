import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
data_dir = "cars"
train_csv = os.path.join(data_dir, "train.csv")
test_csv = os.path.join(data_dir, "test.csv")

train_out_dir = os.path.join(data_dir, "train")
val_out_dir = os.path.join(data_dir, "val")
test_out_dir = os.path.join(data_dir, "test")

# Make sure output directories exist
os.makedirs(train_out_dir, exist_ok=True)
os.makedirs(val_out_dir, exist_ok=True)
os.makedirs(test_out_dir, exist_ok=True)

def copy_images(df, out_dir, image_col="image"):
    """Copy images listed in a DataFrame to out_dir."""
    for img_name in df[image_col].unique():
        src_path = os.path.join(data_dir, img_name)
        dst_path = os.path.join(out_dir, img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"⚠️ Image not found: {src_path}")

# --- Split train into train/ and val/ ---
df_train = pd.read_csv(train_csv, sep=None, engine="python")
print(f"Train CSV shape: {df_train.shape}")

# Split 90% train, 10% val
df_train_split, df_val_split = train_test_split(
    df_train, test_size=0.1, random_state=42, shuffle=True
)

# Copy images
copy_images(df_train_split, train_out_dir)
copy_images(df_val_split, val_out_dir)

# --- Copy test images ---
df_test = pd.read_csv(test_csv, sep=None, engine="python")
copy_images(df_test, test_out_dir)

print(f"✅ Done. Train: {len(df_train_split)}, Val: {len(df_val_split)}, Test: {len(df_test)}")
