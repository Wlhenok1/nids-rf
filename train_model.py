# train_model.py — Random Forest (low memory, high accuracy)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ── STEP 1: Load the dataset ──────────────────────────────────────
print("Loading data...")

col_names = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"
]

df = pd.read_csv('KDDTrain+.csv', header=0, names=col_names)
df = df.drop('difficulty', axis=1)

print(f"Dataset loaded: {len(df)} rows")

# ── STEP 2: Fix the labels ────────────────────────────────────────
attack_types = [
    'neptune', 'satan', 'ipsweep', 'portsweep', 'smurf', 'nmap', 'back',
    'teardrop', 'warezclient', 'pod', 'guess_passwd', 'buffer_overflow',
    'warezmaster', 'land', 'imap', 'rootkit', 'loadmodule', 'ftp_write',
    'multihop', 'phf', 'perl', 'spy'
]

df['label'] = df['label'].apply(
    lambda x: 'attack' if str(x).strip().lower() in attack_types else 'normal'
)

print("\nLabel distribution:")
print(df['label'].value_counts())

# ── STEP 3: Convert text columns to numbers ───────────────────────
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    if column != 'label':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# ── STEP 4: Split into features and labels ────────────────────────
X = df.drop('label', axis=1)
y = df['label']

print(f"\nFinal check before training:")
print(f"  Normal : {sum(y == 'normal')}")
print(f"  Attack : {sum(y == 'attack')}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ── STEP 5: Train Random Forest ───────────────────────────────────
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=50,   # 50 trees — saves memory vs 100
    max_depth=20,      # limit depth to save RAM
    random_state=42,
    n_jobs=1           # single core — safer for free hosting
)
model.fit(X_train, y_train)

# ── STEP 6: Test accuracy ─────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"\n✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# ── STEP 7: Save everything ───────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

# Save scaler as None — Random Forest does not need scaling
with open('scaler.pkl', 'wb') as f:
    pickle.dump(None, f)

print("\n✅ All files saved:")
print("   model.pkl")
print("   encoders.pkl")
print("   columns.pkl")
print("   scaler.pkl")
print("\nNow run: python app.py")
