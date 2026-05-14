# app.py — Network Intrusion Detection System Backend

from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import traceback

app = Flask(__name__)

# ── Load all saved model files ────────────────────────────────────
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ model.pkl loaded")

    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    print("✅ encoders.pkl loaded")

    with open('columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    print("✅ columns.pkl loaded")

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ scaler.pkl loaded (value: {scaler})")

except Exception as e:
    print(f"❌ ERROR loading model files: {e}")
    model           = None
    encoders        = None
    feature_columns = None
    scaler          = None

# ── Column names ──────────────────────────────────────────────────
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

# ── Home page ─────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

# ── Status check ──────────────────────────────────────────────────
@app.route('/status')
def status():
    return jsonify({
        'model'          : model is not None,
        'encoders'       : encoders is not None,
        'columns'        : feature_columns is not None,
        'scaler'         : scaler is not None,
        'files_present'  : os.listdir('.')
    })

# ── Predict route ─────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model files not found on server. Visit /status for details.'})

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # ── STEP 1: Read the uploaded file ────────────────────────
        df = pd.read_csv(file, header=0, names=col_names)

        # ── STEP 2: Drop difficulty column if it exists ───────────
        if 'difficulty' in df.columns:
            df = df.drop('difficulty', axis=1)

        # ── STEP 3: Save and drop label column if it exists ───────
        if 'label' in df.columns:
            true_labels = df['label'].apply(
                lambda x: 'attack' if str(x).strip().lower() != 'normal' else 'normal'
            ).tolist()
            df = df.drop('label', axis=1)
        else:
            true_labels = None

        # ── STEP 4: Encode text columns ───────────────────────────
        for col in df.select_dtypes(include=['object']).columns:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].apply(
                    lambda x: le.transform([str(x)])[0]
                    if str(x) in le.classes_ else 0
                )
            else:
                df[col] = 0

        # ── STEP 5: Align columns with training data ──────────────
        df = df.reindex(columns=feature_columns, fill_value=0)

        # ── STEP 6: Scale only if scaler exists ───────────────────
        # Random Forest does NOT need scaling — scaler will be None
        if scaler is not None:
            df_ready = scaler.transform(df)
        else:
            df_ready = df.values

        # ── STEP 7: Make predictions ──────────────────────────────
        predictions = model.predict(df_ready)

        # ── STEP 8: Build results ─────────────────────────────────
        total  = len(predictions)
        normal = int(sum(predictions == 'normal'))
        attack = int(sum(predictions == 'attack'))

        # Calculate accuracy if we have true labels
        accuracy = None
        if true_labels:
            correct  = sum(p == t for p, t in zip(predictions, true_labels))
            accuracy = round((correct / total) * 100, 2)

        # First 500 rows for the table
        details = []
        for i, pred in enumerate(predictions[:500]):
            row = {
                'index'      : i + 1,
                'prediction' : pred,
                'status'     : '✅ Normal' if pred == 'normal' else '⚠️ Attack'
            }
            if true_labels:
                row['actual']  = true_labels[i]
                row['correct'] = '✓' if pred == true_labels[i] else '✗'
            details.append(row)

        return jsonify({
            'total'    : total,
            'normal'   : normal,
            'attack'   : attack,
            'accuracy' : accuracy,
            'details'  : details
        })

    except Exception as e:
        return jsonify({
            'error' : str(e),
            'trace' : traceback.format_exc()
        })


# ── Run the app ───────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
