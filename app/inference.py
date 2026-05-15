import numpy as np
import tensorflow as tf
import joblib
import os

# ── Path ke file model ────────────────────────────────────────────────
# os.path.dirname(__file__) artinya: folder tempat file ini berada
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'cashflow_lstm.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# ── Load model & scaler (hanya sekali saat pertama dipanggil) ─────────
# Teknik ini namanya "lazy loading" — tidak load ulang tiap ada request
_model  = None
_scaler = None

def _load_model():
    """Muat model dan scaler dari disk, simpan di memori."""
    global _model, _scaler
    if _model is None:
        print("Loading model...")
        _model  = tf.keras.models.load_model(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        print("Model siap!")

# ── Fungsi utama: prediksi cash flow ─────────────────────────────────
def prediksi_cashflow(data_30_hari: list) -> dict:
    """
    Prediksi net cash flow untuk hari berikutnya.
    
    Parameter:
        data_30_hari: list berisi 30 nilai net cash flow harian (Rupiah)
                      contoh: [50000, -20000, 75000, ...]
    
    Return:
        dict berisi hasil prediksi dan status
    """
    # Validasi input
    if len(data_30_hari) != 30:
        return {
            "error": f"Data harus berisi tepat 30 nilai, diterima: {len(data_30_hari)}"
        }

    # Load model kalau belum
    _load_model()

    # Langkah 1: Scale input ke range 0-1 (sama seperti saat training)
    arr    = np.array(data_30_hari).reshape(-1, 1)
    scaled = _scaler.transform(arr)

    # Langkah 2: Reshape ke format yang dimengerti LSTM: (1, 30, 1)
    X = scaled.reshape(1, 30, 1)

    # Langkah 3: Prediksi
    pred_scaled = _model.predict(X, verbose=0)

    # Langkah 4: Kembalikan ke nilai Rupiah (inverse transform)
    pred_rupiah = float(_scaler.inverse_transform(pred_scaled)[0][0])

    # Langkah 5: Susun response
    return {
        "prediksi_cashflow_besok": round(pred_rupiah, 2),
        "status": "positif" if pred_rupiah >= 0 else "negatif",
        "peringatan": "Kas diprediksi defisit! Pertimbangkan kurangi pengeluaran." 
                      if pred_rupiah < 0 else None,
        "satuan": "Rupiah"
    }


# ── Test sederhana (jalankan langsung untuk cek) ──────────────────────
if __name__ == '__main__':
    # Simulasi: 30 hari dengan nilai acak sekitar 50ribu
    dummy_data = [50000.0] * 30
    hasil = prediksi_cashflow(dummy_data)
    print("Hasil prediksi:", hasil)