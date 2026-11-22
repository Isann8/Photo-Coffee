from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from pathlib import Path
from collections import Counter
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import base64
import logging

# Menentukan lokasi folder utama proyek
BASE_DIR = Path(__file__).resolve().parent

# Membuat aplikasi Flask
# Flask diarahkan membaca index.html langsung dari folder utama
app = Flask(
    __name__,
    template_folder=str(BASE_DIR)
)

# Mengatur batas ukuran file gambar maksimal 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Menentukan lokasi model YOLOv8 best.pt
MODEL_PATH = Path(
    r"D:\Isan-1\.Photo Coffee\Dataset\Photo Coffeeyolov8\runs\PHOTO_COFFEE_YOLOv8_SEG\weights\best.pt"
)

# Menentukan lokasi file HTML
HTML_PATH = BASE_DIR / "index.html"

# Membuat folder penyimpanan sementara
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mengatur parameter prediksi YOLOv8
CONFIDENCE = 0.40
IMAGE_SIZE = 640

# Menentukan format gambar yang diperbolehkan
ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}

# Mengatur logging aplikasi
logging.basicConfig(level=logging.INFO)


# Memeriksa keberadaan file index.html
if not HTML_PATH.exists():
    raise FileNotFoundError(
        f"File index.html tidak ditemukan: {HTML_PATH}"
    )


# Memeriksa keberadaan model best.pt
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"File best.pt tidak ditemukan: {MODEL_PATH}"
    )


# Memuat model YOLOv8 Segmentation
print("\nMemuat model PHOTO COFFEE...")

model = YOLO(str(MODEL_PATH))


# Mengambil nama kelas langsung dari model
MODEL_NAMES = model.names


# Menyusun nama kelas berdasarkan ID model
if isinstance(MODEL_NAMES, dict):

    CLASS_NAMES = {
        int(class_id): class_name
        for class_id, class_name in MODEL_NAMES.items()
    }

else:

    CLASS_NAMES = {
        class_id: class_name
        for class_id, class_name in enumerate(MODEL_NAMES)
    }


# Menampilkan informasi model
print("\nModel berhasil dimuat!")
print("Nama kelas:", CLASS_NAMES)
print("Lokasi model:", MODEL_PATH)
print("Lokasi HTML:", HTML_PATH)
print("PHOTO COFFEE siap digunakan.\n")


# Memeriksa format file gambar
def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# Menampilkan halaman utama PHOTO COFFEE
@app.route("/", methods=["GET"])
def home():

    return render_template("index.html")


# Memeriksa status backend dan model
@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "application": "PHOTO COFFEE",
        "backend": "Flask",
        "model": "YOLOv8 Segmentation",
        "classes": CLASS_NAMES
    })


# Endpoint untuk melakukan prediksi gambar
@app.route("/predict", methods=["POST"])
def predict():

    # Memeriksa apakah gambar sudah dikirim
    if "image" not in request.files:

        return jsonify({
            "success": False,
            "message": "Silakan unggah gambar terlebih dahulu."
        }), 400

    file = request.files["image"]

    # Memeriksa apakah file sudah dipilih
    if file.filename == "":

        return jsonify({
            "success": False,
            "message": "File gambar belum dipilih."
        }), 400

    # Memeriksa format gambar
    if not allowed_file(file.filename):

        return jsonify({
            "success": False,
            "message": "Gunakan format JPG, JPEG, PNG, atau WEBP."
        }), 400

    try:

        # Mengamankan nama file
        filename = secure_filename(file.filename)

        # Membaca data gambar
        image_bytes = file.read()

        # Memastikan file tidak kosong
        if not image_bytes:

            return jsonify({
                "success": False,
                "message": "File gambar kosong."
            }), 400

        # Mengubah data gambar menjadi array
        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        # Membaca gambar menggunakan OpenCV
        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        # Memeriksa apakah gambar berhasil dibaca
        if image is None:

            return jsonify({
                "success": False,
                "message": "Gambar tidak dapat dibaca."
            }), 400

        # Melakukan prediksi menggunakan YOLOv8 Segmentation
        results = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            verbose=False
        )

        # Mengambil hasil prediksi pertama
        result = results[0]

        # Menyiapkan penghitung jumlah objek
        counts = Counter()

        # Menyiapkan daftar hasil deteksi
        detections = []

        # Memeriksa hasil deteksi objek
        if result.boxes is not None:

            for box in result.boxes:

                # Mengambil ID kelas
                class_id = int(box.cls.item())

                # Mengambil nama kelas
                class_name = CLASS_NAMES.get(
                    class_id,
                    f"Class {class_id}"
                )

                # Mengambil nilai confidence
                confidence = float(box.conf.item())

                # Menghitung jumlah objek setiap kelas
                counts[class_name] += 1

                # Menyimpan informasi hasil deteksi
                detection = {
                    "class": class_name,
                    "confidence": round(
                        confidence * 100,
                        2
                    )
                }

                detections.append(detection)

        # Menghitung jumlah objek berdasarkan kelas
        class_counts = {
            class_name: counts.get(class_name, 0)
            for class_name in CLASS_NAMES.values()
        }

        # Membuat visualisasi bounding box dan segmentation mask
        annotated_image = result.plot()

        # Mengubah hasil visualisasi menjadi format JPG
        success, encoded_image = cv2.imencode(
            ".jpg",
            annotated_image
        )

        # Memeriksa keberhasilan encoding gambar
        if not success:

            raise RuntimeError(
                "Gagal membuat gambar hasil prediksi."
            )

        # Mengubah gambar menjadi Base64
        image_base64 = base64.b64encode(
            encoded_image.tobytes()
        ).decode("utf-8")

        # Menyiapkan hasil prediksi
        response = {
            "success": True,
            "message": "Analisis gambar berhasil.",
            "filename": filename,
            "total_objects": len(detections),
            "class_counts": class_counts,
            "detections": detections,
            "annotated_image": (
                "data:image/jpeg;base64,"
                + image_base64
            )
        }

        # Mengirim hasil prediksi ke frontend
        return jsonify(response)

    except Exception as error:

        # Menampilkan detail kesalahan pada terminal
        app.logger.exception(
            "Terjadi kesalahan saat melakukan prediksi."
        )

        return jsonify({
            "success": False,
            "message": str(error)
        }), 500


# Menangani ukuran file yang melebihi batas
@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "success": False,
        "message": "Ukuran gambar maksimal 10 MB."
    }), 413


# Menangani halaman yang tidak ditemukan
@app.errorhandler(404)
def page_not_found(error):

    return jsonify({
        "success": False,
        "message": "Halaman tidak ditemukan."
    }), 404


# Menangani kesalahan server
@app.errorhandler(500)
def internal_server_error(error):

    return jsonify({
        "success": False,
        "message": "Terjadi kesalahan pada server."
    }), 500


# Menjalankan aplikasi Flask
if __name__ == "__main__":

    print("================================")
    print("       PHOTO COFFEE SYSTEM")
    print("================================")
    print("Backend       : Flask")
    print("Model         : YOLOv8 Segmentation")
    print("Confidence    :", CONFIDENCE)
    print("Image Size    :", IMAGE_SIZE)
    print("Model Path    :", MODEL_PATH)
    print("HTML Path     :", HTML_PATH)
    print("================================")
    print("Website       : http://127.0.0.1:5000")
    print("================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
