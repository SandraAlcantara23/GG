from flask import Flask, request, send_file, render_template_string, jsonify
from PIL import Image, ImageFilter, ImageOps
import io
import numpy as np
import cv2
import os
import shutil

app = Flask(__name__)

# Imagen fija
FIXED_IMAGE = os.path.join(os.path.dirname(__file__), "Mi.jpg")



def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.uint8(arr))

def send_pil_as_response(img: Image.Image, fmt="JPEG"):
    bio = io.BytesIO()
    img.save(bio, format=fmt, quality=95)
    bio.seek(0)
    return send_file(bio, mimetype="image/jpeg")


def filter_grayscale(pil_img):
    return ImageOps.grayscale(pil_img).convert("RGB")

def filter_invert(pil_img):
    return ImageOps.invert(pil_img.convert("RGB"))

def filter_sepia(pil_img):
    arr = pil_to_numpy(pil_img).astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    tr = 0.393*r + 0.769*g + 0.189*b
    tg = 0.349*r + 0.686*g + 0.168*b
    tb = 0.272*r + 0.534*g + 0.131*b
    out = np.stack([tr, tg, tb], axis=-1)
    out = np.clip(out, 0, 255)
    return numpy_to_pil(out)

def filter_sobel(pil_img):
    npimg = pil_to_numpy(pil_img)
    gray = cv2.cvtColor(npimg, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)
    return numpy_to_pil(out)

def filter_sharpen(pil_img):
    return pil_img.filter(ImageFilter.Kernel(
        (3, 3),
        [0, -1, 0, -1, 5, -1, 0, -1, 0],
        scale=None,
        offset=0
    ))

def filter_blur(pil_img):
    return pil_img.filter(ImageFilter.BoxBlur(2))

# Diccionario de filtros
FILTERS = {
    "blancoYNegro": filter_grayscale,
    "invertir": filter_invert,
    "sepia": filter_sepia,
    "sobel": filter_sobel,
    "sharpen": filter_sharpen,
    "blur": filter_blur,
}


INDEX_HTML = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Filtros con imagen fija</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background-color: #f4f4f9;
      color: #333;
      display: flex;
      flex-direction: column;
      align-items: center;
      margin: 0;
      padding: 20px;
    }

    h1, h3 {
      text-align: center;
      color: #222;
    }

    form {
      margin: 20px 0;
      display: flex;
      gap: 10px;
      align-items: center;
    }

    select, button {
      padding: 8px 12px;
      font-size: 16px;
      border-radius: 8px;
      border: 1px solid #ccc;
      outline: none;
      cursor: pointer;
      transition: 0.3s;
    }

    button {
      background-color: #4CAF50;
      color: white;
      border: none;
    }

    button:hover {
      background-color: #45a049;
    }

    .image-container {
      display: flex;
      gap: 20px;
      flex-wrap: wrap;
      justify-content: center;
      margin-top: 20px;
    }

    .image-box {
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .image-box img {
      max-width: 300px;
      border-radius: 12px;
      border: 2px solid #ccc;
      box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
    }

    .caption {
      margin-top: 8px;
      font-weight: bold;
      color: #555;
    }
  </style>
</head>
<body>
  <h1>Aplicador de Filtros</h1>
  <h3>Selecciona un filtro para la imagen fija</h3>

  <form id="form">
    <select id="filtro" name="filter">
      <option value="blancoYNegro">Blanco y Negro</option>
      <option value="invertir">Invertir colores</option>
      <option value="sepia">Sepia</option>
      <option value="sobel">Sobel (bordes)</option>
      <option value="sharpen">Mejorar (Sharpen)</option>
      <option value="blur">Desenfoque</option>
    </select>
    <button type="submit">Aplicar</button>
  </form>

  <div class="image-container">
    <div class="image-box">
      <img src="/static/Mi.jpg" alt="Original" id="original">
      <div class="caption">Original</div>
    </div>
    <div class="image-box">
      <img src="" alt="Resultado" id="out">
      <div class="caption">Resultado</div>
    </div>
  </div>

  <script>
    document.getElementById("form").addEventListener("submit", async function(e) {
      e.preventDefault();
      const filtro = document.getElementById("filtro").value;

      const res = await fetch("/api/filter?filter=" + filtro, { method: "GET" });
      if(!res.ok){
          const txt = await res.text();
          alert("Error: " + txt);
          return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      document.getElementById("out").src = url;
    });
  </script>
</body>
</html>

"""

# -------------------------
# Rutas
# -------------------------

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/api/filter", methods=["GET"])
def api_filter():
    filter_name = request.args.get("filter", "blancoYNegro")
    try:
        pil_img = Image.open(FIXED_IMAGE).convert("RGB")
    except Exception as e:
        return f"Error al abrir la imagen fija: {e}", 400

    func = FILTERS.get(filter_name)
    if not func:
        return jsonify({
            "error": "Filtro no encontrado",
            "available": list(FILTERS.keys())
        }), 400

    try:
        out_img = func(pil_img)
    except Exception as e:
        return f"Error aplicando filtro: {e}", 500

    return send_pil_as_response(out_img, fmt="JPEG")


if __name__ == "__main__":
    # Asegúrate de que Mi.jpg esté en /static
    os.makedirs("static", exist_ok=True)
    if os.path.exists("Mi.jpg") and not os.path.exists("static/Mi.jpg"):
        shutil.copy("Mi.jpg", "static/Mi.jpg")

    app.run(debug=True, host="0.0.0.0", port=5000)