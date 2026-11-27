"""Imágenes del satélite GOES-19 sobre el NOA (Salta, Jujuy, Tucumán y Catamarca): IR, vapor de agua y visible.

Para cada banda toma los últimos archivos ABI-L2-CMIPF del bucket abierto de NOAA, recorta la región, la reproyecta de la geometría del
satélite a latitud/longitud, la pinta con una paleta y la envía al backend (POST /api/goes/ingreso) como PNG con su georreferencia.
Solo sube los cuadros que el backend todavía no tiene.

Uso:  python -m modelo3d.goes_imagenes            (todas las bandas)
      python -m modelo3d.goes_imagenes ir va      (solo algunas)
Variables: BACKEND_URL, MODELO3D_TOKEN (o GOES_TOKEN), GOES_CUADROS (cuántos cuadros por banda, por defecto 6)."""
import os, re, io, sys, json, base64, datetime as dt, numpy as np, requests

BACKEND = os.environ.get("BACKEND_URL", "https://metricmoisture-backend.onrender.com").rstrip("/")
# Región: cubre Salta, Jujuy, Tucumán y Catamarca con un margen
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = -29.8, -21.6, -69.4, -62.0

# Paletas: (valor, color). IR y vapor de agua en °C de temperatura de brillo.
PALETA_IR = [(-90, "#2b0a3d"), (-80, "#ffffff"), (-70, "#d946ef"), (-65, "#ef4444"), (-60, "#f97316"), (-55, "#facc15"),
             (-50, "#22c55e"), (-40, "#38bdf8"), (-30, "#bfe6ff"), (-20, "#e5e7eb"), (0, "#9ca3af"), (20, "#4b5563"), (40, "#0b0f14")]
PALETA_VA = [(-80, "#ffffff"), (-70, "#7dd3fc"), (-60, "#2563eb"), (-50, "#38bdf8"), (-40, "#bde5f5"), (-30, "#f3ead0"),
             (-20, "#e2b45a"), (-10, "#9a6420"), (0, "#4a2a06")]
BANDAS = {
    "ir":  {"canal": "C13", "paso": 0.02, "unidad": "°C", "paleta": PALETA_IR, "n": 6, "nombre": "Infrarrojo (10,3 µm)"},
    "va":  {"canal": "C08", "paso": 0.02, "unidad": "°C", "paleta": PALETA_VA, "n": 6, "nombre": "Vapor de agua (6,2 µm)"},
    "vis": {"canal": "C02", "paso": 0.01, "unidad": "",   "paleta": None,       "n": 4, "nombre": "Visible (0,64 µm)"},
}

# ---------------------------------------------------------------- navegación GOES-R (lat/lon -> ángulos de barrido)
def _angulos(P, lat, lon):
    """Ángulos de barrido (x, y) en radianes con que el satélite ve cada punto de la superficie. Funciona con arrays."""
    la, lo = np.radians(lat), np.radians(lon); r_eq, r_pol = P["r_eq"], P["r_pol"]
    phic = np.arctan((r_pol ** 2 / r_eq ** 2) * np.tan(la))
    rc = r_pol / np.sqrt(1 - ((r_eq ** 2 - r_pol ** 2) / r_eq ** 2) * np.cos(phic) ** 2)
    sx = P["H"] - rc * np.cos(phic) * np.cos(lo - P["lam0"]); sy = -rc * np.cos(phic) * np.sin(lo - P["lam0"]); sz = rc * np.sin(phic)
    with np.errstate(invalid="ignore"):
        return np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)), np.arctan(sz / sx)

def _tiempo_archivo(key):
    m = re.search(r"_s(\d{13})", key)
    return dt.datetime.strptime(m.group(1), "%Y%j%H%M%S").replace(tzinfo=dt.timezone.utc) if m else None

def listar(fs, canal, horas=3):
    """Archivos de la banda de las últimas horas, del más viejo al más nuevo: [(hora_utc, ruta)]."""
    now = dt.datetime.utcnow(); out = []
    for back in range(horas + 1):
        t = now - dt.timedelta(hours=back)
        carpeta = f"noaa-goes19/ABI-L2-CMIPF/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"
        try: ls = fs.ls(carpeta)
        except FileNotFoundError: continue
        for k in ls:
            if f"{canal}_G19" in k:
                ts = _tiempo_archivo(k)
                if ts: out.append((ts, k))
    return sorted(set(out))

# ---------------------------------------------------------------- lectura + reproyección
def malla():
    lats = np.arange(LAT_MAX, LAT_MIN, -1.0)       # se redefine por banda según el paso
    return lats

def recorte(ds, paso):
    """Valores de CMI sobre una grilla regular lat/lon (norte arriba) con el paso dado. Devuelve (matriz, lat2d, lon2d)."""
    proj = ds["goes_imager_projection"]
    P = {"H": float(proj.perspective_point_height + proj.semi_major_axis), "r_eq": float(proj.semi_major_axis),
         "r_pol": float(proj.semi_minor_axis), "lam0": float(np.radians(proj.longitude_of_projection_origin))}
    x, y = ds["x"].values.astype("float64"), ds["y"].values.astype("float64")
    lats = np.arange(LAT_MAX - paso / 2, LAT_MIN, -paso); lons = np.arange(LON_MIN + paso / 2, LON_MAX, paso)
    lon2d, lat2d = np.meshgrid(lons, lats)
    xs, ys = _angulos(P, lat2d, lon2d)
    # índice fraccionario en el eje x (creciente) e y (decreciente en GOES-R)
    fx = np.interp(xs, x, np.arange(len(x)), left=np.nan, right=np.nan)
    ordy = np.argsort(y); fy = np.interp(ys, y[ordy], ordy.astype(float), left=np.nan, right=np.nan)
    ok = np.isfinite(fx) & np.isfinite(fy)
    if not ok.any(): raise RuntimeError("la región no cae dentro de la imagen")
    ix = np.where(ok, np.rint(fx), 0).astype(int); iy = np.where(ok, np.rint(fy), 0).astype(int)
    i0, i1, j0, j1 = iy[ok].min(), iy[ok].max() + 1, ix[ok].min(), ix[ok].max() + 1
    sub = ds["CMI"].isel(y=slice(i0, i1), x=slice(j0, j1)).values.astype("float32")   # solo se descargan los bloques de la región
    v = sub[iy - i0, ix - j0]; v[~ok] = np.nan
    return v, lat2d, lon2d

# ---------------------------------------------------------------- colores
def _hex(c): return [int(c[i:i + 2], 16) for i in (1, 3, 5)]

def colorear(valores, paleta):
    """Matriz de valores (°C) -> RGBA uint8 interpolando la paleta; NaN transparente."""
    v = np.array([p[0] for p in paleta], float); rgb = np.array([_hex(p[1]) for p in paleta], float)
    out = np.zeros(valores.shape + (4,), np.uint8)
    ok = np.isfinite(valores); vv = np.clip(np.where(ok, valores, v[0]), v[0], v[-1])
    for k in range(3): out[..., k] = np.rint(np.interp(vv, v, rgb[:, k])).astype(np.uint8)
    out[..., 3] = np.where(ok, 255, 0)
    return out

def coseno_zenital(lat, lon, ts):
    """Coseno del ángulo cenital del Sol (fórmulas de NOAA) para cada punto."""
    doy = ts.timetuple().tm_yday; hh = ts.hour + ts.minute / 60 + ts.second / 3600
    g = 2 * np.pi / 365 * (doy - 1 + (hh - 12) / 24)
    eq = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
           - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians((hh * 60 + eq + 4 * lon) / 4 - 180)
    la = np.radians(lat)
    return np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)

def imagen_visible(refl, lat2d, lon2d, ts):
    """Reflectancia -> gris con alfa. Se normaliza por el ángulo del Sol y se desvanece hacia el terminador. None si es de noche."""
    cz = coseno_zenital(lat2d, lon2d, ts)
    if float(np.nanmedian(cz)) < 0.12: return None               # de noche (o recién amaneciendo): no hay visible útil
    r = np.clip(refl / np.clip(cz, 0.2, 1), 0, 1.2)
    g = np.clip(r, 0, 1) ** 0.55                                    # realce de contraste suave
    a = np.clip((cz - 0.06) / 0.18, 0, 1) * np.isfinite(refl)
    out = np.zeros(refl.shape + (2,), np.uint8)
    out[..., 0] = np.rint(np.nan_to_num(g) * 255).astype(np.uint8); out[..., 1] = np.rint(a * 255).astype(np.uint8)
    return out

def a_png(arr):
    from PIL import Image
    b = io.BytesIO(); Image.fromarray(arr, "LA" if arr.shape[-1] == 2 else "RGBA").save(b, "PNG", optimize=True)
    return b.getvalue()

def generar(ds, banda, ts):
    """PNG (bytes) y metadatos de un cuadro. None si no corresponde (p. ej. visible de noche)."""
    cfg = BANDAS[banda]
    v, lat2d, lon2d = recorte(ds, cfg["paso"])
    if banda == "vis":
        arr = imagen_visible(v, lat2d, lon2d, ts)
        if arr is None: return None
        leyenda = [[0, "#000000"], [1, "#ffffff"]]
    else:
        arr = colorear(v - 273.15, cfg["paleta"]); leyenda = cfg["paleta"]
    h, w = arr.shape[:2]
    meta = {"banda": banda, "canal": cfg["canal"], "nombre": cfg["nombre"], "unidad": cfg["unidad"], "w": w, "h": h,
            "bounds": [[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]], "leyenda": [[a, b] for a, b in leyenda]}
    return a_png(arr), meta

# ---------------------------------------------------------------- envío
def _tok(): return os.environ.get("GOES_TOKEN") or os.environ.get("MODELO3D_TOKEN", "")

def existentes(banda):
    try:
        r = requests.get(f"{BACKEND}/api/goes/{banda}/lista", timeout=60)
        r.raise_for_status(); return {c["epoch"] for c in r.json().get("cuadros", [])}
    except Exception as e:
        print("lista", banda, e); return set()

def enviar(banda, ts, png, meta):
    cuerpo = {"banda": banda, "ts": ts.isoformat(), "meta": meta, "png_b64": base64.b64encode(png).decode()}
    r = requests.post(f"{BACKEND}/api/goes/ingreso", json=cuerpo, headers={"X-Token": _tok()}, timeout=90)
    print("POST", banda, ts.isoformat(), r.status_code, r.text[:120]); r.raise_for_status()

def correr(bandas=None, n=None, subir=True):
    import s3fs, xarray as xr
    fs = s3fs.S3FileSystem(anon=True); resumen = {}
    for banda in (bandas or list(BANDAS)):
        cfg = BANDAS[banda]; cuantos = int(n or os.environ.get("GOES_CUADROS", cfg["n"]))
        try:
            archivos = listar(fs, cfg["canal"])[-cuantos:]
            ya = existentes(banda) if subir else set()
            nuevos = [(t, k) for t, k in archivos if int(t.timestamp()) not in ya]
            hechos = 0
            for ts, key in nuevos:
                with xr.open_dataset(fs.open(key), engine="h5netcdf") as ds:
                    res = generar(ds, banda, ts)
                if res is None: print(banda, ts.isoformat(), "omitido (sin luz solar)"); continue
                if subir: enviar(banda, ts, *res)
                hechos += 1
            resumen[banda] = {"archivos": len(archivos), "enviados": hechos}
        except Exception as e:
            print("ERROR", banda, type(e).__name__, e); resumen[banda] = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(resumen)); return resumen

if __name__ == "__main__":
    correr([a for a in sys.argv[1:] if a in BANDAS] or None)
