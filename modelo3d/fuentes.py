"""Descarga de datos: sondeo (Open-Meteo), satélite GOES-19 (AWS abierto) y celdas (backend). Requiere red."""
import os, re, datetime as dt, numpy as np, requests

BACKEND = os.environ.get("BACKEND_URL", "https://metricmoisture-backend.onrender.com")
NIVELES = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200]

def sondeo(lat, lon):
    """Perfil vertical del modelo numérico (Open-Meteo) para el punto. Devuelve arrays p,z,T,Td,u,v ordenados desde el suelo."""
    h = ",".join(f"{v}_{n}hPa" for n in NIVELES for v in
                 ("temperature", "relative_humidity", "geopotential_height", "wind_speed", "wind_direction"))
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=60, params={
        "latitude": lat, "longitude": lon, "hourly": h + ",surface_pressure,temperature_2m,dew_point_2m,wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "ms", "forecast_days": 1, "timezone": "UTC"})
    r.raise_for_status(); H = r.json()["hourly"]
    i = min(dt.datetime.utcnow().hour, len(H["time"]) - 1)
    from . import termo
    ps = H["surface_pressure"][i]; elev = r.json().get("elevation", 0)
    P = [ps]; Z = [elev]; T = [H["temperature_2m"][i]]; Td = [H["dew_point_2m"][i]]
    ws = [H["wind_speed_10m"][i]]; wd = [H["wind_direction_10m"][i]]
    for n in NIVELES:
        if n >= ps - 15: continue                      # nivel bajo el terreno
        t = H[f"temperature_{n}hPa"][i]; rh = H[f"relative_humidity_{n}hPa"][i]; zz = H[f"geopotential_height_{n}hPa"][i]
        if None in (t, rh, zz): continue
        P.append(n); Z.append(zz); T.append(t); Td.append(float(termo.td_from_rh(t, rh)))
        ws.append(H[f"wind_speed_{n}hPa"][i]); wd.append(H[f"wind_direction_{n}hPa"][i])
    ws, wd = np.array(ws), np.radians(wd)
    return (np.array(P, float), np.array(Z, float), np.array(T, float), np.array(Td, float),
            -ws * np.sin(wd), -ws * np.cos(wd))

def celdas_meta(radar):
    """Devuelve (lista de celdas, datos del seguimiento para diagnóstico)."""
    r = requests.get(f"{BACKEND}/api/radar/{radar}/seguimiento", timeout=60); r.raise_for_status()
    j = r.json()
    out = []
    if isinstance(j.get("celda"), dict):
        out.append(dict(j["celda"], principal=True))
    out += [c for c in (j.get("otras_celdas") or []) if isinstance(c, dict)]
    for i, c in enumerate(out):
        c.setdefault("id", f"{radar}-{i + 1}")
    ult = (j.get("frames") or [{}])[-1].get("hora")
    return out, {"seguimiento_actualizado": j.get("actualizado"), "ultimo_frame": ult, "mensaje": j.get("mensaje"), "n_celdas": len(out)}

def celdas(radar):
    return celdas_meta(radar)[0]

# ---- Satélite GOES-19 ABI banda 13 (IR 10.3 µm) ----
def _ultimo_cmi(banda="C13"):
    """Último archivo de la banda en el bucket público. Si la carpeta de la hora actual aún no existe (primeros minutos de cada
    hora) o está vacía, retrocede hasta 3 horas."""
    import s3fs
    fs = s3fs.S3FileSystem(anon=True); now = dt.datetime.utcnow(); ultimo_error = None
    for back in range(4):
        t = now - dt.timedelta(hours=back)
        carpeta = f"noaa-goes19/ABI-L2-CMIPF/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/"
        try:
            ls = fs.ls(carpeta)
        except FileNotFoundError as e:
            ultimo_error = e; continue
        ls = sorted(x for x in ls if f"{banda}_G19" in x)
        if ls: return fs, ls[-1]
    raise RuntimeError(f"sin archivos GOES recientes ({ultimo_error})")

def abrir_cmi():
    """Abre el último archivo de la banda 13 y deja listo lo necesario para ubicar cualquier punto en la imagen (se abre una sola vez por corrida)."""
    import xarray as xr
    fs, key = _ultimo_cmi()
    ds = xr.open_dataset(fs.open(key), engine="h5netcdf")
    proj = ds["goes_imager_projection"]
    m = re.search(r"_s(\d{14})", key)
    return {"H": proj.perspective_point_height + proj.semi_major_axis, "r_eq": proj.semi_major_axis, "r_pol": proj.semi_minor_axis,
            "lam0": np.radians(proj.longitude_of_projection_origin), "x": ds["x"].values, "y": ds["y"].values,
            "cmi": ds["CMI"].values, "hora": m.group(1) if m else None}

def _angulos(c, lat, lon, h_m=0.0):
    """Ángulos de barrido (x, y) con que el satélite ve un punto a h_m metros sobre la superficie en (lat, lon).
    Con h_m > 0 es el paralaje: el tope de la nube se ve en un píxel más alejado del punto subsatélite que su pie."""
    la, lo = np.radians(lat), np.radians(lon); r_eq, r_pol = c["r_eq"], c["r_pol"]
    phic = np.arctan((r_pol ** 2 / r_eq ** 2) * np.tan(la))
    rc = r_pol / np.sqrt(1 - ((r_eq ** 2 - r_pol ** 2) / r_eq ** 2) * np.cos(phic) ** 2) + h_m
    sx = c["H"] - rc * np.cos(phic) * np.cos(lo - c["lam0"]); sy = -rc * np.cos(phic) * np.sin(lo - c["lam0"]); sz = rc * np.sin(phic)
    return np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)), np.arctan(sz / sx)

def tb_en(c, lat, lon, h_m=0.0, radio_km=12):
    """Temperatura de brillo mínima (°C) alrededor de una celda, buscando en el píxel donde se ve su tope a h_m de altura."""
    xs, ys = _angulos(c, lat, lon, h_m)
    i, j = int(np.argmin(abs(c["y"] - ys))), int(np.argmin(abs(c["x"] - xs)))
    n = max(int(radio_km / 2.0), 2)   # ~2 km por píxel
    w = c["cmi"][max(i - n, 0): i + n + 1, max(j - n, 0): j + n + 1]
    return float(np.nanmin(w)) - 273.15 if np.isfinite(w).any() else None

def tb_minima(celdas_xy, radio_km=12):
    """Versión simple (sin paralaje, tope a nivel del suelo). La corrida usa abrir_cmi() + tb_en() con la altura del tope."""
    c = abrir_cmi()
    return [tb_en(c, la, lo, 0.0, radio_km) for la, lo in celdas_xy], c["hora"]

def paralaje_km(lat, lon, altura_m, sublon=-75.2):
    """Corrimiento (km) con que se ve una nube alta: su tope aparece desplazado d_km LEJOS del punto subsatélite (ecuador, sublon)
    respecto de su posición real. Devuelve (d_km, vza_rad). tb_en() lo aplica sola al ubicar el píxel; esto sirve para informarlo."""
    vza = np.arccos(np.clip(np.cos(np.radians(lat)) * np.cos(np.radians(lon - sublon)), -1, 1))
    return altura_m / 1000 * np.tan(vza), vza
