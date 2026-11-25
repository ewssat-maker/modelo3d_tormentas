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

def celdas(radar):
    r = requests.get(f"{BACKEND}/api/radar/{radar}/seguimiento", timeout=60); r.raise_for_status()
    j = r.json()
    out = []
    if isinstance(j.get("celda"), dict):
        out.append(j["celda"])
    out += [c for c in (j.get("otras_celdas") or []) if isinstance(c, dict)]
    return out

# ---- Satélite GOES-19 ABI banda 13 (IR 10.3 µm) ----
def _ultimo_cmi(banda="C13"):
    import s3fs
    fs = s3fs.S3FileSystem(anon=True); now = dt.datetime.utcnow()
    for back in (0, 1):
        t = now - dt.timedelta(hours=back)
        ls = fs.ls(f"noaa-goes19/ABI-L2-CMIPF/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/")
        ls = sorted(x for x in ls if f"M6{banda}" in x or f"M3{banda}" in x or f"{banda}_G19" in x)
        if ls: return fs, ls[-1]
    raise RuntimeError("sin archivos GOES recientes")

def tb_minima(celdas_xy, radio_km=12):
    """Para cada (lat,lon) devuelve la temperatura de brillo mínima (°C) en su entorno (tope convectivo) y la hora del escaneo."""
    import xarray as xr
    fs, key = _ultimo_cmi()
    ds = xr.open_dataset(fs.open(key), engine="h5netcdf")
    proj = ds["goes_imager_projection"]; H = proj.perspective_point_height + proj.semi_major_axis
    r_eq, r_pol, lam0 = proj.semi_major_axis, proj.semi_minor_axis, np.radians(proj.longitude_of_projection_origin)
    x, y = ds["x"].values, ds["y"].values; cmi = ds["CMI"].values
    out = []
    for lat, lon in celdas_xy:
        la, lo = np.radians(lat), np.radians(lon)
        phic = np.arctan((r_pol ** 2 / r_eq ** 2) * np.tan(la))
        rc = r_pol / np.sqrt(1 - ((r_eq ** 2 - r_pol ** 2) / r_eq ** 2) * np.cos(phic) ** 2)
        sx = H - rc * np.cos(phic) * np.cos(lo - lam0); sy = -rc * np.cos(phic) * np.sin(lo - lam0); sz = rc * np.sin(phic)
        xs = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)); ys = np.arctan(sz / sx)
        i, j = int(np.argmin(abs(y - ys))), int(np.argmin(abs(x - xs)))
        n = max(int(radio_km / 2.0), 2)   # ~2 km por píxel
        w = cmi[max(i - n, 0): i + n + 1, max(j - n, 0): j + n + 1]
        out.append(float(np.nanmin(w)) - 273.15 if np.isfinite(w).any() else None)
    m = re.search(r"_s(\d{14})", key)
    return out, (m.group(1) if m else None)

def paralaje_km(lat, lon, altura_m, sublon=-75.2):
    """Corrimiento (km) de una nube alta vista por el satélite: aparece desplazada hacia el punto subsatélite
    (ecuador, sublon). Devuelve (d_km, vza_rad); la posición real queda a d_km del aparente, alejándose del subsatélite."""
    vza = np.arccos(np.clip(np.cos(np.radians(lat)) * np.cos(np.radians(lon - sublon)), -1, 1))
    return altura_m / 1000 * np.tan(vza), vza
