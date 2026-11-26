"""Corre el modelo para todos los radares y envía el JSON al backend. Uso: python -m modelo3d.run"""
import os, json, datetime as dt, requests, numpy as np
from . import termo, celda, fuentes

RADARES = {"RMA11": (-26.85, -65.2), "RMA20": (-31.4, -64.2), "RMA7": (-26.8, -60.5)}  # ref. aproximada; sondeo por celda

def correr(radares=("RMA11", "RMA20", "RMA7"), usar_satelite=True):
    resultado = {"generado_utc": dt.datetime.utcnow().isoformat() + "Z", "aviso": "Estructura estimada (modelo), no medición del barrido vertical", "radares": {}}
    for rd in radares:
        try:
            cs, meta = fuentes.celdas_meta(rd)
        except Exception as e:
            resultado["radares"][rd] = {"error": f"celdas: {e}", "celdas": []}; continue
        cs = sorted(cs, key=lambda c: -(c.get("max_dbz") or 0))[:15]     # las 15 más intensas
        salida, hsat, aviso_sat, cmi = [], None, None, None
        if usar_satelite and cs:
            try: cmi = fuentes.abrir_cmi(); hsat = cmi["hora"]
            except Exception as e:
                aviso_sat = f"{type(e).__name__}: {e}"; print("SAT ERROR", rd, aviso_sat)
        cache = {}
        for c in cs:
            k = (round(c["lat"], 1), round(c["lon"], 1))      # un sondeo por celda de 0.1° (se reutiliza)
            try:
                if k not in cache:
                    p, z, T, Td, u, v = fuentes.sondeo(*k); cache[k] = (z, T, termo.analizar(p, z, T, Td, u, v))
                z, T, amb = cache[k]
                tb, top, par = None, None, None
                if cmi is not None:
                    radio = float(np.clip(np.sqrt((c.get("area_km2") or 30) / np.pi) + 3, 6, 14))
                    h = 9000.0                                  # primer intento: tope típico; se corrige con el paralaje
                    for _ in range(3):                          # el tope se ve corrido según su altura: se busca en el píxel correcto
                        tb = fuentes.tb_en(cmi, c["lat"], c["lon"], h, radio)
                        nuevo = termo.altura_por_temperatura(z, T, tb) if (tb is not None and tb < -5) else None
                        if nuevo is None: break
                        top, par = nuevo, fuentes.paralaje_km(c["lat"], c["lon"], nuevo)[0]
                        if abs(nuevo - h) < 500: break
                        h = float(np.clip(nuevo, 3000.0, 17000.0))
                s = celda.modelar(c, amb, top); s["tb_ir_c"] = None if tb is None else round(tb, 1); s["ambiente"] = amb
                if par is not None: s["paralaje_km"] = round(float(par), 1)
                salida.append(s)
            except Exception as e:
                salida.append({"id": c.get("id"), "error": str(e)})
        resultado["radares"][rd] = {"satelite_utc": hsat, "celdas": salida}
        if aviso_sat: resultado["radares"][rd]["aviso_sat"] = aviso_sat
        resultado["radares"][rd]["diagnostico"] = meta
    return resultado

def enviar(res):
    url = os.environ["BACKEND_URL"].rstrip("/") + "/api/modelo3d/ingreso"
    r = requests.post(url, json=res, headers={"X-Token": os.environ["MODELO3D_TOKEN"]}, timeout=60)
    print("POST", r.status_code, r.text[:200]); r.raise_for_status()

if __name__ == "__main__":
    res = correr(); print(json.dumps(res)[:400]); enviar(res)
