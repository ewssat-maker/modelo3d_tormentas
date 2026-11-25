"""Corre el modelo para todos los radares y envía el JSON al backend. Uso: python -m modelo3d.run"""
import os, json, datetime as dt, requests, numpy as np
from . import termo, celda, fuentes

RADARES = {"RMA11": (-26.85, -65.2), "RMA20": (-31.4, -64.2), "RMA7": (-26.8, -60.5)}  # ref. aproximada; sondeo por celda

def correr(radares=("RMA11", "RMA20", "RMA7"), usar_satelite=True):
    resultado = {"generado_utc": dt.datetime.utcnow().isoformat() + "Z", "aviso": "Estructura estimada (modelo), no medición del barrido vertical", "radares": {}}
    for rd in radares:
        try:
            cs = fuentes.celdas(rd)
        except Exception as e:
            resultado["radares"][rd] = {"error": f"celdas: {e}", "celdas": []}; continue
        cs = sorted(cs, key=lambda c: -(c.get("max_dbz") or 0))[:15]     # las 15 más intensas
        salida, tbs, hsat, aviso_sat = [], [None] * len(cs), None, None
        if usar_satelite and cs:
            try: tbs, hsat = fuentes.tb_minima([(c["lat"], c["lon"]) for c in cs])
            except Exception as e:
                aviso_sat = f"{type(e).__name__}: {e}"; print("SAT ERROR", rd, aviso_sat)
        cache = {}
        for c, tb in zip(cs, tbs):
            k = (round(c["lat"], 1), round(c["lon"], 1))      # un sondeo por celda de 0.1° (se reutiliza)
            try:
                if k not in cache:
                    p, z, T, Td, u, v = fuentes.sondeo(*k); cache[k] = (z, T, termo.analizar(p, z, T, Td, u, v))
                z, T, amb = cache[k]
                top = termo.altura_por_temperatura(z, T, tb) if (tb is not None and tb < -30) else None
                s = celda.modelar(c, amb, top); s["tb_ir_c"] = None if tb is None else round(tb, 1); s["ambiente"] = amb
                salida.append(s)
            except Exception as e:
                salida.append({"id": c.get("id"), "error": str(e)})
        resultado["radares"][rd] = {"satelite_utc": hsat, "celdas": salida}
        if aviso_sat: resultado["radares"][rd]["aviso_sat"] = aviso_sat
    return resultado

def enviar(res):
    url = os.environ["BACKEND_URL"].rstrip("/") + "/api/modelo3d/ingreso"
    r = requests.post(url, json=res, headers={"X-Token": os.environ["MODELO3D_TOKEN"]}, timeout=60)
    print("POST", r.status_code, r.text[:200]); r.raise_for_status()

if __name__ == "__main__":
    res = correr(); print(json.dumps(res)[:400]); enviar(res)
