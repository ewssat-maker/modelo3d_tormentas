"""Estructura 3D paramétrica de una celda: perfil vertical de Z, radio, inclinación, indicador de granizo.
Es una ESTIMACIÓN a partir de la física del ambiente; no una medición del barrido completo."""
import numpy as np

def perfil_z(zmax_dbz, base_m, iso0_m, tope_m, paso_m=500):
    """Reflectividad (dBZ) por nivel. Núcleo hasta la isoterma 0 °C, decae hacia el tope (~ -2.2 dBZ/km sobre iso0
    y caída rápida en el último tramo); bajo el núcleo, leve atenuación hacia la base."""
    zs = np.arange(0, tope_m + paso_m, paso_m, dtype=float)
    out = []
    nuc = min(max(iso0_m, base_m + 500), tope_m)
    for z in zs:
        if z < base_m: d = -25 * (base_m - z) / max(base_m, 1) - 10   # bajo la base: casi sin eco
        elif z <= nuc: d = zmax_dbz - 3 * (nuc - z) / max(nuc - base_m, 1)
        else:
            f = (z - nuc) / max(tope_m - nuc, 1)
            d = zmax_dbz - 3 * 0 - (zmax_dbz - 12) * f ** 1.5
        out.append([round(float(z)), round(float(max(d, -10)), 1)])
    return out

def radio_km(area_km2, z, tope_m):
    """Radio de la sección horizontal por altura: sección máxima = área observada; yunque ensancha arriba."""
    r0 = np.sqrt(max(area_km2, 1.0) / np.pi)
    f = z / max(tope_m, 1)
    return float(r0 * (0.6 + 0.4 * min(f * 1.4, 1) + (0.5 * max(f - 0.8, 0) / 0.2 if f > 0.8 else 0)))

def granizo(zmax, cape, iso0, iso20, tope, zdr=None, kdp=None):
    s = 0
    s += 2 if zmax >= 55 else 1 if zmax >= 50 else 0
    s += 2 if cape >= 2500 else 1 if cape >= 1200 else 0
    if tope and iso20 and tope - iso20 >= 3000: s += 1
    if iso0 and iso0 < 4800: s += 1
    if zdr is not None and kdp is not None and zmax >= 50 and zdr < 1.0: s += 1
    return {"indice": s, "nivel": "alto" if s >= 5 else "medio" if s >= 3 else "bajo"}

def modelar(celda, amb, tope_ir_m=None):
    """celda: dict del seguimiento del backend; amb: salida de termo.analizar. Devuelve estructura 3D."""
    zmax = float(celda.get("max_dbz") or 40)
    base = amb["lcl_m"]
    tope_t = amb.get("el_m") or (base + 6000)
    tope = float(tope_ir_m) if tope_ir_m else float(tope_t)
    tope = max(tope, base + 2000)
    iso0 = amb.get("iso0_m") or 4500
    rec = {"fuente_tope": "satélite IR" if tope_ir_m else "equilibrio termodinámico (EL)"}
    zs = perfil_z(zmax, base, iso0, tope)
    area = float(celda.get("area_km2") or 30)
    capas = [[z, d, round(radio_km(area, z, tope), 2)] for z, d in zs]
    # inclinación por cizalladura: desplazamiento horizontal ~ shear * tiempo de ascenso característico
    su, sv = amb.get("shear_u", 0), amb.get("shear_v", 0)
    k = tope / 1000 * 0.35   # km de desplazamiento por (m/s) de cizalla 0-6 km
    inc = {"dx_km": round(su * k / 10, 2), "dy_km": round(sv * k / 10, 2)}
    return {"id": celda.get("id"), "lat": celda.get("lat"), "lon": celda.get("lon"), "max_dbz": zmax,
            "base_m": base, "tope_m": round(tope), "iso0_m": iso0, "capas": capas, "inclinacion": inc,
            "granizo": granizo(zmax, amb["cape"], iso0, amb.get("iso_m20_m"), tope), "w_max": amb["w_max"], **rec}
