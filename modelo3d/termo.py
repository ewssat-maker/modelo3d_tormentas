"""Termodinámica de parcela en numpy puro (sin MetPy). Unidades: hPa, °C, m, m/s."""
import numpy as np

Rd, Cp, g, Lv, eps = 287.04, 1005.7, 9.80665, 2.501e6, 0.622

def es(T):            # presión de vapor de saturación (hPa), Bolton 1980
    return 6.112 * np.exp(17.67 * T / (T + 243.5))

def rs(p, T):         # razón de mezcla de saturación (kg/kg)
    e = es(T); return eps * e / (p - e)

def td_from_rh(T, rh):
    e = np.clip(rh, 1, 100) / 100 * es(T)
    l = np.log(e / 6.112); return 243.5 * l / (17.67 - l)

def lcl(p0, T0, Td0):
    """Presión (hPa) y T (°C) del LCL, Bolton 1980."""
    Tk, Tdk = T0 + 273.15, Td0 + 273.15
    TL = 1 / (1 / (Tdk - 56) + np.log(Tk / Tdk) / 800) + 56
    pL = p0 * (TL / Tk) ** (Cp / Rd)
    return pL, TL - 273.15

def _moist_lapse(p, Tk):
    """dT/dp pseudoadiabático (K/hPa)."""
    r = rs(p, Tk - 273.15)
    num = Rd * Tk + Lv * r
    den = Cp + Lv * Lv * r * eps / (Rd * Tk * Tk)
    return num / (den * p)

def parcela(p_prof, p0, T0, Td0):
    """Temperatura de la parcela (°C) en cada nivel p_prof (hPa, decreciente desde p0)."""
    pL, TL = lcl(p0, T0, Td0)
    out = np.empty_like(p_prof, dtype=float)
    Tk = TL + 273.15; pc = pL
    for i, p in enumerate(p_prof):
        if p >= pL:
            out[i] = (T0 + 273.15) * (p / p0) ** (Rd / Cp) - 273.15
        else:
            # integrar húmeda de pc a p en pasos de 5 hPa (RK2)
            while pc > p + 1e-9:
                dp = max(-5.0, p - pc)
                k1 = _moist_lapse(pc, Tk)
                k2 = _moist_lapse(pc + dp, Tk + k1 * dp)
                Tk += 0.5 * (k1 + k2) * dp; pc += dp
            out[i] = Tk - 273.15
    return out, pL

def z_de_p(p, z_sfc, T_c, p0):
    """Altura hidrostática (m) si no se dispone de geopotencial."""
    Tv = T_c + 273.15
    z = np.zeros_like(p); z[0] = z_sfc
    for i in range(1, len(p)):
        Tm = 0.5 * (Tv[i] + Tv[i - 1])
        z[i] = z[i - 1] + Rd * Tm / g * np.log(p[i - 1] / p[i])
    return z

def cruce(z, f, nivel=0.0, desde=0):
    """Primera altura >= desde donde f cruza `nivel` (interpolación lineal)."""
    for i in range(max(desde, 1), len(z)):
        a, b = f[i - 1] - nivel, f[i] - nivel
        if a == 0: return z[i - 1]
        if a * b < 0: return z[i - 1] + (z[i] - z[i - 1]) * a / (a - b)
    return None

def _una_parcela(p, z, T, Td):
    """Flotabilidad de una parcela que sale del primer nivel de los arrays. Devuelve dict o None."""
    pc, pL = parcela(p, p[0], T[0], Td[0])
    Tv_p = (pc + 273.15) * (1 + 0.61 * rs(p, pc))
    Tv_e = (T + 273.15) * (1 + 0.61 * rs(p, Td))
    flot = g * (Tv_p - Tv_e) / Tv_e
    zL = float(np.interp(pL, p[::-1], z[::-1]))
    lfc = el = None
    i0 = int(np.searchsorted(z, zL)); pos = flot > 0
    for i in range(max(i0, 1), len(z)):
        if lfc is None and pos[i] and not pos[i - 1]:
            lfc = z[i - 1] + (z[i] - z[i - 1]) * (-flot[i - 1]) / (flot[i] - flot[i - 1])
        if lfc is not None and (not pos[i]) and pos[i - 1]:
            el = z[i - 1] + (z[i] - z[i - 1]) * flot[i - 1] / (flot[i - 1] - flot[i]); break
    if lfc is None and pos[max(i0, 0)]: lfc = zL          # ya flotante en el LCL
    if lfc is not None and el is None: el = float(z[-1])
    cape = cin = 0.0
    for i in range(1, len(z)):
        dz = z[i] - z[i - 1]; b = 0.5 * (flot[i] + flot[i - 1])
        if lfc is None: continue
        if z[i] <= lfc: cin += min(b, 0) * dz
        elif z[i - 1] < el: cape += max(b, 0) * dz
    return dict(cape=float(cape), cin=float(cin) if lfc is not None else None, zL=zL,
                lfc=None if lfc is None else float(lfc), el=None if el is None else float(el))

def analizar(p, z, T, Td, u=None, v=None, capa_hpa=300):
    """Parcela MÁS INESTABLE entre los niveles dentro de `capa_hpa` sobre el suelo (típico en convección con sierras).
    p (hPa, decreciente), z (m s.n.m.), T, Td (°C)."""
    p, z, T, Td = map(np.asarray, (p, z, T, Td))
    mejor, orig = None, 0
    for k in range(len(p)):
        if p[k] < p[0] - capa_hpa: break
        r = _una_parcela(p[k:], z[k:], T[k:], Td[k:])
        if mejor is None or r["cape"] > mejor["cape"]: mejor, orig = r, k
    cape = mejor["cape"]
    z0 = cruce(z, T, 0.0); z20 = cruce(z, T, -20.0)
    r = dict(cape=round(cape, 1), cin=None if mejor["cin"] is None else round(mejor["cin"], 1),
             lcl_m=round(mejor["zL"]), lfc_m=None if mejor["lfc"] is None else round(mejor["lfc"]),
             el_m=None if mejor["el"] is None else round(mejor["el"]),
             iso0_m=None if z0 is None else round(z0), iso_m20_m=None if z20 is None else round(z20),
             w_max=round(float(np.sqrt(2 * cape)), 1), w_tipica=round(float(0.5 * np.sqrt(2 * cape)), 1),
             parcela_origen_hpa=float(p[orig]), z_suelo_m=round(float(z[0])))
    if u is not None and v is not None:
        zs = z - z[0]
        u6, v6 = np.interp(6000, zs, u), np.interp(6000, zs, v)
        u1, v1 = np.interp(1000, zs, u), np.interp(1000, zs, v)
        r["shear06_ms"] = round(float(np.hypot(u6 - u[0], v6 - v[0])), 1)
        r["shear01_ms"] = round(float(np.hypot(u1 - u[0], v1 - v[0])), 1)
        r["viento_medio_u"] = round(float(np.mean(u[zs <= 6000])), 2)
        r["viento_medio_v"] = round(float(np.mean(v[zs <= 6000])), 2)
        r["shear_u"], r["shear_v"] = round(float(u6 - u[0]), 2), round(float(v6 - v[0]), 2)
    return r

def altura_por_temperatura(z, T, Tb):
    """Altura donde el ambiente alcanza la T de brillo IR Tb (°C): tope de nube por IR (el más alto si hay inversión)."""
    z, T = np.asarray(z), np.asarray(T)
    for i in range(len(z) - 1, 0, -1):
        if (T[i] - Tb) * (T[i - 1] - Tb) <= 0 and T[i] != T[i - 1]:
            return float(z[i - 1] + (z[i] - z[i - 1]) * (Tb - T[i - 1]) / (T[i] - T[i - 1]))
    return float(z[0]) if Tb > T[0] else float(z[-1])
