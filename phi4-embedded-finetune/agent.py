#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local AI Agent — harita tabanlı sürüm.

Ne değişti (v1'e göre):
  * PROJE HARİTASI. Ajan artık her soruda dosyaları sıfırdan okumuyor. Bir
    projeye ilk girdiğinde tüm ağacı bir kez tarayıp `.ajan_harita.json`
    dosyasını oluşturuyor. Sonraki her işlemde bu haritayı okuyor; sadece
    boyutu/tarihi değişmiş dosyaları yeniden ayrıştırıyor.
  * BAĞIMLILIK ve ETKİ ANALİZİ. Harita include ilişkilerini, sembolleri ve
    fonksiyon çağrılarını ÇİFT YÖNLÜ tutuyor. "Bu dosyayı değiştirirsem ne
    kırılır" bir graf sorgusu; yeniden tarama gerektirmiyor.
  * Araç sırası harita öncelikli: önce harita_ozet / dosya_bilgi / etki_analizi,
    en son read_file.
  * Dosya okuma ve yazma artık onay sormuyor.
  * v1'deki döngü hataları düzeltildi (aşağıda "DÜZELTME" notlarına bak).

Çalıştırma:
    python agent.py                 # o anki klasörü proje kabul eder
    python agent.py /proje/yolu     # belirtilen klasörü haritalar

REPL içinde:
    :proje <yol>    başka bir projeye geç
    :harita         haritanın özetini yazdır
    :yenile         haritayı sıfırdan kur
    :araclar        araç listesi
    exit            çık
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# AYARLAR
# ---------------------------------------------------------------------------

LLAMA_SERVER_URL = os.environ.get(
    "LLAMA_URL", "http://localhost:8080/v1/chat/completions")
MODEL_ADI = os.environ.get("LLAMA_MODEL", "phi-4")
MAX_ITERATIONS = 14
MAX_TOKENS = 1500
SICAKLIK = 0.3

HARITA_DOSYASI = ".ajan_harita.json"
HARITA_SURUM = 1

# Shell komutları için onay. Dosya okuma/yazma zaten hiç sormuyor.
# Kendine güveniyorsan False bırak; tedbirli olmak istersen True yap.
SHELL_ONAY_SOR = False

# Haritaya alınmayacak klasör/dosya kalıpları
HARIC = [
    ".git", ".svn", ".embagent", "build", "Build", "cmake-build-*", "out", "dist", ".pio",
    ".vscode", ".idea", "node_modules", "__pycache__", ".venv", "venv",
    "managed_components", ".ajan_harita.json", "*.o", "*.d", "*.elf", "*.bin",
    "*.hex", "*.map", "*.su", "*.pyc", "*.zip", "*.rar", "*.7z", "*.exe",
]

MAX_DOSYA_BOYUT = 2_000_000   # bundan büyük dosyalar haritaya alınmaz

C_UZANTI = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".inc", ".ino"}
DTS_UZANTI = {".dts", ".dtsi", ".overlay"}
LD_UZANTI = {".ld", ".lds", ".icf"}
ASM_UZANTI = {".s", ".asm"}
METIN_UZANTI = (C_UZANTI | DTS_UZANTI | LD_UZANTI | ASM_UZANTI |
                {".cmake", ".mk", ".mak", ".py", ".txt", ".conf", ".ini",
                 ".yml", ".yaml", ".json", ".md", ".cfg", ".prj", ".rs",
                 ".defaults", ".csv"})


# ===========================================================================
# BÖLÜM 1 — KOD ÇÖZÜMLEYİCİ
# ===========================================================================
# libclang gibi ağır bir bağımlılık yok. İhtiyacımız olan şey bir derleyici
# değil, bir harita: kim kimi include ediyor, hangi sembol nerede, kim kimi
# çağırıyor. Bunun için "maskeleme" yeterli.

def maskele(kaynak):
    """Yorumların ve string içeriklerinin yerine boşluk koyar.

    Her byte'ın yerini ve satır sonlarını korur; böylece maskelenmiş metinde
    bulunan konum, orijinal dosyada da aynı satıra denk gelir. Regex'lerin
    yorum satırındaki koda ya da string içindeki süslü paranteze takılmasını
    bu engelliyor.
    """
    cikti = list(kaynak)
    i, n = 0, len(kaynak)
    while i < n:
        c = kaynak[i]
        s = kaynak[i + 1] if i + 1 < n else ""
        if c == "/" and s == "/":
            while i < n and kaynak[i] != "\n":
                cikti[i] = " "
                i += 1
        elif c == "/" and s == "*":
            cikti[i] = cikti[i + 1] = " "
            i += 2
            while i < n and not (kaynak[i] == "*" and i + 1 < n and kaynak[i + 1] == "/"):
                if kaynak[i] != "\n":
                    cikti[i] = " "
                i += 1
            if i < n:
                cikti[i] = " "
                if i + 1 < n:
                    cikti[i + 1] = " "
                i += 2
        elif c in ("'", '"'):
            tirnak = c
            i += 1
            while i < n and kaynak[i] != tirnak:
                if kaynak[i] == "\\":
                    cikti[i] = " "
                    i += 1
                    if i < n and kaynak[i] != "\n":
                        cikti[i] = " "
                        i += 1
                    continue
                if kaynak[i] != "\n":
                    cikti[i] = " "
                i += 1
            i += 1
        else:
            i += 1
    return "".join(cikti)


def _satir_indeksi(kaynak):
    idx, p = [0], kaynak.find("\n")
    while p != -1:
        idx.append(p + 1)
        p = kaynak.find("\n", p + 1)
    return idx


def _satir_no(idx, konum):
    alt, ust = 0, len(idx) - 1
    while alt < ust:
        orta = (alt + ust + 1) // 2
        if idx[orta] <= konum:
            alt = orta
        else:
            ust = orta - 1
    return alt + 1


def _paren_eslesi(metin, acilis):
    derinlik = 0
    for i in range(acilis, len(metin)):
        if metin[i] == "(":
            derinlik += 1
        elif metin[i] == ")":
            derinlik -= 1
            if derinlik == 0:
                return i
    return -1


def _suslu_eslesi(metin, acilis):
    derinlik = 0
    for i in range(acilis, len(metin)):
        if metin[i] == "{":
            derinlik += 1
        elif metin[i] == "}":
            derinlik -= 1
            if derinlik == 0:
                return i
    return -1


ANAHTAR = {
    "if", "for", "while", "switch", "return", "sizeof", "defined", "do", "else",
    "case", "catch", "static_assert", "_Static_assert", "alignof", "_Alignof",
    "__attribute__", "typeof", "__typeof__", "decltype", "and", "or", "not",
    "new", "delete", "throw", "constexpr", "noexcept", "explicit", "template",
    "typename", "operator", "asm", "__asm__", "volatile", "register", "return",
}
KUYRUK = {"const", "noexcept", "override", "final", "volatile", "__attribute__",
          "IRAM_ATTR", "DRAM_ATTR", "__weak", "__always_inline", "_Noreturn"}

TIP_IPUCU = re.compile(
    r"\b(void|char|short|int|long|float|double|signed|unsigned|bool|_Bool|"
    r"static|inline|extern|const|volatile|struct|union|enum|auto|"
    r"[a-zA-Z_]\w*_t|uint\d+_t|int\d+_t|size_t|esp_err_t|BaseType_t|"
    r"HAL_StatusTypeDef|IRAM_ATTR|DRAM_ATTR|__weak)\b")

RE_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*([<"])([^>"]+)[>"]', re.M)
RE_DEFINE = re.compile(r'^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)', re.M)
RE_KAYIT = re.compile(r'\b(struct|union|enum|class)\s+([A-Za-z_]\w*)\s*(?=[:{])')
RE_TYPEDEF_SON = re.compile(r'\}\s*([A-Za-z_]\w*)\s*;')
RE_CAGRI = re.compile(r'\b([A-Za-z_]\w*)\s*\(')

RE_ISR = re.compile(r'\b(IRAM_ATTR|__attribute__\s*\(\s*\(\s*interrupt|ISR\s*\()')
RE_ISR_YASAK = re.compile(r'\b(vTaskDelay|HAL_Delay|delay|printf|malloc|free|'
                          r'sprintf|osDelay)\s*\(')
RE_TASK = re.compile(r'\bxTaskCreate(?:Static|PinnedToCore)?\s*\(')


def bos_kayit(dil="metin", satir=0):
    return {"dil": dil, "satir": satir, "include": [], "sembol": [], "cagri": {},
            "saglar": [], "gerektirir": [], "kaynaklar": [], "isaret": []}


def c_coz(kaynak):
    k = bos_kayit("c", kaynak.count("\n") + 1)
    m_kaynak = maskele(kaynak)
    idx = _satir_indeksi(kaynak)

    # DİKKAT: include'lar HAM kaynakta aranır. maskele() string içeriğini
    # siliyor ve #include "x.h" ifadesindeki tırnak bir string literal.
    for m in RE_INCLUDE.finditer(kaynak):
        k["include"].append({"ad": m.group(2), "sistem": m.group(1) == "<",
                             "satir": _satir_no(idx, m.start())})

    for m in RE_DEFINE.finditer(m_kaynak):
        k["sembol"].append({"ad": m.group(1), "tur": "makro",
                            "satir": _satir_no(idx, m.start())})
    for m in RE_KAYIT.finditer(m_kaynak):
        k["sembol"].append({"ad": m.group(2), "tur": "tip",
                            "satir": _satir_no(idx, m.start()),
                            "imza": f"{m.group(1)} {m.group(2)}"})
    for m in RE_TYPEDEF_SON.finditer(m_kaynak):
        k["sembol"].append({"ad": m.group(1), "tur": "tip",
                            "satir": _satir_no(idx, m.start())})

    govdeler = []
    konum, n = 0, len(m_kaynak)
    while konum < n:
        m = RE_CAGRI.search(m_kaynak, konum)
        if not m:
            break
        ad = m.group(1)
        acilis = m.end() - 1
        if ad in ANAHTAR:
            konum = m.end()
            continue
        kapanis = _paren_eslesi(m_kaynak, acilis)
        if kapanis == -1:
            konum = m.end()
            continue

        j = kapanis + 1
        while j < n:
            if m_kaynak[j].isspace():
                j += 1
                continue
            kelime = re.match(r'[A-Za-z_]\w*', m_kaynak[j:])
            if kelime and kelime.group(0) in KUYRUK:
                j += kelime.end()
                continue
            if m_kaynak[j] == "(" and kelime is None:
                ic = _paren_eslesi(m_kaynak, j)
                if ic != -1:
                    j = ic + 1
                    continue
            break

        bas = max(m_kaynak.rfind(";", 0, m.start()),
                  m_kaynak.rfind("}", 0, m.start()),
                  m_kaynak.rfind("{", 0, m.start()))
        onek = m_kaynak[bas + 1:m.start()].strip()

        if j < n and m_kaynak[j] == "{":
            if TIP_IPUCU.search(onek) or onek == "":
                son = _suslu_eslesi(m_kaynak, j)
                if son == -1:
                    son = n - 1
                imza = " ".join((onek + " " + ad +
                                 m_kaynak[acilis:kapanis + 1]).split())[:250]
                k["sembol"].append({
                    "ad": ad, "tur": "fonksiyon",
                    "satir": _satir_no(idx, m.start()),
                    "son_satir": _satir_no(idx, son), "imza": imza,
                    "statik": bool(re.search(r'\bstatic\b', onek))})
                govdeler.append((ad, j, son, imza))
                konum = j + 1          # gövdenin içine gir, çağrıları topla
                continue
            konum = m.end()
            continue

        if j < n and m_kaynak[j] == ";" and TIP_IPUCU.search(onek):
            imza = " ".join((onek + " " + ad +
                             m_kaynak[acilis:kapanis + 1]).split())[:250]
            k["sembol"].append({"ad": ad, "tur": "prototip",
                                "satir": _satir_no(idx, m.start()), "imza": imza})
        konum = m.end()

    for ad, bas, son, imza in govdeler:
        cagrilar = []
        for cm in RE_CAGRI.finditer(m_kaynak, bas, son):
            c = cm.group(1)
            if c not in ANAHTAR and c != ad and c not in cagrilar:
                cagrilar.append(c)
        if cagrilar:
            k["cagri"][ad] = cagrilar[:150]

    # Gömülü sisteme özgü işaretler
    if RE_TASK.search(m_kaynak):
        k["isaret"].append("rtos-task-olusturuyor")
    for ad, bas, son, imza in govdeler:
        govde = m_kaynak[bas:son]
        isr_mi = (RE_ISR.search(imza) or
                  ad.lower().endswith(("_isr", "_irqhandler", "_handler")))
        if isr_mi:
            kotu = sorted({b.group(1) for b in RE_ISR_YASAK.finditer(govde)})
            satir = next((s["satir"] for s in k["sembol"]
                          if s["ad"] == ad and s["tur"] == "fonksiyon"), 0)
            if kotu:
                k["isaret"].append(
                    f"ISR {ad}() satir {satir}: kesme icinde bloklayici cagri "
                    f"({', '.join(kotu)})")
            else:
                k["isaret"].append(f"ISR {ad}() satir {satir}")
    if re.search(r'\b(malloc|calloc|realloc)\s*\(', m_kaynak):
        k["isaret"].append("dinamik-bellek-kullanimi")
    if re.search(r'\b(strcpy|strcat|sprintf|gets)\s*\(', m_kaynak):
        k["isaret"].append("sinirsiz-string-fonksiyonu")
    return k


RE_IDF = re.compile(r'idf_component_register\s*\((.*?)\)', re.S | re.I)
RE_CMAKE_ANAHTAR = re.compile(
    r'\b(SRCS|SRC_DIRS|INCLUDE_DIRS|REQUIRES|PRIV_REQUIRES|PRIV_INCLUDE_DIRS)\b')
RE_ADD_EXE = re.compile(r'\badd_(executable|library)\s*\(\s*([A-Za-z0-9_\-\.]+)(.*?)\)',
                        re.S | re.I)
RE_LINK = re.compile(r'\btarget_link_libraries\s*\(\s*([A-Za-z0-9_\-\.]+)(.*?)\)',
                     re.S | re.I)
RE_SUBDIR = re.compile(r'\badd_subdirectory\s*\(\s*([^\s)]+)', re.I)
RE_PROJE = re.compile(r'\bproject\s*\(\s*([A-Za-z0-9_\-\.]+)', re.I)


def cmake_coz(kaynak):
    k = bos_kayit("cmake", kaynak.count("\n") + 1)
    metin = re.sub(r'#[^\n]*', "", kaynak)
    for m in RE_PROJE.finditer(metin):
        k["saglar"].append(m.group(1))
        k["sembol"].append({"ad": m.group(1), "tur": "proje", "satir": 1})
    for m in RE_IDF.finditer(metin):
        aktif = None
        for tok in re.findall(r'[^\s()]+', m.group(1)):
            if RE_CMAKE_ANAHTAR.fullmatch(tok):
                aktif = tok.upper()
                continue
            tok = tok.strip('"\'')
            if not tok:
                continue
            if aktif in ("SRCS", "SRC_DIRS"):
                k["kaynaklar"].append(tok)
            elif aktif in ("REQUIRES", "PRIV_REQUIRES"):
                k["gerektirir"].append(tok)
            elif aktif in ("INCLUDE_DIRS", "PRIV_INCLUDE_DIRS"):
                k["isaret"].append(f"include-dizini:{tok}")
    for m in RE_ADD_EXE.finditer(metin):
        k["saglar"].append(m.group(2))
        k["sembol"].append({"ad": m.group(2), "tur": "hedef", "satir": 1})
        for tok in re.findall(r'[^\s()]+', m.group(3)):
            tok = tok.strip('"\'')
            if tok and not tok.isupper() and "." in tok:
                k["kaynaklar"].append(tok)
    for m in RE_LINK.finditer(metin):
        for tok in re.findall(r'[^\s()]+', m.group(2)):
            tok = tok.strip('"\'')
            if tok and tok.upper() not in ("PUBLIC", "PRIVATE", "INTERFACE"):
                k["gerektirir"].append(tok)
    for m in RE_SUBDIR.finditer(metin):
        k["gerektirir"].append(m.group(1).strip('"\''))
    k["saglar"] = sorted(set(k["saglar"]))
    k["gerektirir"] = sorted(set(k["gerektirir"]))
    k["kaynaklar"] = sorted(set(k["kaynaklar"]))
    return k


def kconfig_coz(kaynak):
    k = bos_kayit("kconfig", kaynak.count("\n") + 1)
    idx = _satir_indeksi(kaynak)
    for m in re.finditer(r'^\s*(menuconfig|config)\s+([A-Za-z0-9_]+)', kaynak, re.M):
        k["sembol"].append({"ad": m.group(2), "tur": "config",
                            "satir": _satir_no(idx, m.start())})
    k["gerektirir"] = sorted({m.group(1) for m in re.finditer(
        r'^\s*(?:depends on|select)\s+([A-Za-z0-9_]+)', kaynak, re.M)})
    for m in re.finditer(r'^\s*r?source\s+"([^"]+)"', kaynak, re.M):
        k["include"].append({"ad": m.group(1), "sistem": False,
                             "satir": _satir_no(idx, m.start())})
    return k


def dts_coz(kaynak):
    k = bos_kayit("devicetree", kaynak.count("\n") + 1)
    idx = _satir_indeksi(kaynak)
    for m in re.finditer(r'^\s*#include\s*[<"]([^>"]+)[>"]', kaynak, re.M):
        k["include"].append({"ad": m.group(1), "sistem": True,
                             "satir": _satir_no(idx, m.start())})
    for m in re.finditer(r'^\s*([A-Za-z0-9_\-]+)\s*:\s*([A-Za-z0-9_\-@,\.]+)\s*\{',
                         kaynak, re.M):
        k["sembol"].append({"ad": m.group(1), "tur": "dugum",
                            "satir": _satir_no(idx, m.start()), "imza": m.group(2)})
    k["saglar"] = sorted({m.group(1) for m in
                          re.finditer(r'compatible\s*=\s*"([^"]+)"', kaynak)})
    return k


def linker_coz(kaynak):
    k = bos_kayit("linker", kaynak.count("\n") + 1)
    idx = _satir_indeksi(kaynak)
    for m in re.finditer(
            r'([A-Za-z0-9_\.]+)\s*\([^)]*\)\s*:\s*ORIGIN\s*=\s*([^,]+),\s*LENGTH\s*=\s*(\S+)',
            kaynak, re.I):
        k["sembol"].append({"ad": m.group(1), "tur": "bellek-bolgesi",
                            "satir": _satir_no(idx, m.start()),
                            "imza": f"ORIGIN={m.group(2).strip()} LENGTH={m.group(3).strip()}"})
    for m in re.finditer(r'^\s*(\.[A-Za-z0-9_\.]+)\s*:', kaynak, re.M):
        k["sembol"].append({"ad": m.group(1), "tur": "bolum",
                            "satir": _satir_no(idx, m.start())})
    return k


def make_coz(kaynak):
    k = bos_kayit("make", kaynak.count("\n") + 1)
    idx = _satir_indeksi(kaynak)
    for m in re.finditer(r'^([A-Za-z0-9_\-\.\/%$()]+)\s*:(?!=)', kaynak, re.M):
        k["sembol"].append({"ad": m.group(1), "tur": "hedef",
                            "satir": _satir_no(idx, m.start())})
    return k


def py_coz(kaynak):
    k = bos_kayit("python", kaynak.count("\n") + 1)
    idx = _satir_indeksi(kaynak)
    for m in re.finditer(r'^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.,\s]+))',
                         kaynak, re.M):
        mod = m.group(1) or (m.group(2) or "").split(",")[0].strip()
        if mod:
            k["include"].append({"ad": mod, "sistem": True,
                                 "satir": _satir_no(idx, m.start())})
    for m in re.finditer(r'^\s*(?:async\s+)?def\s+(\w+)\s*\(', kaynak, re.M):
        k["sembol"].append({"ad": m.group(1), "tur": "fonksiyon",
                            "satir": _satir_no(idx, m.start())})
    for m in re.finditer(r'^\s*class\s+(\w+)', kaynak, re.M):
        k["sembol"].append({"ad": m.group(1), "tur": "tip",
                            "satir": _satir_no(idx, m.start())})
    return k


def asm_coz(kaynak):
    k = bos_kayit("asm", kaynak.count("\n") + 1)
    idx = _satir_indeksi(kaynak)
    for m in re.finditer(r'^\s*\.?(?:global|globl|type)\s+([A-Za-z_\.]\w*)',
                         kaynak, re.M):
        k["sembol"].append({"ad": m.group(1), "tur": "fonksiyon",
                            "satir": _satir_no(idx, m.start())})
    return k


def kaynak_coz(gorece_yol, kaynak):
    ad = gorece_yol.rsplit("/", 1)[-1].lower()
    uzanti = ("." + ad.rsplit(".", 1)[-1]) if "." in ad else ""
    if uzanti in C_UZANTI:
        return c_coz(kaynak)
    if ad == "cmakelists.txt" or uzanti == ".cmake":
        return cmake_coz(kaynak)
    if ad.startswith("kconfig"):
        return kconfig_coz(kaynak)
    if uzanti in DTS_UZANTI:
        return dts_coz(kaynak)
    if uzanti in LD_UZANTI:
        return linker_coz(kaynak)
    if ad in ("makefile", "gnumakefile") or uzanti in (".mk", ".mak"):
        return make_coz(kaynak)
    if uzanti == ".py":
        return py_coz(kaynak)
    if uzanti in ASM_UZANTI:
        return asm_coz(kaynak)
    return bos_kayit(uzanti.lstrip(".") or "metin", kaynak.count("\n") + 1)


# ===========================================================================
# BÖLÜM 2 — PROJE HARİTASI
# ===========================================================================

EKOSISTEM_IZLERI = [
    ("ESP-IDF", re.compile(r"(^|/)sdkconfig(\.defaults.*)?$"), 40),
    ("ESP-IDF", re.compile(r"(^|/)idf_component\.yml$"), 30),
    ("ESP-IDF", re.compile(r"^main/CMakeLists\.txt$"), 25),
    ("ESP-IDF", re.compile(r"^components/[^/]+/CMakeLists\.txt$"), 20),
    ("Zephyr", re.compile(r"(^|/)west\.ya?ml$"), 40),
    ("Zephyr", re.compile(r"(^|/)prj\.conf$"), 40),
    ("Zephyr", re.compile(r"\.overlay$"), 20),
    ("STM32 HAL/CubeMX", re.compile(r"\.ioc$"), 45),
    ("STM32 HAL/CubeMX", re.compile(r"(^|/)startup_stm32.*\.s$"), 30),
    ("STM32 HAL/CubeMX", re.compile(r"Drivers/STM32.*_HAL_Driver/"), 30),
    ("STM32 HAL/CubeMX", re.compile(r"^Core/(Src|Inc)/"), 20),
    ("PlatformIO", re.compile(r"(^|/)platformio\.ini$"), 50),
    ("Arduino", re.compile(r"\.ino$"), 40),
    ("Rust gomulu", re.compile(r"(^|/)memory\.x$"), 40),
    ("Bare-metal C", re.compile(r"\.ld$"), 15),
    ("Bare-metal C", re.compile(r"(^|/)startup.*\.s$"), 15),
    ("Bare-metal C", re.compile(r"(^|/)(CMSIS|cmsis)/"), 15),
]

# Katman sırası: aşağıdaki katman yukarıdakine bağımlı OLMAMALI.
KATMAN = {"app": 6, "main": 6, "application": 6, "src": 5, "features": 5,
          "services": 4, "middleware": 4, "middlewares": 4, "lib": 4,
          "components": 4, "drivers": 3, "driver": 3, "bsp": 2, "hal": 2,
          "port": 2, "arch": 1, "cmsis": 1, "vendor": 1, "core": 1,
          "startup": 0, "linker": 0}


def _katman(yol):
    for parca in yol.split("/")[:-1]:
        p = parca.lower()
        if p in KATMAN:
            return p, KATMAN[p]
    return None


def _haric_mi(yol):
    import fnmatch
    parcalar = yol.split("/")
    for kalip in HARIC:
        if fnmatch.fnmatch(yol, kalip) or fnmatch.fnmatch(parcalar[-1], kalip):
            return True
        if any(fnmatch.fnmatch(p, kalip) for p in parcalar[:-1]):
            return True
    return False


def _metin_oku(yol, limit=MAX_DOSYA_BOYUT):
    try:
        ham = Path(yol).read_bytes()[:limit]
    except OSError:
        return None
    if b"\x00" in ham[:4096]:
        return None
    return ham.decode("utf-8", "replace")


class ProjeHaritasi:
    """Projenin yapısal indeksi.

    Akış tam olarak şu:
      1. Bir projeye bakarken önce .ajan_harita.json var mı diye bakılır.
      2. Yoksa bir kerelik tam tarama yapılıp oluşturulur.
      3. Varsa yüklenir; sadece boyut/tarih parmak izi değişmiş dosyalar
         yeniden ayrıştırılır. Değişmemişse hiçbir dosya açılmaz.
      4. Tüm sorgular (yapı, sembol, bağımlılık, etki) haritadan cevaplanır.
    """

    def __init__(self, kok):
        self.kok = Path(kok).resolve()
        self.dosyalar = {}
        self.ekosistem = []
        self.olusturma = 0.0
        self.tarama_ms = 0
        self._turev()

    # ------------------------------------------------------------ yaşam döngüsü

    @property
    def harita_yolu(self):
        return self.kok / HARITA_DOSYASI

    @classmethod
    def yukle_veya_olustur(cls, kok, zorla=False, yaz=print):
        h = cls(kok)
        yol = h.harita_yolu
        if yol.exists() and not zorla:
            try:
                veri = json.loads(yol.read_text(encoding="utf-8"))
                if veri.get("surum") == HARITA_SURUM:
                    h.dosyalar = veri.get("dosyalar", {})
                    h.ekosistem = veri.get("ekosistem", [])
                    h.olusturma = veri.get("olusturma", 0.0)
                    yaz(f"[harita] {len(h.dosyalar)} dosya {HARITA_DOSYASI} "
                        f"dosyasindan yuklendi")
                    degisen = h.guncelle(yaz)
                    if not degisen:
                        yaz("[harita] guncel, hicbir dosya yeniden okunmadi")
                    return h
                yaz(f"[harita] surum eski, yeniden kuruluyor")
            except (OSError, json.JSONDecodeError) as e:
                yaz(f"[harita] okunamadi ({e}), yeniden kuruluyor")

        yaz("[harita] harita yok — ilk tam tarama yapiliyor")
        h.tara(yaz)
        h.kaydet()
        return h

    def _yuru(self):
        cikti = []
        for klasor, altlar, dosyalar in os.walk(self.kok):
            gorece_k = os.path.relpath(klasor, self.kok).replace(os.sep, "/")
            gorece_k = "" if gorece_k == "." else gorece_k
            altlar[:] = [a for a in altlar
                         if not _haric_mi(f"{gorece_k}/{a}".lstrip("/"))]
            for ad in dosyalar:
                gorece = f"{gorece_k}/{ad}".lstrip("/")
                if _haric_mi(gorece):
                    continue
                try:
                    st = os.stat(os.path.join(klasor, ad))
                except OSError:
                    continue
                if st.st_size > MAX_DOSYA_BOYUT:
                    continue
                cikti.append((gorece, st.st_size, st.st_mtime))
        cikti.sort()
        return cikti

    def _bir_dosya(self, gorece, boyut, mtime):
        kayit = {"boyut": boyut, "iz": f"{boyut}:{int(mtime)}"}
        ad = gorece.rsplit("/", 1)[-1].lower()
        uzanti = ("." + ad.rsplit(".", 1)[-1]) if "." in ad else ""
        cozulur = (uzanti in METIN_UZANTI or
                   ad in ("makefile", "gnumakefile", "cmakelists.txt") or
                   ad.startswith("kconfig"))
        if not cozulur:
            kayit["dil"] = uzanti.lstrip(".") or "ikili"
            return kayit
        metin = _metin_oku(self.kok / gorece)
        if metin is None:
            kayit["dil"] = "ikili"
            return kayit
        kayit.update(kaynak_coz(gorece, metin))
        kayit["cozuldu"] = True
        for anahtar in ("include", "sembol", "cagri", "saglar", "gerektirir",
                        "kaynaklar", "isaret"):
            if not kayit.get(anahtar):
                kayit.pop(anahtar, None)
        return kayit

    def tara(self, yaz=print):
        basla = time.time()
        girdiler = self._yuru()
        yaz(f"[harita] {len(girdiler)} dosya taraniyor")
        self.dosyalar = {}
        for gorece, boyut, mtime in girdiler:
            self.dosyalar[gorece] = self._bir_dosya(gorece, boyut, mtime)
        self._ekosistem_belirle()
        self.olusturma = time.time()
        self.tarama_ms = int((self.olusturma - basla) * 1000)
        self._turev()
        yaz(f"[harita] {len(self.semboller)} sembol, "
            f"{sum(len(v) for v in self.include.values())} include baglantisi, "
            f"{self.tarama_ms} ms")

    def guncelle(self, yaz=print):
        girdiler = self._yuru()
        gorulen, degisen = set(), 0
        for gorece, boyut, mtime in girdiler:
            gorulen.add(gorece)
            eski = self.dosyalar.get(gorece)
            iz = f"{boyut}:{int(mtime)}"
            if eski and eski.get("iz") == iz:
                continue
            self.dosyalar[gorece] = self._bir_dosya(gorece, boyut, mtime)
            degisen += 1
        silinen = [g for g in self.dosyalar if g not in gorulen]
        for g in silinen:
            del self.dosyalar[g]
        if degisen or silinen:
            yaz(f"[harita] {degisen} dosya degismis, {len(silinen)} silinmis "
                f"— sadece bunlar yeniden okundu")
            self.olusturma = time.time()
            self._turev()
            self.kaydet()
        else:
            self._turev()
        return degisen + len(silinen)

    def dosya_tazele(self, gorece):
        """Ajan bir dosyayı yazdıktan sonra haritayı tam taramadan güncelle."""
        tam = self.kok / gorece
        if tam.exists():
            st = tam.stat()
            self.dosyalar[gorece] = self._bir_dosya(gorece, st.st_size, st.st_mtime)
        else:
            self.dosyalar.pop(gorece, None)
        self._turev()
        self.kaydet()

    def kaydet(self):
        try:
            self.harita_yolu.write_text(json.dumps({
                "surum": HARITA_SURUM,
                "olusturma": self.olusturma,
                "tarama_ms": self.tarama_ms,
                "kok": str(self.kok),
                "ekosistem": self.ekosistem,
                "dosyalar": self.dosyalar,
            }, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            print(f"[harita] kaydedilemedi: {e}")

    # ------------------------------------------------------------ türev indeksler

    def _ekosistem_belirle(self):
        puan = defaultdict(int)
        for yol in self.dosyalar:
            for ad, kalip, p in EKOSISTEM_IZLERI:
                if kalip.search(yol):
                    puan[ad] += p
        icerik = {"ESP-IDF": r"idf_component_register|IDF_PATH|app_main",
                  "Zephyr": r"find_package\s*\(\s*Zephyr|DEVICE_DT_DEFINE",
                  "STM32 HAL/CubeMX": r"stm32\w+_hal\.h|HAL_Init\s*\(|USER CODE BEGIN",
                  "Arduino": r"void\s+setup\s*\(\s*\)|void\s+loop\s*\(\s*\)"}
        for yol, kayit in list(self.dosyalar.items())[:400]:
            if kayit.get("dil") not in ("c", "cmake"):
                continue
            metin = _metin_oku(self.kok / yol, 20000) or ""
            for ad, kalip in icerik.items():
                if re.search(kalip, metin):
                    puan[ad] += 25
        siralı = sorted(puan.items(), key=lambda kv: -kv[1])
        self.ekosistem = [{"ad": a, "puan": p} for a, p in siralı if p >= 15][:3]

    def _turev(self):
        self.semboller = defaultdict(list)
        self.include = defaultdict(list)
        self.ters_include = defaultdict(list)
        self.cagiranlar = defaultdict(list)
        self.cozulmeyen = defaultdict(list)

        taban = defaultdict(list)
        for yol in self.dosyalar:
            taban[yol.rsplit("/", 1)[-1]].append(yol)

        for yol, kayit in self.dosyalar.items():
            for s in kayit.get("sembol", []):
                self.semboller[s["ad"]].append({
                    "dosya": yol, "tur": s.get("tur", "?"),
                    "satir": s.get("satir", 0),
                    "son_satir": s.get("son_satir", 0),
                    "imza": s.get("imza", "")})
            for inc in kayit.get("include", []):
                hedef = self._include_coz(yol, inc["ad"], taban)
                if hedef:
                    if hedef not in self.include[yol]:
                        self.include[yol].append(hedef)
                    if yol not in self.ters_include[hedef]:
                        self.ters_include[hedef].append(yol)
                else:
                    self.cozulmeyen[yol].append(inc["ad"])
            for cagiran, cagrilanlar in (kayit.get("cagri") or {}).items():
                for c in cagrilanlar:
                    yer = f"{yol}::{cagiran}"
                    if yer not in self.cagiranlar[c]:
                        self.cagiranlar[c].append(yer)

    def _include_coz(self, kaynak_yol, ad, taban):
        ad = ad.replace("\\", "/")
        klasor = kaynak_yol.rsplit("/", 1)[0] if "/" in kaynak_yol else ""
        if klasor:
            aday = os.path.normpath(f"{klasor}/{ad}").replace(os.sep, "/")
            if aday in self.dosyalar:
                return aday
        if ad in self.dosyalar:
            return ad
        vurus = taban.get(ad.rsplit("/", 1)[-1], [])
        if len(vurus) == 1:
            return vurus[0]
        if vurus:
            son_ek = [v for v in vurus if v.endswith("/" + ad)]
            if len(son_ek) == 1:
                return son_ek[0]
            ust = kaynak_yol.split("/")[0]
            ayni = [v for v in (son_ek or vurus) if v.split("/")[0] == ust]
            if len(ayni) == 1:
                return ayni[0]
            return (son_ek or vurus)[0]
        return None

    # ------------------------------------------------------------ sorgular

    def _dosya_bul(self, hedef):
        hedef = str(hedef).strip().replace("\\", "/").lstrip("./")
        if hedef in self.dosyalar:
            return hedef
        vurus = [d for d in self.dosyalar if d.endswith("/" + hedef)]
        if len(vurus) == 1:
            return vurus[0]
        taban_vurus = [d for d in self.dosyalar if d.rsplit("/", 1)[-1] == hedef]
        if len(taban_vurus) == 1:
            return taban_vurus[0]
        return vurus[0] if vurus else (taban_vurus[0] if taban_vurus else None)

    def birimler(self):
        gruplar = defaultdict(list)
        for yol in self.dosyalar:
            p = yol.split("/")
            if p[0] == "components" and len(p) > 2:
                gruplar["components/" + p[1]].append(yol)
            elif len(p) == 1:
                gruplar["<kok>"].append(yol)
            else:
                gruplar[p[0]].append(yol)
        return dict(sorted(gruplar.items()))

    def giris_noktalari(self):
        aranan = ("app_main", "main", "setup", "loop", "Reset_Handler", "SystemInit")
        cikti = []
        for ad in aranan:
            for d in self.semboller.get(ad, []):
                if d["tur"] == "fonksiyon":
                    cikti.append(f"{ad}() -> {d['dosya']}:{d['satir']}")
        return cikti

    def dongular(self, limit=6):
        bulunan, renk, yigin = [], {}, []

        def gez(dugum):
            if len(bulunan) >= limit:
                return
            renk[dugum] = 1
            yigin.append(dugum)
            for sonraki in self.include.get(dugum, []):
                if len(bulunan) >= limit:
                    break
                r = renk.get(sonraki, 0)
                if r == 0:
                    gez(sonraki)
                elif r == 1 and sonraki in yigin:
                    bulunan.append(yigin[yigin.index(sonraki):] + [sonraki])
            yigin.pop()
            renk[dugum] = 2

        sys.setrecursionlimit(10000)
        for d in list(self.include):
            if renk.get(d, 0) == 0:
                gez(d)
        return bulunan

    def katman_ihlalleri(self):
        cikti = []
        for kaynak, hedefler in self.include.items():
            kk = _katman(kaynak)
            if not kk:
                continue
            for h in hedefler:
                hk = _katman(h)
                if hk and hk[1] > kk[1]:
                    cikti.append(f"{kaynak} -> {h}  ({kk[0]} katmani {hk[0]} "
                                 f"katmanina bagimli; bagimlilik asagi dogru olmali)")
        return cikti

    def sahipsizler(self):
        adlandirilan = set()
        for kayit in self.dosyalar.values():
            for s in kayit.get("kaynaklar", []):
                adlandirilan.add(s.rsplit("/", 1)[-1])
        cikti = []
        for yol, kayit in self.dosyalar.items():
            if kayit.get("dil") != "c":
                continue
            taban = yol.rsplit("/", 1)[-1]
            if yol.endswith((".h", ".hpp", ".hh")):
                if not self.ters_include.get(yol):
                    cikti.append(yol)
                continue
            if taban not in adlandirilan and not self.ters_include.get(yol):
                giris = any(d["dosya"] == yol for ad in ("app_main", "main", "setup")
                            for d in self.semboller.get(ad, []))
                if not giris:
                    cikti.append(yol)
        return sorted(cikti)

    # ------------------------------------------------------------ metin çıktılar

    def ozet(self):
        satirlar = []
        eko = ", ".join(f"{e['ad']} ({e['puan']})" for e in self.ekosistem) or "belirsiz"
        toplam_satir = sum(k.get("satir", 0) or 0 for k in self.dosyalar.values())
        fonk = sum(1 for tanimlar in self.semboller.values()
                   for t in tanimlar if t["tur"] == "fonksiyon")
        satirlar += [
            f"PROJE HARITASI  kok={self.kok.name}",
            f"Ekosistem: {eko}",
            f"{len(self.dosyalar)} dosya, {toplam_satir} satir, {fonk} fonksiyon, "
            f"{sum(len(v) for v in self.include.values())} include baglantisi",
            "", "BIRIMLER (dosya / satir):"]
        for birim, dosyalar in self.birimler().items():
            ls = sum(self.dosyalar[d].get("satir", 0) or 0 for d in dosyalar)
            satirlar.append(f"  {birim:<28} {len(dosyalar):>4} dosya {ls:>7} satir")

        gn = self.giris_noktalari()
        if gn:
            satirlar += ["", "GIRIS NOKTALARI:"] + [f"  {g}" for g in gn]

        hub = sorted(((d, len(v)) for d, v in self.ters_include.items()),
                     key=lambda kv: -kv[1])[:10]
        if hub:
            satirlar += ["", "EN COK BAGIMLI OLUNAN DOSYALAR (dikkatli degistir):"]
            satirlar += [f"  {n:>3} bagimli  {d}" for d, n in hub]

        dng = self.dongular()
        if dng:
            satirlar += ["", "INCLUDE DONGULERI:"] + ["  " + " -> ".join(c) for c in dng]

        ihl = self.katman_ihlalleri()
        if ihl:
            satirlar += ["", f"KATMAN IHLALLERI ({len(ihl)}):"] + [f"  {i}" for i in ihl[:12]]

        isaretler = [(y, i) for y, k in self.dosyalar.items()
                     for i in k.get("isaret", [])]
        if isaretler:
            satirlar += ["", f"KOD UYARILARI ({len(isaretler)}):"]
            satirlar += [f"  {y}: {i}" for y, i in isaretler[:20]]

        coz = sorted({a for v in self.cozulmeyen.values() for a in v})
        if coz:
            satirlar += ["", f"AGAC DISI HEADER'LAR (SDK/toolchain, {len(coz)} adet): "
                             + ", ".join(coz[:15])]

        sah = self.sahipsizler()
        if sah:
            satirlar += ["", f"HICBIR YERDE KULLANILMAYAN DOSYALAR ({len(sah)}): "
                             + ", ".join(sah[:15])]
        return "\n".join(satirlar)

    def dosya_karti(self, yol):
        d = self._dosya_bul(yol)
        if not d:
            return f"HATA: '{yol}' haritada yok. Once harita_ozet calistir."
        k = self.dosyalar[d]
        s = [f"DOSYA {d}  ({k.get('dil','?')}, {k.get('satir',0)} satir, "
             f"{k.get('boyut',0)} byte)"]
        inc = self.include.get(d, [])
        if inc:
            s.append(f"include ettikleri ({len(inc)}): " + ", ".join(inc[:25]))
        dis = [i["ad"] for i in k.get("include", [])
               if i["ad"] in self.cozulmeyen.get(d, [])]
        if dis:
            s.append("agac disi header: " + ", ".join(sorted(set(dis))[:20]))
        ters = self.ters_include.get(d, [])
        if ters:
            s.append(f"bunu include edenler ({len(ters)}): " + ", ".join(ters[:25]))
        semboller = k.get("sembol", [])
        if semboller:
            s.append(f"semboller ({len(semboller)}):")
            for sem in semboller[:70]:
                aralik = f":{sem['satir']}" + (f"-{sem['son_satir']}"
                                               if sem.get("son_satir") else "")
                dis_cagri = len([c for c in self.cagiranlar.get(sem["ad"], [])
                                 if not c.startswith(d + "::")])
                s.append(f"  {sem.get('tur','?'):<10}{aralik:<12} "
                         f"{(sem.get('imza') or sem['ad'])[:110]}"
                         + (f"   [{dis_cagri} dis cagri]" if dis_cagri else ""))
        cagri = k.get("cagri") or {}
        if cagri:
            s.append("dosya ici cagri grafigi:")
            for cagiran, cagrilanlar in list(cagri.items())[:25]:
                s.append(f"  {cagiran} -> " + ", ".join(cagrilanlar[:14]))
        if k.get("isaret"):
            s.append("uyarilar: " + "; ".join(k["isaret"]))
        if k.get("gerektirir"):
            s.append("build bagimliliklari: " + ", ".join(k["gerektirir"]))
        if k.get("kaynaklar"):
            s.append("build kaynaklari: " + ", ".join(k["kaynaklar"][:20]))
        return "\n".join(s)

    def _gecisli_bagimlilar(self, yol, derinlik=6):
        gorulen = set()
        kuyruk = deque((d, 1) for d in self.ters_include.get(yol, []))
        while kuyruk:
            dugum, d = kuyruk.popleft()
            if dugum in gorulen or d > derinlik:
                continue
            gorulen.add(dugum)
            for s in self.ters_include.get(dugum, []):
                if s not in gorulen:
                    kuyruk.append((s, d + 1))
        gorulen.discard(yol)
        return sorted(gorulen)

    def etki(self, hedef):
        dosya = self._dosya_bul(hedef)

        if dosya is None:
            tanimlar = self.semboller.get(hedef) or []
            if not tanimlar:
                dusuk = hedef.lower()
                for ad, t in self.semboller.items():
                    if dusuk in ad.lower():
                        tanimlar.extend({**x, "ad": ad} for x in t)
                tanimlar = tanimlar[:20]
            if not tanimlar:
                return f"HATA: '{hedef}' ne dosya ne sembol olarak haritada var."
            cagri_yerleri = self.cagiranlar.get(hedef, [])
            cagiran_dosyalar = sorted({c.split("::")[0] for c in cagri_yerleri})
            ev = sorted({t["dosya"] for t in tanimlar})
            asagi = set()
            for e in ev:
                asagi |= set(self._gecisli_bagimlilar(e))
            s = [f"ETKI ANALIZI — '{hedef}' (sembol)",
                 f"risk: {self._risk(len(cagri_yerleri), len(asagi))}",
                 f"Tanimli oldugu dosyalar: {', '.join(ev)}",
                 f"Cagrildigi yer sayisi: {len(cagri_yerleri)} "
                 f"({len(cagiran_dosyalar)} dosyada)"]
            for c in cagri_yerleri[:30]:
                s.append(f"  {c}")
            if asagi:
                s.append(f"Header'inin altinda kalan dosyalar ({len(asagi)}): "
                         + ", ".join(sorted(asagi)[:25]))
            return "\n".join(s)

        dogrudan = sorted(self.ters_include.get(dosya, []))
        gecisli = [g for g in self._gecisli_bagimlilar(dosya) if g not in dogrudan]
        kayit = self.dosyalar.get(dosya, {})
        disa_acik = []
        for sem in kayit.get("sembol", []):
            if sem.get("tur") not in ("fonksiyon", "prototip", "makro", "tip"):
                continue
            yerler = [c for c in self.cagiranlar.get(sem["ad"], [])
                      if not c.startswith(dosya + "::")]
            if yerler:
                disa_acik.append((sem["ad"], sem.get("satir", 0), yerler))
        disa_acik.sort(key=lambda x: -len(x[2]))

        etkilenen = sorted({b for b, dosyalar in self.birimler().items()
                            if any(d in set(dogrudan) | set(gecisli) for d in dosyalar)})
        s = [f"ETKI ANALIZI — {dosya} (dosya)",
             f"risk: {self._risk(sum(len(x[2]) for x in disa_acik), len(dogrudan) + len(gecisli))}",
             f"Dogrudan bagimli ({len(dogrudan)}): " + (", ".join(dogrudan) or "yok"),
             f"Gecisli bagimli ({len(gecisli)}): " + (", ".join(gecisli[:25]) or "yok"),
             f"Etkilenen birimler: " + (", ".join(etkilenen) or "yok"),
             f"Bu dosyanin bagimli oldugu: " + (", ".join(self.include.get(dosya, [])) or "yok")]
        if disa_acik:
            s.append("Disaridan kullanilan semboller:")
            for ad, satir, yerler in disa_acik[:20]:
                s.append(f"  {ad} (satir {satir}) — {len(yerler)} dis cagri: "
                         + ", ".join(yerler[:8]))
        return "\n".join(s)

    @staticmethod
    def _risk(cagri, bagimli):
        puan = cagri + 2 * bagimli
        return "YUKSEK" if puan >= 40 else ("ORTA" if puan >= 12 else "dusuk")

    def sembol_ara(self, ad):
        tanimlar = self.semboller.get(ad)
        if not tanimlar:
            dusuk = ad.lower()
            tanimlar = []
            for s, t in self.semboller.items():
                if dusuk in s.lower():
                    tanimlar.extend({**x, "ad": s} for x in t)
            tanimlar = tanimlar[:30]
        if not tanimlar:
            return f"'{ad}' haritada bulunamadi."
        s = [f"'{ad}' icin {len(tanimlar)} sonuc:"]
        for t in tanimlar[:30]:
            aralik = f":{t['satir']}" + (f"-{t['son_satir']}" if t.get("son_satir") else "")
            s.append(f"  {t.get('tur','?'):<10} {t['dosya']}{aralik}  "
                     f"{(t.get('imza') or '')[:100]}")
        cagri = self.cagiranlar.get(ad, [])
        if cagri:
            s.append(f"cagrildigi {len(cagri)} yer: " + ", ".join(cagri[:20]))
        return "\n".join(s)

    def agac(self, limit=500):
        klasorler = defaultdict(list)
        for yol in sorted(self.dosyalar):
            k = yol.rsplit("/", 1)[0] if "/" in yol else "."
            klasorler[k].append(yol.rsplit("/", 1)[-1])
        cikti, sayac = [], 0
        for k in sorted(klasorler):
            cikti.append(f"{k}/")
            for ad in klasorler[k]:
                cikti.append(f"    {ad}")
                sayac += 1
                if sayac >= limit:
                    cikti.append(f"    ... ({len(self.dosyalar) - sayac} tane daha)")
                    return "\n".join(cikti)
        return "\n".join(cikti)


# ===========================================================================
# BÖLÜM 3 — ARAÇLAR
# ===========================================================================

DURUM = {"harita": None, "proje": None,
         "harita_sorgusu": 0, "dosya_okuma": 0, "yazma": []}


def _harita():
    if DURUM["harita"] is None:
        # Ajan bir proje belirtmeden harita istedi: o anki klasörü kullan.
        DURUM["proje"] = Path.cwd()
        DURUM["harita"] = ProjeHaritasi.yukle_veya_olustur(DURUM["proje"])
    return DURUM["harita"]


def _gorece(yol):
    """Mutlak ya da göreceli yolu, harita anahtarına çevirir."""
    h = _harita()
    p = Path(yol)
    if not p.is_absolute():
        p = h.kok / yol
    try:
        return str(p.resolve().relative_to(h.kok)).replace(os.sep, "/")
    except ValueError:
        return None      # proje dışında; harita güncellemesi gerekmez


def arac_harita_ozet(args):
    return _harita().ozet()


def arac_harita_agac(args):
    return _harita().agac()


def arac_dosya_bilgi(args):
    return _harita().dosya_karti(args.get("path", ""))


def arac_etki_analizi(args):
    return _harita().etki(args.get("target", args.get("hedef", "")))


def arac_sembol_bul(args):
    return _harita().sembol_ara(args.get("name", args.get("ad", "")))


def arac_kim_include_ediyor(args):
    h = _harita()
    d = h._dosya_bul(args.get("path", ""))
    if not d:
        return f"HATA: '{args.get('path')}' haritada yok."
    ters = h.ters_include.get(d, [])
    return (f"{d} dosyasini {len(ters)} dosya include ediyor:\n  "
            + "\n  ".join(ters)) if ters else f"{d} dosyasini kimse include etmiyor."


def arac_proje_sorunlari(args):
    h = _harita()
    s = []
    dng = h.dongular(10)
    if dng:
        s.append("INCLUDE DONGULERI:")
        s += ["  " + " -> ".join(c) for c in dng]
    ihl = h.katman_ihlalleri()
    if ihl:
        s.append(f"KATMAN IHLALLERI ({len(ihl)}):")
        s += [f"  {i}" for i in ihl[:25]]
    sah = h.sahipsizler()
    if sah:
        s.append(f"KULLANILMAYAN DOSYALAR ({len(sah)}):")
        s += [f"  {o}" for o in sah[:30]]
    uyari = [f"  {y}: {i}" for y, k in h.dosyalar.items() for i in k.get("isaret", [])]
    if uyari:
        s.append(f"KOD UYARILARI ({len(uyari)}):")
        s += uyari[:40]
    return "\n".join(s) or "Haritada yapisal bir sorun gorunmuyor."


def arac_ara(args):
    """Sadece haritadaki dosyalarda regex araması. Diski baştan gezmez."""
    h = _harita()
    kalip = args.get("pattern", args.get("kalip", ""))
    suzgec = args.get("glob", "*")
    try:
        rx = re.compile(kalip)
    except re.error as e:
        return f"HATA: hatali regex: {e}"
    vurus, bakilan = [], 0
    for yol, kayit in h.dosyalar.items():
        if not kayit.get("cozuldu"):
            continue
        if suzgec != "*" and not Path(yol).match(suzgec):
            continue
        metin = _metin_oku(h.kok / yol)
        if metin is None:
            continue
        bakilan += 1
        for i, satir in enumerate(metin.splitlines(), 1):
            if rx.search(satir):
                vurus.append(f"{yol}:{i}: {satir.strip()[:160]}")
                if len(vurus) >= 150:
                    break
        if len(vurus) >= 150:
            break
    return (f"{len(vurus)} sonuc ({bakilan} dosyada):\n" + "\n".join(vurus)
            if vurus else f"'{kalip}' icin sonuc yok ({bakilan} dosya tarandi).")


def arac_fonksiyon_oku(args):
    """Haritadaki satır aralığını kullanarak sadece bir fonksiyonu okur."""
    h = _harita()
    ad = args.get("name", args.get("ad", ""))
    dosya_suzgec = args.get("file", "")
    tanimlar = [t for t in h.semboller.get(ad, [])
                if t["tur"] == "fonksiyon"
                and (not dosya_suzgec or t["dosya"].endswith(dosya_suzgec))]
    if not tanimlar:
        return (f"'{ad}' icin fonksiyon tanimi haritada yok "
                "(makro, prototip ya da SDK sembolu olabilir).")
    cikti = []
    for t in tanimlar[:3]:
        DURUM["dosya_okuma"] += 1
        satirlar = _metin_oku(h.kok / t["dosya"]).splitlines()
        bas = max(1, t["satir"])
        son = min(len(satirlar), t.get("son_satir") or (bas + 80))
        cikti.append(f"--- {t['dosya']}:{bas}-{son} ---")
        cikti += [f"{i:>5}  {satirlar[i-1]}" for i in range(bas, son + 1)]
    return "\n".join(cikti)


def arac_read_file(args):
    """Son çare. Onay sormaz."""
    yol = args.get("path", "")
    bas = int(args.get("start_line", 1) or 1)
    son = int(args.get("end_line", 0) or 0)
    try:
        p = Path(yol)
        if not p.is_absolute() and DURUM["proje"]:
            p = DURUM["proje"] / yol
        satirlar = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        return f"HATA: {e}"
    DURUM["dosya_okuma"] += 1
    son = min(son or len(satirlar), len(satirlar))
    bas = max(1, bas)
    govde = "\n".join(f"{i:>5}  {satirlar[i-1]}" for i in range(bas, son + 1))
    if len(govde) > 24000:
        govde = govde[:24000] + "\n...[kesildi — dar bir satir araligi iste]"
    return f"{yol} satir {bas}-{son} / {len(satirlar)}\n{govde}"


def arac_write_file(args):
    """Onay sormaz. Yazdıktan sonra haritayı o dosya için günceller."""
    yol = args.get("path", "")
    icerik = args.get("content", "")
    try:
        p = Path(yol)
        if not p.is_absolute() and DURUM["proje"]:
            p = DURUM["proje"] / yol
        p.parent.mkdir(parents=True, exist_ok=True)
        vardi = p.exists()
        p.write_text(icerik, encoding="utf-8")
    except Exception as e:
        return f"HATA: {e}"
    DURUM["yazma"].append(str(p))
    g = _gorece(p)
    if g and DURUM["harita"]:
        DURUM["harita"].dosya_tazele(g)
    return (f"{'Uzerine yazildi' if vardi else 'Olusturuldu'}: {p} "
            f"({len(icerik)} karakter). Harita guncellendi.")


def arac_edit_file(args):
    """Dosyanın içinde tek ve benzersiz bir metin bloğunu değiştirir."""
    yol = args.get("path", "")
    bul = args.get("find", "")
    koy = args.get("replace", "")
    try:
        p = Path(yol)
        if not p.is_absolute() and DURUM["proje"]:
            p = DURUM["proje"] / yol
        metin = p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return f"HATA: {e}"
    adet = metin.count(bul)
    if adet == 0:
        return ("HATA: aranan metin dosyada yok. Once o bolgeyi read_file ile "
                "oku ve birebir kopyala.")
    if adet > 1:
        return (f"HATA: aranan metin {adet} kez geciyor. Benzersiz olmasi icin "
                "cevresinden birkac satir daha ekle.")
    p.write_text(metin.replace(bul, koy), encoding="utf-8")
    DURUM["yazma"].append(str(p))
    g = _gorece(p)
    if g and DURUM["harita"]:
        DURUM["harita"].dosya_tazele(g)
    return f"Duzenlendi: {p}. Harita guncellendi."


def arac_tasi(args):
    kaynak = args.get("source", "")
    hedef = args.get("destination", "")
    try:
        ps = Path(kaynak)
        ph = Path(hedef)
        if not ps.is_absolute() and DURUM["proje"]:
            ps = DURUM["proje"] / kaynak
        if not ph.is_absolute() and DURUM["proje"]:
            ph = DURUM["proje"] / hedef
        if ph.exists():
            return f"HATA: {ph} zaten var."
        ph.parent.mkdir(parents=True, exist_ok=True)
        ps.rename(ph)
    except Exception as e:
        return f"HATA: {e}"
    h = DURUM["harita"]
    g_eski, g_yeni = _gorece(ps), _gorece(ph)
    not_ = ""
    if h and g_eski:
        bagimli = h.ters_include.get(g_eski, [])
        if bagimli:
            not_ = ("\nDIKKAT: bu dosyayi su dosyalar include ediyor, "
                    "#include yollarini duzeltmen gerekebilir: "
                    + ", ".join(bagimli[:20]))
        h.dosya_tazele(g_eski)
    if h and g_yeni:
        h.dosya_tazele(g_yeni)
    return f"Tasindi: {ps} -> {ph}. Harita guncellendi.{not_}"


def arac_run_shell(args):
    komut = args.get("command", "")
    if SHELL_ONAY_SOR:
        print(f"\n  >> Ajan su komutu calistirmak istiyor: {komut}")
        if input("  >> Onayliyor musun? (e/h): ").strip().lower() != "e":
            return "Kullanici komutu reddetti."
    try:
        sonuc = subprocess.run(komut, shell=True, capture_output=True,
                               text=True, timeout=180,
                               cwd=str(DURUM["proje"]) if DURUM["proje"] else None)
        cikti = sonuc.stdout + sonuc.stderr
    except Exception as e:
        return f"HATA: {e}"
    if len(cikti) > 6000:
        cikti = cikti[:1500] + "\n...[kesildi]...\n" + cikti[-4000:]
    return f"exit={sonuc.returncode}\n{cikti or '(cikti yok)'}"


def arac_web_search(args):
    sorgu = args.get("query", "")
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            sonuclar = list(ddgs.text(sorgu, max_results=5))
    except ImportError:
        return "HATA: 'duckduckgo-search' kurulu degil: pip install duckduckgo-search"
    except Exception as e:
        return f"HATA: {e}"
    if not sonuclar:
        return "Sonuc bulunamadi."
    return "\n".join(f"- {r.get('title','')}: {r.get('body','')}\n  ({r.get('href','')})"
                     for r in sonuclar)


# Sıra önemli: prompt'ta bu sırayla listeleniyor, ucuz olanlar üstte.
ARACLAR = {
    "harita_ozet": arac_harita_ozet,
    "dosya_bilgi": arac_dosya_bilgi,
    "etki_analizi": arac_etki_analizi,
    "sembol_bul": arac_sembol_bul,
    "kim_include_ediyor": arac_kim_include_ediyor,
    "proje_sorunlari": arac_proje_sorunlari,
    "harita_agac": arac_harita_agac,
    "ara": arac_ara,
    "fonksiyon_oku": arac_fonksiyon_oku,
    "read_file": arac_read_file,
    "write_file": arac_write_file,
    "edit_file": arac_edit_file,
    "tasi": arac_tasi,
    "run_shell": arac_run_shell,
    "web_search": arac_web_search,
}

ARAC_ACIKLAMA = """\
HARITA ARACLARI (bunlar bedava — dosya acmazlar, once bunlari kullan):
- harita_ozet: Projenin tamamini ozetler: ekosistem, birimler, giris noktalari,
  en cok bagimli olunan dosyalar, include dongulari, katman ihlalleri, kod
  uyarilari. Girdi: {}
- dosya_bilgi: Bir dosyayi ACMADAN hakkindaki her seyi verir: include ettikleri,
  onu include edenler, tum sembolleri satir araliklariyla, dosya ici cagri
  grafigi. Girdi: {"path": "src/main.c"}
- etki_analizi: Bir dosyayi ya da sembolu degistirirsen ne kirilir. Degisiklik
  onermeden ya da yapmadan ONCE bunu calistir. Girdi: {"target": "led.h"}
- sembol_bul: Bir fonksiyon/makro/tip nerede tanimli, nerelerden cagriliyor.
  Girdi: {"name": "uart_init"}
- kim_include_ediyor: Bir header'i kimler include ediyor. Girdi: {"path": "led.h"}
- proje_sorunlari: Haritanin buldugu yapisal sorunlar. Girdi: {}
- harita_agac: Klasor/dosya listesi. Girdi: {}
- ara: Haritadaki dosyalarda regex aramasi. Girdi: {"pattern": "HAL_Init", "glob": "*.c"}

OKUMA ARACLARI:
- fonksiyon_oku: Haritayi kullanarak SADECE bir fonksiyonun govdesini okur.
  Tek fonksiyon yetiyorsa read_file yerine bunu kullan. Girdi: {"name": "app_main"}
- read_file: Dosyanin ham metnini okur. SON CARE. Buyuk dosyada satir araligi ver.
  Girdi: {"path": "src/main.c", "start_line": 1, "end_line": 120}

YAZMA ARACLARI:
- write_file: Dosya olusturur ya da uzerine yazar. Girdi: {"path": "...", "content": "..."}
- edit_file: Dosyadaki tek ve benzersiz bir metin blogunu degistirir. Tum dosyayi
  yeniden yazmaktan daha guvenli. Girdi: {"path": "...", "find": "...", "replace": "..."}
- tasi: Dosya tasir/yeniden adlandirir; onu include eden dosyalari uyarir.
  Girdi: {"source": "led.c", "destination": "src/led.c"}

DIGER:
- run_shell: Kabuk komutu calistirir. Girdi: {"command": "make"}
- web_search: Internette arama. Girdi: {"query": "..."}
"""


# ===========================================================================
# BÖLÜM 4 — SISTEM PROMPT'U
# ===========================================================================

SYSTEM_PROMPT = """Sen gomulu sistemler (embedded) konusunda uzman bir yazilim
muhendisisin. C, C++, RTOS, bellek kisitli sistemler ve ESP-IDF, Zephyr, STM32
HAL/CubeMX, PlatformIO, Arduino, bare-metal CMake/Make build sistemlerini
bilirsin.

CALISMA SIRAN — her seferinde buna uy:
1. Ise HER ZAMAN harita_ozet ile basla. Proje haritasi onbellekte hazir duruyor;
   dosyalari tek tek okuyarak baslama.
2. Yapisal sorulari dosya_bilgi, sembol_bul, kim_include_ediyor ve ara ile
   cevapla. Bunlar dosya acmaz, bedavadir.
3. Bir seyi degistirmeyi onermeden ya da degistirmeden ONCE etki_analizi
   calistir. Neyin neye bagli oldugunu bilmeden dosyaya dokunma.
4. Bir dosyanin ham metnine gercekten ihtiyacin varsa oku. Tek fonksiyon
   yetiyorsa fonksiyon_oku kullan, read_file degil.
5. Degisikligi yap. Mumkunse run_shell ile derleyip dogrula.
6. FINAL ANSWER ile duz Turkce ozet ver.

MUHENDISLIK OLCUTLERIN:
- Bagimlilik asagi dogru akar: app -> services -> drivers -> HAL -> vendor.
  Bir surucunun uygulama header'ini include etmesi hatadir.
- Bir modulun public API'si header'idir. Baska bir module ../ ile uzanmak hatadir.
- ISR kisadir. Bayrak set eder ya da kuyruga atar. Kesme icinde printf, malloc,
  bloklayici delay, uzun dongu olmaz.
- ISR ile ana baglam arasinda paylasilan degisken volatile olur ve erisim
  atomik ya da kritik bolge icinde yapilir.
- Init sonrasi dinamik bellek ayirmak kisitli sistemde koku verir, soyle.
- Donanim tanimi konfigurasyona (devicetree, Kconfig, board dosyalari) aittir;
  C icine serpistirilmis #ifdef merdivenlerine degil.
- Uretilen kod (CubeMX Core/, sdkconfig) elle duzenlenecek yer degildir.

KURALLAR:
- Haritada olmayan bir dosyayi, sembolu ya da API'yi UYDURMA. Emin degilsen
  sembol_bul ile bak.
- Yapmadigin bir degisikligi yaptim deme.
- Bir arac hata dondururse oku ve yaklasimini degistir; ayni cagriyi tekrarlama.
- Goremedigin donanim hakkinda tahmin yurutme, "bilmiyorum" de.

""" + ARAC_ACIKLAMA + """
ARAC KULLANIM FORMATI — bir arac kullanacaksan TAM OLARAK boyle cevap ver ve
baska hicbir sey yazma:

ACTION: <arac_adi>
ACTION_INPUT: <JSON girdi>

Bir arac sonucu (OBSERVATION) aldiktan sonra ya baska bir ACTION calistir ya da
isin bittiyse:

FINAL ANSWER: <kullaniciya verilecek cevap>

Tek seferde tek arac. Cevabini zaten biliyorsan bosuna arac kullanma.
"""


# ===========================================================================
# BÖLÜM 5 — AJAN DÖNGÜSÜ
# ===========================================================================

def modeli_cagir(mesajlar):
    try:
        yanit = requests.post(
            LLAMA_SERVER_URL,
            json={"model": MODEL_ADI, "messages": mesajlar,
                  "temperature": SICAKLIK, "max_tokens": MAX_TOKENS},
            timeout=600,
        )
    except requests.exceptions.RequestException as e:
        # DÜZELTME: v1'de sunucuya ulasilamayinca ham istisna patliyordu.
        raise RuntimeError(
            f"llama.cpp sunucusuna ulasilamiyor ({LLAMA_SERVER_URL}): {e}\n"
            "Docker kullaniyorsan sunucuyu --host 0.0.0.0 ile baslat ve portu "
            "disari ac (-p 8080:8080).") from e
    try:
        veri = yanit.json()
    except ValueError:
        raise RuntimeError(f"Sunucu JSON dondurmedi (HTTP {yanit.status_code}): "
                           f"{yanit.text[:400]}")
    # DÜZELTME: v1'de dogrudan ["choices"][0] okunuyordu; sunucu hata
    # dondurdugunde KeyError ile cokuyordu.
    if "choices" not in veri:
        raise RuntimeError(f"Sunucu hatasi: {json.dumps(veri)[:400]}")
    return veri["choices"][0]["message"].get("content") or ""


# DÜZELTME: ACTION_INPUT regex'i v1'de acgozluydu (\{.*\}) ve arkasindaki
# metni de yutabiliyordu. Artik dengeli parantez sayarak kesiyoruz.
def _ilk_json(metin):
    bas = metin.find("{")
    if bas == -1:
        return None
    derinlik, string_ici, kacis = 0, False, False
    for i in range(bas, len(metin)):
        c = metin[i]
        if string_ici:
            if kacis:
                kacis = False
            elif c == "\\":
                kacis = True
            elif c == '"':
                string_ici = False
            continue
        if c == '"':
            string_ici = True
        elif c == "{":
            derinlik += 1
        elif c == "}":
            derinlik -= 1
            if derinlik == 0:
                try:
                    return json.loads(metin[bas:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def eylem_coz(metin):
    m = re.search(r"ACTION:\s*([A-Za-z_]\w*)", metin)
    if not m:
        return None, None
    arac = m.group(1).strip()
    girdi_bolumu = metin[metin.find("ACTION_INPUT"):] if "ACTION_INPUT" in metin else ""
    girdi = _ilk_json(girdi_bolumu) or {}
    return arac, girdi


def nihai_cevap_coz(metin):
    m = re.search(r"FINAL ANSWER:\s*(.*)", metin, re.DOTALL)
    return m.group(1).strip() if m else None


def _mesajlari_buda(mesajlar, tut=12):
    """Sistem promptu + son N mesaj. Uzun oturumda context tasmasin diye."""
    if len(mesajlar) <= tut + 1:
        return mesajlar
    return [mesajlar[0]] + mesajlar[-tut:]


def ajani_calistir(soru):
    mesajlar = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": soru}]
    DURUM["harita_sorgusu"] = 0
    DURUM["dosya_okuma"] = 0
    DURUM["yazma"] = []
    son_metin = ""

    for adim in range(MAX_ITERATIONS):
        print(f"\n--- Adim {adim + 1} ---")
        try:
            cevap = modeli_cagir(_mesajlari_buda(mesajlar))
        except RuntimeError as e:
            return f"HATA: {e}"
        print(cevap)
        son_metin = cevap

        # DÜZELTME: v1'de once FINAL ANSWER aranıyordu. Model plan metninde
        # "FINAL ANSWER" gecirirse dongu erken kopuyordu. Once ACTION'a bak.
        arac, girdi = eylem_coz(cevap)
        if arac and arac in ARACLAR:
            if arac.startswith(("harita", "dosya_bilgi", "etki", "sembol",
                                "kim_include", "proje_sorunlari", "ara")):
                DURUM["harita_sorgusu"] += 1
            gozlem = ARACLAR[arac](girdi)
            if len(gozlem) > 20000:
                gozlem = gozlem[:20000] + "\n...[gozlem kesildi]"
            print(f"\n  [OBSERVATION] ({arac}): {gozlem[:600]}"
                  + ("..." if len(gozlem) > 600 else ""))
            mesajlar.append({"role": "assistant", "content": cevap})
            mesajlar.append({"role": "user",
                             "content": f"OBSERVATION: {gozlem}"})
            continue

        if arac and arac not in ARACLAR:
            mesajlar.append({"role": "assistant", "content": cevap})
            mesajlar.append({"role": "user", "content":
                             f"OBSERVATION: HATA: '{arac}' diye bir arac yok. "
                             f"Kullanilabilir araclar: {', '.join(ARACLAR)}"})
            continue

        nihai = nihai_cevap_coz(cevap)
        if nihai:
            return nihai
        return cevap        # ne ACTION ne FINAL ANSWER — duz cevap saymis

    return (son_metin + "\n\n[Maksimum adim sayisina ulasildi.]").strip()


# ===========================================================================
# BÖLÜM 6 — REPL
# ===========================================================================

def proje_ac(yol):
    p = Path(yol).expanduser().resolve()
    if not p.is_dir():
        print(f"HATA: {p} bir klasor degil.")
        return
    DURUM["proje"] = p
    DURUM["harita"] = ProjeHaritasi.yukle_veya_olustur(p)
    print(f"[proje] {p}")


def main():
    print("Local AI Agent — harita tabanli surum")
    print(f"Model sunucusu: {LLAMA_SERVER_URL}")

    hedef = sys.argv[1] if len(sys.argv) > 1 else "."
    proje_ac(hedef)

    print("\nKomutlar: :proje <yol>  :harita  :yenile  :araclar  exit\n")
    while True:
        try:
            soru = input("\nSoru: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not soru:
            continue
        if soru.lower() in ("exit", "quit", "çık", "cik"):
            break
        if soru.startswith(":proje"):
            parca = soru.split(None, 1)
            if len(parca) > 1:
                proje_ac(parca[1].strip())
            else:
                print(f"[proje] {DURUM['proje']}")
            continue
        if soru == ":harita":
            print(_harita().ozet())
            continue
        if soru == ":yenile":
            DURUM["harita"] = ProjeHaritasi.yukle_veya_olustur(
                DURUM["proje"], zorla=True)
            continue
        if soru == ":araclar":
            print(ARAC_ACIKLAMA)
            continue

        cevap = ajani_calistir(soru)
        print(f"\n=== CEVAP ===\n{cevap}\n")
        print(f"[harita sorgusu: {DURUM['harita_sorgusu']} | "
              f"acilan dosya: {DURUM['dosya_okuma']} | "
              f"yazilan: {len(DURUM['yazma'])}]")


if __name__ == "__main__":
    main()
