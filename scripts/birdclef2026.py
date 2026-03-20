"""BirdCLEF 2026 shared constants and CSV helpers."""
import csv
import json
from pathlib import Path

AUDIO_EXTS = {".wav", ".ogg", ".mp3", ".flac"}
SAMPLE_RATE_PERCH  = 32000
SAMPLE_RATE_YAMNET = 16000
CLIP_SECS = 5

# 234 target species — exact order from sample_submission.csv header
SPECIES = [
    "1161364","116570","1176823","1491113","1595929","209233","22930","22956",
    "22961","22967","22973","22983","22985","23150","23154","23158","23176",
    "23724","24279","24285","24287","24321","244024","25073","25092","25214",
    "326272","41970","43435","47144",
    "47158son01","47158son02","47158son03","47158son04","47158son05","47158son06",
    "47158son07","47158son08","47158son09","47158son10","47158son11","47158son12",
    "47158son13","47158son14","47158son15","47158son16","47158son17","47158son18",
    "47158son19","47158son20","47158son21","47158son22","47158son23","47158son24",
    "47158son25","476521","516975","517063","555123","555145","555146","64898",
    "65377","65380","66971","67107","67252","70711","738183","74113","74580","760266",
    "ashgre1","astcra1","bafcur1","baffal1","banana","barant1","batbel1","baymac",
    "bbwduc","bcwfin2","bkcdon","bkhpar","blchaw1","blheag1","blttit1","bncfly",
    "bobfly1","brcmar1","brnowl","bucmot4","bucpar","bufpar","bunibi1","burowl",
    "camfli1","chacha1","chbmoc1","chobla1","chvcon1","cibspi1","coffal1","compau",
    "compot1","crbthr1","crebec1","dwatin1","epaori4","eulfly1","fabwre1","fepowl",
    "ficman1","flawar1","fotfly","fusfly1","gilhum1","giwrai1","glteme1","grasal3",
    "greani1","greant1","greela","grekis","grepot1","gretho2","greyel","grfdov1",
    "grhtan1","gycwor1","horscr1","houspa","hyamac1","larela1","lesela1","lesgrf1",
    "limpki","linwoo1","litcuc2","litnig1","mabpar","magant1","magtan2","masgna1",
    "nacnig1","ocecra1","oliwoo1","orbtro3","orwpar","osprey","pabspi1","palhor3",
    "paltan1","phecuc1","picpig2","pirfly1","plasla1","platyr1","plcjay1","pluibi1",
    "purjay1","pvttyr1","ragmac1","rebscy1","recfin1","redjun","relser1","rinkin1",
    "rivwar1","roahaw","rubthr1","rufcac2","rufcas2","rufgna3","rufhor2","rufnig1",
    "ruftho1","ruftof1","rumfly1","ruther1","rutjac1","sabspa1","saffin","saytan1",
    "scadov1","schpar1","scther1","shcfly1","shshaw","shtnig1","sibtan2","smbani",
    "smbtin1","sobcac1","sobtyr1","socfly1","sofspi1","souant1","soulap1","souscr1",
    "spbant3","spispi1","sptnig1","squcuc1","stbwoo2","strcuc1","strher2","strowl1",
    "swthum1","swtman1","tattin1","thlwre1","toctou1","trokin","trsowl","undtin1",
    "varant1","watjac1","wesfie1","wfwduc1","whbant2","whbwar2","whiwoo1","whlspi1",
    "whnjay1","whtdov","whwpic1","y00678","yebcar","yebela1","yecmac","yecpar",
    "yehcar1","yeofly1",
]
N_SPECIES = len(SPECIES)
UNIFORM_P = 1.0 / N_SPECIES

# Numeric XC IDs that appear in SPECIES (for Perch mapping)
NUMERIC_XC_IDS = {s for s in SPECIES if s.isdigit()}


def collect_audio(p: Path) -> list:
    files = []
    if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
        return [p]
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.suffix.lower() in AUDIO_EXTS:
                files.append(f)
    return files


def write_submission(output_dir: str, rows: list, score: float):
    """Write submission.csv (BirdCLEF 2026) and predictions.json."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "submission.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id"] + SPECIES)
        for row_id, probs in rows:
            w.writerow([row_id] + [f"{p:.18f}" for p in probs])

    (out / "predictions.json").write_text(json.dumps({
        "score":           score,
        "rows":            len(rows),
        "submission_file": "submission.csv",
        "species":         N_SPECIES,
    }, indent=2))


def uniform_probs(np_module):
    return np_module.full(N_SPECIES, UNIFORM_P)


def load_expected_rows(sample_submission_path: str) -> "dict[str, list[int]]":
    """Parse sample_submission.csv → {soundscape_stem: [end_sec, ...]} ordered dict.

    Row-id format: ``{stem}_{end_sec}``  e.g. ``BC2026_Test_0001_S05_20250227_010002_5``.
    Returns stems in the order they first appear in the CSV.
    """
    import collections
    expected: "dict[str, list[int]]" = collections.OrderedDict()
    with open(sample_submission_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            row_id = row[0]
            parts = row_id.rsplit("_", 1)
            if len(parts) != 2:
                continue
            stem, end_str = parts
            try:
                end_sec = int(end_str)
            except ValueError:
                continue
            expected.setdefault(stem, []).append(end_sec)
    return expected
