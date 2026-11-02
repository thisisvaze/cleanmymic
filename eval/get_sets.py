"""Download the public test sets (evaluation only; several are non-commercial).

    uv run eval/get_sets.py ~/data/eval

- daps/      DAPS: studio 'clean' + the same speech recorded on iPad/iPhone in real rooms (44.1 kHz, CC BY-NC 4.0)
- drvctk/    Device-Recorded VCTK test set: replayed VCTK on consumer devices incl. MacBook Air (16 kHz, CC BY 4.0)
- dns5/      DNS Challenge 5 blind test set: real headset/speakerphone recordings, no clean reference; each with a
             clip of its talker alone (enrol/)
- chime6/    CHiME-6 dev+eval: dinner parties in homes, each guest's in-ear binaural mics (16 kHz) and transcripts
             (openslr.org/150, CC BY-SA 4.0; the Kinect arrays are skipped while streaming)
- easycom/   EasyCom: conversations with restaurant noise played at 71 dB, close-talk and AR-glasses mics, transcripts
             (CC BY-NC 4.0; 66 GB of parts, mostly video: only audio, transcripts and metadata are kept)
Safe to re-run: finished downloads and extractions are skipped.
"""
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

DAPS = "https://zenodo.org/api/records/4660670/files/daps.tar.gz/content"
DRVCTK = "https://datashare.ed.ac.uk/server/api/core/bitstreams/cfd8794f-81a5-45b7-9a62-6381e6a7cdb1/content"
DNS5 = "https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_BlindTestSet.zip"
CHIME6 = "https://openslr.elda.org/resources/150/CHiME6_{}.tar.gz"  # openslr.org/150's EU mirror (fastest from Finland)
EASYCOM = "https://github.com/facebookresearch/EasyComDataset/releases/download/v1.0.0/EasyComDataset.tar.gz.{:03d}"


def download(url, path):
    if not path.exists():
        part = path.with_name(path.name + ".part")
        subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-o", str(part), url], check=True)
        part.rename(path)
    return path


def stream(urls, dest, *patterns):
    """The concatenation of urls (one tar.gz, maybe in parts) straight through tar, keeping the members matching
    patterns. tar's exit status is not trusted (CHiME-6's transcripts archive has trailing bytes): the caller checks."""
    dest.mkdir(parents=True, exist_ok=True)
    curl = subprocess.Popen(["curl", "-fsSL", "--retry", "5", *urls], stdout=subprocess.PIPE)
    subprocess.run(["tar", "xzf", "-", "-C", str(dest), "--wildcards", *patterns], stdin=curl.stdout)
    curl.wait()


def extract(archive, dest, keep=lambda name: True):
    if (dest / ".done").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest, [n for n in z.namelist() if keep(n)])
    else:
        with tarfile.open(archive, "r|gz") as t:
            for m in t:
                if keep(m.name):
                    t.extract(m, dest, filter="data")
    (dest / ".done").touch()
    print(f"{dest.name}: extracted", flush=True)


if __name__ == "__main__":
    out = Path(sys.argv[1]).expanduser()
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    extract(download(DRVCTK, raw / "DR-VCTK.zip"), out / "drvctk", lambda n: "testset" in n or n.endswith(".txt"))
    extract(download(DNS5, raw / "dns5_blind.zip"), out / "dns5")
    # DAPS: keep the studio 'clean' versions and the real-device recordings; skip 'produced' and 'cleanraw'
    extract(download(DAPS, raw / "daps.tar.gz"), out / "daps",
            lambda n: any(f"/{d}" in n for d in ("clean/", "ipad", "iphone")) and "cleanraw" not in n)
    chime = out / "chime6"
    if not (chime / ".done").exists():
        stream([CHIME6.format("transcriptions")], chime, "*.json")
        for part in ("dev", "eval"):
            stream([CHIME6.format(part)], chime, "*_P[0-9][0-9].wav")
        if len(list(chime.rglob("*_P[0-9][0-9].wav"))) >= 16:
            (chime / ".done").touch()
    easycom = out / "easycom"
    if not (easycom / ".done").exists():
        stream([EASYCOM.format(i) for i in range(1, 34)], easycom, *(f"*Main/{d}/*" for d in (
            "Close_Microphone_Audio", "Glasses_Microphone_Array_Audio", "Speech_Transcriptions", "Metadata")))
        if any(easycom.rglob("Speech_Transcriptions/*/*.json")):
            (easycom / ".done").touch()
