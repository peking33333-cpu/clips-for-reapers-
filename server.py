import os, re, json, uuid, time, shutil, socket, ipaddress, threading, traceback, zipfile
import urllib.request, urllib.parse, urllib.error
from pathlib import Path
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import engine as E

D = Path(os.getenv("DATA_DIR", "data")).resolve(); D.mkdir(parents=True, exist_ok=True)
MAX_BYTES = int(float(os.getenv("MAX_UPLOAD_GB", "10")) * (1 << 30))
MAX_HOURS = float(os.getenv("MAX_DURATION_HOURS", "6"))
KEEP_HOURS = float(os.getenv("RETENTION_HOURS", "24"))
SLOTS = threading.BoundedSemaphore(max(1, int(os.getenv("MAX_JOBS", "1"))))
UA = "Mozilla/5.0 (CLIPFORGE)"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".flv", ".ts", ".wmv", ".mpg", ".mpeg", ".3gp"}
RES = E.RES
ID = re.compile(r"^[0-9a-f]{10}$")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","), allow_methods=["*"], allow_headers=["*"])
J, CTX, LOCK = {}, {}, threading.RLock()


# ------------------------------------------------------------------ job store
def upd(i, **k):
    with LOCK:
        J[i].update(k)
        (D / i / "job.json").write_text(json.dumps(J[i]))


def load(i):
    if not ID.match(i):
        raise HTTPException(404, "Unknown job")
    with LOCK:
        if i not in J:
            p = D / i / "job.json"
            if not p.exists():
                raise HTTPException(404, "Unknown job (it may have expired)")
            J[i] = json.loads(p.read_text())
            if J[i].get("status") not in ("Done", "Failed", "Cancelled") and i not in CTX:
                J[i].update(status="Failed", error="The server restarted while this job was running. Please start it again.")
        return J[i]


def public(j):
    return {k: j.get(k) for k in ("id", "name", "status", "progress", "error", "clips", "aspect", "count", "dur", "duration", "analysis")}


def new_job(name, src, count, dur, aspect, sub, mute, url=None):
    i = uuid.uuid4().hex[:10]
    return i, {"id": i, "name": name, "src": src, "url": url, "count": count, "dur": dur, "aspect": aspect, "sub": sub,
               "mute": mute, "status": "Queued", "progress": 0, "error": "", "clips": [], "created": time.time()}


def start(i, j):
    with LOCK:
        J[i] = j; CTX[i] = E.Ctx()
        (D / i / "job.json").write_text(json.dumps(j))
    threading.Thread(target=run_job, args=(i,), daemon=True).start()


def cfg(count, dur, aspect, sub, mute):
    if aspect not in RES:
        raise HTTPException(400, "Unsupported aspect ratio")
    try:
        s = json.loads(sub or "{}")
        assert isinstance(s, dict)
    except Exception:
        raise HTTPException(400, "Invalid subtitle settings")
    return max(1, min(20, count)), max(10, min(120, dur)), aspect, s, str(mute).lower() in ("1", "true", "yes", "on")


# ------------------------------------------------------------ safe URL fetching
def check_public(u):
    p = urllib.parse.urlparse(u)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise ValueError("Only http(s) links are supported.")
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
    except OSError:
        raise ValueError("Could not find that website.")
    if not infos or any(not ipaddress.ip_address(x[4][0]).is_global for x in infos):
        raise ValueError("That address isn't reachable (local/private network addresses are blocked).")


class Guard(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_public(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


OPENER = urllib.request.build_opener(Guard)


def direct_info(u):
    check_public(u)
    for method, hdr in (("HEAD", {}), ("GET", {"Range": "bytes=0-0"})):
        try:
            rq = urllib.request.Request(u, method=method, headers={"User-Agent": UA, **hdr})
            with OPENER.open(rq, timeout=20) as r:
                return (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        except urllib.error.HTTPError as e:
            if method == "HEAD" and e.code in (400, 403, 405, 501):
                continue
            raise


def fix_share(u):
    p = urllib.parse.urlparse(u)
    if (p.hostname or "").endswith("dropbox.com"):                      # Dropbox share link -> raw file
        q = dict(urllib.parse.parse_qsl(p.query)); q["dl"] = "1"; q.pop("st", None)
        return p._replace(query=urllib.parse.urlencode(q)).geturl()
    return u


def download(ctx, url, dest, cb):
    check_public(url)
    rq = urllib.request.Request(url, headers={"User-Agent": UA})
    got = 0
    with OPENER.open(rq, timeout=60) as r, open(dest, "wb") as f:
        tot = int(r.headers.get("Content-Length") or 0)
        if tot > MAX_BYTES:
            raise E.EngineError(f"That file is larger than the {MAX_BYTES >> 30} GB limit.")
        while True:
            ctx.check()
            ch = r.read(1 << 20)
            if not ch:
                break
            got += len(ch)
            if got > MAX_BYTES:
                raise E.EngineError(f"That file is larger than the {MAX_BYTES >> 30} GB limit.")
            f.write(ch); cb(got / tot if tot else 0.5)
    if got == 0:
        raise E.EngineError("The link returned an empty file.")


# ------------------------------------------------------------------ the pipeline
def process(i, ctx, d, j, prog):
    src = d / j["src"]
    if j.get("url"):
        download(ctx, j["url"], src, lambda f: prog("Downloading video", 8 * f))
    prog("Checking video", 8)
    info = E.probe(src)
    if info["duration"] < 3:
        raise E.EngineError("The video is too short to clip.")
    if info["duration"] > MAX_HOURS * 3600:
        raise E.EngineError(f"The video is longer than the {MAX_HOURS:g} hour limit on this server.")
    if not info["has_audio"]:
        raise E.EngineError("This video has no audio track, so it can't be transcribed or exported with sound.")
    upd(i, duration=round(info["duration"], 1))
    prog("Extracting audio", 10)
    wav = d / "a.wav"
    E.extract_audio(ctx, src, wav)
    cuts = []

    def scenes():
        try:
            cuts.extend(E.scene_cuts(ctx, src))
        except Exception:
            pass                                                         # scene cuts are an extra signal, not essential
    th = threading.Thread(target=scenes if os.getenv("SCENE_DETECT", "1") == "1" else (lambda: None), daemon=True); th.start()
    S, lang = E.transcribe(ctx, wav, info["duration"], lambda f: prog("Transcribing", 12 + 48 * f))
    (d / "transcript.json").write_text(json.dumps(S))
    prog("Analysing audio and scenes", 60)
    db = E.audio_levels(wav)
    while th.is_alive():
        th.join(0.5); ctx.check()
    sig = E.Signals(db, cuts, info["duration"])
    prog("Finding best moments", 63)
    cands, meta = E.pick_moments(ctx, S, sig, j["count"], j["dur"], info["duration"], lambda f: prog("Finding best moments", 63 + 3 * f))
    meta.update(language=lang, words=sum(len(s["w"]) for s in S), scene_changes=len(cuts))
    upd(i, analysis=meta)
    moments = E.refine(cands, S, sig, j["dur"], info["duration"])[:j["count"]]
    if not moments:
        raise E.EngineError("No suitable moments were found in this video.")
    clips, errs = [], []
    for n, m in enumerate(moments, 1):
        base, span = 66 + 34 * (n - 1) / len(moments), 34 / len(moments)
        try:
            c = E.build_clip(ctx, d, src, n, m, info, j["aspect"], j["sub"], S, j["mute"],
                             lambda t, n=n, base=base: prog(f"{t}, clip {n}/{len(moments)}", base + span * 0.1))
            clips.append(c)
        except (E.Cancelled, E.EngineError) as e:
            if isinstance(e, E.Cancelled):
                raise
            errs.append(f"clip {n}: {e}")
        upd(i, clips=list(clips), progress=int(base + span))
    if not clips:
        raise E.EngineError("Every clip failed to render. " + " | ".join(errs)[:400])
    prog("Packaging", 99)
    make_zip(i, clips)
    for f in (src, wav):
        f.unlink(missing_ok=True)
    upd(i, status="Done", progress=100, clips=clips,
        error=(f"{len(errs)} clip(s) could not be rendered ({'; '.join(errs)[:300]})" if errs else ""))


def run_job(i):
    ctx, d = CTX[i], D / i
    def prog(status, p, **k):
        ctx.check(); upd(i, status=status, progress=int(p), **k)
    try:
        while not SLOTS.acquire(timeout=0.5):
            ctx.check()
        try:
            process(i, ctx, d, J[i], prog)
        finally:
            SLOTS.release()
    except E.Cancelled:
        upd(i, status="Cancelled", progress=0, error="")
    except Exception as e:
        traceback.print_exc()
        if isinstance(e, E.EngineError):
            msg = str(e)
        elif isinstance(e, urllib.error.HTTPError):
            msg = f"Download failed (HTTP {e.code}). The video may be private, deleted or restricted."
        elif isinstance(e, (urllib.error.URLError, ValueError, OSError)) and J[i].get("url"):
            msg = f"Download failed: {getattr(e, 'reason', e)}"
        else:
            msg = f"Unexpected error ({type(e).__name__}): {str(e)[:200]}"
        upd(i, status="Failed", error=msg)
    finally:
        CTX.pop(i, None)


def make_zip(i, clips):
    d = D / i
    tmp = d / "all.tmp.zip"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:           # MP4s are already compressed
        for c in clips:
            z.write(d / f"clip{c['n']}.mp4", f"clip{c['n']}_{slug(c['title'])}.mp4")
    os.replace(tmp, d / "all.zip")


def slug(t):
    return re.sub(r"[^a-z0-9]+", "-", (t or "").lower()).strip("-")[:40] or "clip"


# ------------------------------------------------------------------ endpoints
@app.post("/api/upload")
async def up(file: UploadFile, count: int = Form(5), dur: int = Form(30), aspect: str = Form("9:16"), sub: str = Form("{}"), mute: str = Form("false")):
    count, dur, aspect, s, mute = cfg(count, dur, aspect, sub, mute)
    ext = Path(file.filename or "v.mp4").suffix.lower()
    ext = ext if re.fullmatch(r"\.[a-z0-9]{1,5}", ext) else ".mp4"
    i, j = new_job(file.filename or "upload", "src" + ext, count, dur, aspect, s, mute)
    (D / i).mkdir()
    n = 0
    try:
        with open(D / i / j["src"], "wb") as f:
            while ch := await file.read(1 << 20):
                n += len(ch)
                if n > MAX_BYTES:
                    raise HTTPException(413, f"File is larger than the {MAX_BYTES >> 30} GB limit.")
                f.write(ch)
        if n == 0:
            raise HTTPException(400, "The uploaded file is empty.")
    except BaseException:
        shutil.rmtree(D / i, ignore_errors=True); raise
    start(i, j)
    return {"id": i}


@app.get("/api/job/{i}")
def job(i: str):
    return public(load(i))


@app.post("/api/job/{i}/cancel")
def cancel(i: str):
    load(i)
    c = CTX.get(i)
    if c:
        c.stop()
    return {"ok": True}


@app.get("/clip/{i}/{n}")
def clip(i: str, n: int, dl: int = 0):
    j = load(i)
    c = next((x for x in j.get("clips", []) if x["n"] == n), None)
    p = D / i / f"clip{n}.mp4"
    if not c or not p.exists():
        raise HTTPException(404, "Clip not found (it may have expired)")
    return FileResponse(p, media_type="video/mp4", filename=f"clip{n}_{slug(c['title'])}.mp4" if dl else None)


@app.get("/zip/{i}")
def zp(i: str):
    j = load(i)
    clips = [c for c in j.get("clips", []) if (D / i / f"clip{c['n']}.mp4").exists()]
    if not clips:
        raise HTTPException(404, "No clips to download yet")
    make_zip(i, clips)
    return FileResponse(D / i / "all.zip", media_type="application/zip", filename="clips.zip")


@app.get("/api/health")
def health():
    import shutil as sh
    try:
        import faster_whisper; stt = True
    except ImportError:
        stt = False
    return {"ffmpeg": bool(sh.which("ffmpeg") and sh.which("ffprobe")), "speech_to_text": stt,
            "ai_moment_selection": bool(os.getenv("ANTHROPIC_API_KEY")), "whisper_model": os.getenv("WHISPER", "small")}


def jget(u, h=None, data=None):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, data, h or {}), timeout=20))


@app.post("/api/probe")
def probe(url: str = Form(...)):
    u = url.strip(); p = urllib.parse.urlparse(u); h = (p.hostname or "").lower().removeprefix("www.")
    out = {"url": u, "importable": False, "live": None}
    try:
        if h in ("youtube.com", "m.youtube.com", "youtu.be"):
            o = jget("https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(u, safe=""))
            out.update(platform="YouTube", title=o["title"], creator=o["author_name"], thumb=o["thumbnail_url"],
                note="YouTube has no authorised way to fetch video files from a link. Download your own recording from YouTube Studio (or get the owner's permission) and upload it below. Live status and live monitoring are not available.")
        elif h.endswith("twitch.tv"):
            note = "Twitch has no authorised way to fetch video files from a link. Download your own broadcast from Creator Dashboard > Video Producer and upload it below."
            out.update(platform="Twitch", note=note)
            cid, sec = os.getenv("TWITCH_CLIENT_ID"), os.getenv("TWITCH_CLIENT_SECRET")
            if not cid:
                out.update(title="Twitch link recognised", note=note + " Set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET to show title, creator and live status.")
            else:
                t = jget("https://id.twitch.tv/oauth2/token", data=urllib.parse.urlencode({"client_id": cid, "client_secret": sec, "grant_type": "client_credentials"}).encode())
                hd = {"Client-Id": cid, "Authorization": "Bearer " + t["access_token"]}
                m = re.search(r"/videos/(\d+)", p.path)
                if m:
                    v = jget("https://api.twitch.tv/helix/videos?id=" + m.group(1), hd)["data"][0]
                    out.update(title=v["title"], creator=v["user_name"], live=False, duration=v["duration"],
                        thumb=v["thumbnail_url"].replace("%{width}", "640").replace("%{height}", "360"))
                else:
                    ch = p.path.strip("/").split("/")[0]
                    s = jget("https://api.twitch.tv/helix/streams?user_login=" + ch, hd)["data"]
                    out.update(creator=ch, live=bool(s), title=s[0]["title"] if s else "Channel is offline",
                        thumb=s[0]["thumbnail_url"].replace("{width}", "640").replace("{height}", "360") if s else None)
        else:
            u = fix_share(u); p = urllib.parse.urlparse(u)
            ct = direct_info(u)
            ext = Path(p.path).suffix.lower()
            if not (ct.startswith("video/") or (ct in ("application/octet-stream", "binary/octet-stream") and ext in VIDEO_EXT)):
                raise ValueError("That link isn't a direct video file (expected an .mp4/.mov/.webm… file link).")
            out.update(url=u, platform="Direct link", title=Path(p.path).name or "video", creator=h, importable=True, live=False)
    except urllib.error.HTTPError as e:
        raise HTTPException(400, f"Video unavailable (HTTP {e.code}). It may be private, deleted or restricted.")
    except Exception as e:
        raise HTTPException(400, str(getattr(e, "reason", e)) or "Could not check that link.")
    return out


@app.post("/api/import")
def imp(url: str = Form(...), count: int = Form(5), dur: int = Form(30), aspect: str = Form("9:16"), sub: str = Form("{}"), mute: str = Form("false")):
    count, dur, aspect, s, mute = cfg(count, dur, aspect, sub, mute)
    o = probe(url)
    if not o["importable"]:
        raise HTTPException(400, o["note"])
    ext = Path(urllib.parse.urlparse(o["url"]).path).suffix.lower()
    i, j = new_job(o["title"], "src" + (ext if ext in VIDEO_EXT else ".mp4"), count, dur, aspect, s, mute, url=o["url"])
    (D / i).mkdir()
    start(i, j)
    return {"id": i}


def janitor():
    while True:
        try:
            for p in D.iterdir():
                if p.is_dir() and ID.match(p.name) and p.name not in CTX and time.time() - p.stat().st_mtime > KEEP_HOURS * 3600:
                    shutil.rmtree(p, ignore_errors=True); J.pop(p.name, None)
        except Exception:
            pass
        time.sleep(3600)


threading.Thread(target=janitor, daemon=True).start()


@app.get("/")
def home():
    here = Path(__file__).parent
    for p in (here / "index.html", here.parent / "frontend" / "index.html"):
        if p.exists():
            return FileResponse(p)
    raise HTTPException(404, "Frontend is hosted separately")
