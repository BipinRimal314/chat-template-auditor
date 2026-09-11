"""Pull chat templates from the Hub, distinguishing gated from genuinely absent."""
import json, urllib.request, urllib.error

RAW = "https://huggingface.co/{repo}/raw/main/{f}"
API = "https://huggingface.co/api/models/{repo}"
UA = {"User-Agent": "chat-template-auditor"}


def _get(url, token=None):
    h = dict(UA)
    if token:
        h["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def fetch_template(repo, token=None):
    """Returns (template_source, source_file, status). status is one of
    ok / gated / no-template / error."""
    for f in ("chat_template.jinja", "chat_template.json"):
        try:
            txt = _get(RAW.format(repo=repo, f=f), token)
            if f.endswith(".json"):
                d = json.loads(txt)
                txt = d.get("chat_template") if isinstance(d, dict) else None
                if not isinstance(txt, str):
                    continue
            return txt, f, "ok"
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return None, None, "gated"
            if e.code not in (404,):
                return None, None, f"error:{e.code}"
        except Exception:
            return None, None, "error"

    try:
        d = json.loads(_get(RAW.format(repo=repo, f="tokenizer_config.json"), token))
    except urllib.error.HTTPError as e:
        return None, None, "gated" if e.code in (401, 403) else "no-template"
    except Exception:
        return None, None, "error"

    ct = d.get("chat_template")
    if isinstance(ct, list) and ct:
        ct = ct[0].get("template") if isinstance(ct[0], dict) else ct[0]
    if isinstance(ct, str) and ct.strip():
        return ct, "tokenizer_config.json", "ok"
    return None, None, "no-template"


def model_meta(repo, token=None):
    try:
        return json.loads(_get(API.format(repo=repo), token))
    except Exception:
        return {}
