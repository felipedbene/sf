# requirements: google-auth google-auth-oauthlib google-api-python-client pdfplumber pandas networkx pyvis openai jinja2 tqdm html2text pytesseract pillow

import os
import re
import json
import base64
import hashlib
import sys
from datetime import datetime
from typing import List, Dict, Union
import email
from email import policy
from email.parser import BytesParser
from html import unescape

def _html_to_text(html: str) -> str:
    """Convert HTML to plain text using html2text if available."""
    try:
        import html2text

        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        return h.handle(html)
    except Exception:
        logging.debug("html2text unavailable; falling back to regex stripping")
        # Replace common block tags with newlines to preserve structure
        html = re.sub(r"<(br|p|div|li|tr)[^>]*>", "\n", html, flags=re.I)
        # Remove all remaining tags
        text = re.sub(r"<[^>]+>", "", html)
        return unescape(text)

import logging
import pdfplumber
import pandas as pd
import networkx as nx
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import openai
from pyvis.network import Network
from tqdm import tqdm

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
EVIDENCE_DIR = "evidence"
CACHE_DIR = "cache"

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# File handler captures full debug logs
file_handler = logging.FileHandler("debug.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))

# Stream handler shows highlights to the user
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))

logger.handlers = [file_handler, console_handler]
# Suppress verbose pdfminer warnings such as missing CropBox messages
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def get_credentials() -> Credentials:
    """Load Gmail credentials using token.json cache."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds

def fetch_claim_emails(
    claim_number: str | None = None,
    keywords: List[str] | None = None,
    use_cache: bool = True,
) -> List[dict]:
    """Fetch claim-related emails matching a claim number and keywords."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    query_parts: List[str] = []
    if claim_number:
        query_parts.append(f'subject:"{claim_number}"')
    if keywords:
        query_parts.extend(keywords)
    if not query_parts:
        query_parts.append('"State Farm Claim"')
    query = " ".join(query_parts)

    cache_key = hashlib.md5(query.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"gmail_messages_{cache_key}.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    creds = get_credentials()
    service = build("gmail", "v1", credentials=creds)
    resp = service.users().messages().list(userId="me", q=query).execute()
    messages = []
    for item in tqdm(resp.get("messages", []), desc="Downloading emails"):
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=item["id"], format="full")
            .execute()
        )
        messages.append(msg)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(messages, f)

    return messages



def _extract_eml_text(path: str) -> str:
    """Return plain text from an .eml file."""
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype in ("text/plain", "text/html"):
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    text = payload.decode(charset, errors="ignore")
                    if ctype == "text/html":
                        text = _html_to_text(text)
                    parts.append(text)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            parts.append(payload.decode(charset, errors="ignore"))
    return "\n".join(parts)


DEFAULT_JUNK_PATTERNS = [
    r"^\s*Page\s+\d+",
    r"GOLD COAST AUTO BODY INC",
    r"^\s*(?:\d+/\d+/\d+|\d+)\s+E\d{2}\b",
]


def _load_junk_patterns() -> List[re.Pattern]:
    """Load regex patterns for text cleanup from config.json if present."""
    patterns = list(DEFAULT_JUNK_PATTERNS)
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                cfg = json.load(f)
            patterns.extend(cfg.get("junk_patterns", []))
        except Exception as exc:
            logging.debug(f"Failed to load config.json: {exc}")
    return [re.compile(p, flags=re.I) for p in patterns]


JUNK_PATTERNS = _load_junk_patterns()


def _clean_text(text: str) -> str:
    """Remove configured junk lines from extracted text."""
    patterns = globals().get("JUNK_PATTERNS")
    if patterns is None:
        default_pats = [
            r"^\s*Page\s+\d+",
            r"GOLD COAST AUTO BODY INC",
            r"^\s*(?:\d+/\d+/\d+|\d+)\s+E\d{2}\b",
        ]
        patterns = [re.compile(p, flags=re.I) for p in default_pats]
    cleaned_lines = []
    for line in text.splitlines():
        if any(p.search(line) for p in patterns):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _save_attachment(service, msg_id: str, part: dict, folder: str) -> Union[str, None]:
    """Save attachment to folder, recursing into nested parts."""
    if part.get("parts"):
        for sub in part["parts"]:
            _save_attachment(service, msg_id, sub, folder)
        return None

    att_id = part.get("body", {}).get("attachmentId")
    if not att_id:
        return None

    filename = part.get("filename") or "part"
    filename = re.sub(r"[\\/*?:\"<>|]", "_", filename)
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        return path

    try:
        att = (
            service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=msg_id, id=att_id)
            .execute()
        )
        data = base64.urlsafe_b64decode(att["data"].encode("UTF-8"))
        with open(path, "wb") as f:
            f.write(data)
        logging.debug(f"Saved attachment {path}")
        return path
    except Exception as exc:
        logging.debug(f"Failed saving attachment {filename}: {exc}")
        return None


def download_and_extract(messages: List[dict], use_cache: bool = True) -> List[Dict]:
    """Download attachments and extract text and OCR from emails."""
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "texts.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    creds = get_credentials()
    service = build("gmail", "v1", credentials=creds)

    texts: List[Dict] = []
    index_entries = []

    for msg in tqdm(messages, desc="Processing messages"):
        msg_id = msg.get("id")
        headers = {h["name"]: h.get("value", "") for h in msg.get("payload", {}).get("headers", [])}
        sender = headers.get("From", "unknown")
        date_ts = int(msg.get("internalDate", "0")) / 1000
        date_str = datetime.fromtimestamp(date_ts).strftime("%Y-%m-%d")
        folder_name = f"{date_str}_{re.sub(r'[^a-zA-Z0-9_]+', '_', sender)}"
        folder_path = os.path.join(EVIDENCE_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        meta = {
            "msg_id": msg_id,
            "sender": sender,
            "date": date_str,
            "message_id": headers.get("Message-ID"),
            "in_reply_to": headers.get("In-Reply-To"),
            "references": headers.get("References"),
        }

        def process_part(part: dict):
            if part.get("parts"):
                for sp in part["parts"]:
                    process_part(sp)
                return

            mime = part.get("mimeType", "")
            body = part.get("body", {})
            data = body.get("data")

            if data and not body.get("attachmentId") and mime in ("text/plain", "text/html"):
                try:
                    text = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                    if mime == "text/html":
                        text = _html_to_text(text)
                    texts.append({"text": text, **meta})
                except Exception as exc:
                    logging.debug(f"Failed to decode inline text: {exc}")
                return

            path = _save_attachment(service, msg_id, part, folder_path)
            if not path:
                return

            index_entries.append({"msg_id": msg_id, "filename": os.path.basename(path), "sender": sender, "date": date_str, "mime_type": mime})

            try:
                if mime == "application/pdf":
                    with pdfplumber.open(path) as pdf:
                        pdf_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                    texts.append({"text": _clean_text(pdf_text), **meta})
                elif mime.startswith("text/"):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        texts.append({"text": f.read(), **meta})
                elif mime == "message/rfc822" or path.lower().endswith(".eml"):
                    texts.append({"text": _extract_eml_text(path), **meta})
                elif mime in ("image/jpeg", "image/png"):
                    try:
                        from PIL import Image
                        import pytesseract

                        ocr_text = pytesseract.image_to_string(Image.open(path))
                        texts.append({"text": ocr_text, **meta})
                    except Exception as exc:
                        logging.debug(f"OCR failed for {path}: {exc}")
            except Exception as exc:
                logging.debug(f"Failed processing attachment {path}: {exc}")

        payload = msg.get("payload", {})
        process_part(payload)

    # evidence/index.json
    try:
        with open(os.path.join(EVIDENCE_DIR, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index_entries, f, indent=2)
    except Exception as exc:
        logging.debug(f"Failed writing index.json: {exc}")

    try:
        with open(os.path.join(EVIDENCE_DIR, "evidence_summary.csv"), "w", encoding="utf-8") as f:
            f.write("msg_id,sender,date,filename,preview\n")
            for entry in index_entries:
                text_snippets = [t["text"] for t in texts if t["msg_id"] == entry["msg_id"]]
                preview = " ".join(text_snippets)[:200].replace("\n", " ").replace("\r", " ")
                f.write(
                    f"{entry['msg_id']},{entry['sender']},{entry['date']},{entry['filename']},\"{preview}\"\n"
                )
    except Exception as exc:
        logging.debug(f"Failed writing evidence_summary.csv: {exc}")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(texts, f)
    return texts

def _normalize_timestamp(ts: str) -> str:
    """Return an ISO formatted timestamp if possible."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d %I:%M:%S %p",
        "%Y-%m-%d %I:%M %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %I:%M %p",
    ):
        try:
            return datetime.strptime(ts, fmt).isoformat()
        except ValueError:
            continue
    return ts


def parse_events(texts: List) -> List[Dict]:
    """Extract events from various timestamped formats."""
    events: List[Dict] = []
    seen = set()
    ts_part = (
        r"(?P<timestamp>(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})"
        r"[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)"
    )
    next_ts = r"(?=(?:\r?\n)+(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})[ T]\d{1,2}:\d{2}|\Z)"
    patterns = [
        re.compile(
            ts_part
            + r"[ \t]*(?P<actor>[^:]+):[ \t]*(?P<action>[^.\n]+)\.?[ \t]*(?P<details>.*?)"
            + next_ts,
            re.DOTALL,
        ),
        re.compile(
            ts_part
            + r"[ \t]*-[ \t]*(?P<actor>[^-]+)[ \t]*-[ \t]*(?P<action>[^-\n]+)[ \t]*-[ \t]*(?P<details>.*?)"
            + next_ts,
            re.DOTALL,
        ),
        re.compile(
            ts_part
            + r"[ \t]+(?P<actor>[^-:\n]+)[ \t]+(?P<action>[^-:\n]+)[ \t]*(?P<details>.*?)"
            + next_ts,
            re.DOTALL,
        ),
    ]
    count = 0
    for item in tqdm(texts, desc="Parsing events"):
        meta = {}
        if isinstance(item, dict):
            text = item.get("text", "")
            meta = {k: item.get(k) for k in ("msg_id", "sender", "date", "message_id", "in_reply_to", "references")}
        else:
            text = item
        text = _clean_text(text)
        for pattern in patterns:
            for m in pattern.finditer(text):
                key = (m.group("timestamp"), m.group("actor"), m.group("action"), m.groupdict().get("details", ""))
                if key in seen:
                    continue
                seen.add(key)
                count += 1
                events.append(
                    {
                        "id": f"evt_{count:03d}",
                        "actor": m.group("actor").strip(),
                        "type": "Action",
                        "action": m.group("action").strip(),
                        "timestamp": _normalize_timestamp(m.group("timestamp")),
                        "details": m.groupdict().get("details", "").strip(),
                        **meta,
                    }
                )
    return events

def build_graph(events: List[Dict], max_back_links: int = 1) -> nx.DiGraph:
    """Return a directed graph linking each event to earlier ones.

    Nodes are added for each event in chronological order. By default an event
    links only to the immediately preceding event. ``max_back_links`` controls
    how many prior events may connect to the current one. If the event ``details``
    contain the phrases "in response to" or "following" (case insensitive), the
    event links to **all** previous events regardless of this limit.
    """

    G = nx.DiGraph()
    for evt in tqdm(events, desc="Adding nodes"):
        clean = {k: v for k, v in evt.items() if v is not None}
        G.add_node(evt["id"], **clean)
    for i, evt in enumerate(tqdm(events, desc="Linking events")):
        details_lower = evt["details"].lower()
        if "in response to" in details_lower or "following" in details_lower:
            back_range = range(i)
        else:
            if max_back_links is None or max_back_links <= 0:
                back_range = []
            else:
                back_range = range(max(0, i - max_back_links), i)
        for j in back_range:
            prev = events[j]
            G.add_edge(prev["id"], evt["id"])
    return G

def detect_irregularities(events: List[Dict], use_cache: bool = True) -> List[Dict]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    events_json = json.dumps(events, sort_keys=True)
    cache_key = hashlib.md5(events_json.encode("utf-8")).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"openai_{cache_key}.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        if not result:
            print("No irregularities found")
        return result

    openai.api_key = os.environ.get("OPENAI_API_KEY")
    system_prompt = (
        "You are a claims-compliance auditor. Given this list of events in order, "
        "identify any irregular or suspicious patterns (e.g. payments before approvals, "
        "missing communications, duplicate requests).  Return a JSON array of event IDs "
        "flagged, each with a brief reason and a severity score (1-10)."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": events_json},
    ]
    try:
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages
        )
    except Exception as exc:
        print(f"OpenAI API request failed: {exc}")
        print("No irregularities found")
        return []

    content = resp.choices[0].message.content
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Sometimes the assistant responds with a preamble or other text
        # surrounding the JSON. Attempt to extract the JSON block.
        match = re.search(r"(\{.*\}|\[.*\])\s*$", content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
            except json.JSONDecodeError:
                print("Could not parse irregularities response:")
                print(content)
                print("No irregularities found")
                return []
        else:
            print("Could not parse irregularities response:")
            print(content)
            print("No irregularities found")
            return []
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    if not result:
        print("No irregularities found")
    return result

def annotate_graph(G: nx.DiGraph, irregulars: List[Dict]) -> None:
    # OpenAI responses may occasionally be JSON strings or contain items that are
    # themselves JSON encoded. Normalize them into dictionaries before use.
    if isinstance(irregulars, str):
        try:
            irregulars = json.loads(irregulars)
        except json.JSONDecodeError:
            print("Could not parse irregularities response")
            return
    for irr in irregulars:
        if isinstance(irr, str):
            try:
                irr = json.loads(irr)
            except json.JSONDecodeError:
                continue
        evt_id = irr.get("id") or irr.get("event_id")
        if evt_id in G.nodes:
            score = irr.get("score")
            if score is None:
                score = irr.get("severity_score") or irr.get("severity")
            G.nodes[evt_id]["irregularity_score"] = score
            G.nodes[evt_id]["irregularity_reason"] = irr.get("reason")

def summarize_irregularities(G: nx.DiGraph, irregulars: List[Dict]) -> pd.DataFrame:
    rows = []
    if isinstance(irregulars, str):
        try:
            irregulars = json.loads(irregulars)
        except json.JSONDecodeError:
            print("Could not parse irregularities response")
            irregulars = []
    for irr in irregulars:
        if isinstance(irr, str):
            try:
                irr = json.loads(irr)
            except json.JSONDecodeError:
                continue
        evt_id = irr.get("id") or irr.get("event_id")
        node = G.nodes.get(evt_id, {})
        rows.append(
            {
                "Event ID": evt_id,
                "Actor": node.get("actor"),
                "Action": node.get("action"),
                "Timestamp": node.get("timestamp"),
                "Score": irr.get("score")
                if irr.get("score") is not None
                else irr.get("severity_score")
                if irr.get("severity_score") is not None
                else irr.get("severity"),
                "Reason": irr.get("reason"),
            }
        )
    df = pd.DataFrame(rows)
    print(
        df.to_string(
            index=False,
            justify="left",
        )
    )
    return df

def _severity_color(score: int) -> str:
    if score is None:
        score = 0
    score = max(0, min(10, int(score)))
    r = int(255 * score / 10)
    g = int(255 * (10 - score) / 10)
    return f"#{r:02x}{g:02x}00"


def visualize(G: nx.DiGraph) -> None:
    net = Network(directed=True)
    for node, data in G.nodes(data=True):
        score = data.get("irregularity_score", 0)
        color = _severity_color(score)
        title = f"{data.get('details','')}\nReason: {data.get('irregularity_reason','')}"
        label = f"{data.get('actor')}\n{data.get('action')}"
        net.add_node(node, label=label, title=title, color=color)
    for src, dst in G.edges():
        net.add_edge(src, dst)
    html_file = "claim_irregularity_map.html"
    try:
        # write_html avoids attempting to open a browser which can
        # fail in some environments and also bypasses template caching
        # issues seen with ``show``.
        net.write_html(html_file)
    except Exception:
        # Workaround for occasional pyvis template loading issues which
        # manifest as 'NoneType has no attribute "render"'.  Attempt to
        # reload the default template and retry once before giving up.
        try:
            if hasattr(net, "get_template"):
                net.template = net.get_template()
            net.write_html(html_file)
        except Exception as exc:
            print(f"Visualization failed: {exc}")
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(
                    "<html><body><p>Unable to generate visualization.</p></body></html>"
                )
    nx.write_graphml(G, "claim_graph.graphml")

def main():
    refresh = "--refresh" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    claim_number = args[0] if args else None
    keywords = args[1:] if len(args) > 1 else None
    messages = fetch_claim_emails(claim_number=claim_number, keywords=keywords, use_cache=not refresh)
    texts = download_and_extract(messages, use_cache=not refresh)
    events = parse_events(texts)
    G = build_graph(events)
    irregulars = detect_irregularities(events, use_cache=not refresh)
    annotate_graph(G, irregulars)
    summarize_irregularities(G, irregulars)
    visualize(G)


if __name__ == "__main__":
    main()
