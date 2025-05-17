import os
import re
import json
import base64
import hashlib
import sys
from datetime import datetime
from typing import List, Dict

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

def fetch_claim_emails(use_cache: bool = True) -> List[dict]:
    """Fetch emails for 'State Farm claim 13-83R9-01P' thread."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "gmail_messages.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    creds = get_credentials()
    service = build("gmail", "v1", credentials=creds)
    query = '"State Farm Claim"'
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

def _save_attachment(service, msg_id, part, path):
    att_id = part["body"].get("attachmentId")
    if not att_id:
        return
    if os.path.exists(path):
        return
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


def download_and_extract(messages: List[dict], use_cache: bool = True) -> List[str]:
    """Download attachments and extract text from emails and PDFs."""
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "texts.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    creds = get_credentials()
    service = build("gmail", "v1", credentials=creds)
    texts = []
    for msg in tqdm(messages, desc="Processing messages"):
        msg_id = msg["id"]
        payload = msg.get("payload", {})
        parts = payload.get("parts", [])
        if payload.get("body", {}).get("data"):
            body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")
            texts.append(body)
        for part in parts:
            filename = part.get("filename") or "part"
            # Sanitize filename to avoid invalid characters on Windows
            filename = re.sub(r"[\\/*?:\"<>|]", "_", filename)
            mime = part.get("mimeType", "")
            if part.get("body", {}).get("attachmentId"):
                path = os.path.join(EVIDENCE_DIR, f"{msg_id}_{filename}")
                _save_attachment(service, msg_id, part, path)
                if mime == "application/pdf":
                    with pdfplumber.open(path) as pdf:
                        pdf_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                    texts.append(pdf_text)
                elif mime.startswith("text/"):
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        texts.append(f.read())
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
    ):
        try:
            return datetime.strptime(ts, fmt).isoformat()
        except ValueError:
            continue
    return ts


def parse_events(texts: List[str]) -> List[Dict]:
    """Extract events from various timestamped formats."""
    events = []
    ts_part = r"(?P<timestamp>(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})[ T]\d{2}:\d{2}(?::\d{2})?)"
    patterns = [
        re.compile(ts_part + r"\s*(?P<actor>[^:]+):\s*(?P<action>[^.\n]+)\.?\s*(?P<details>.*)", re.DOTALL),
        re.compile(ts_part + r"\s*-\s*(?P<actor>[^-]+)\s*-\s*(?P<action>[^-\n]+)\s*-\s*(?P<details>.*)", re.DOTALL),
    ]
    count = 0
    for text in tqdm(texts, desc="Parsing events"):
        for line in text.splitlines():
            for pattern in patterns:
                m = pattern.search(line)
                if m:
                    count += 1
                    events.append(
                        {
                            "id": f"evt_{count:03d}",
                            "actor": m.group("actor").strip(),
                            "type": "Action",
                            "action": m.group("action").strip(),
                            "timestamp": _normalize_timestamp(m.group("timestamp")),
                            "details": m.group("details").strip(),
                        }
                    )
                    break
    return events

def build_graph(events: List[Dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for evt in tqdm(events, desc="Adding nodes"):
        G.add_node(evt["id"], **evt)
    for i, evt in enumerate(tqdm(events, desc="Linking events")):
        ts_i = datetime.fromisoformat(evt["timestamp"])
        for j in range(i):
            prev = events[j]
            ts_j = datetime.fromisoformat(prev["timestamp"])
            if any(x in evt["details"].lower() for x in ["in response to", "following"]):
                G.add_edge(prev["id"], evt["id"])
            elif 0 < (ts_i - ts_j).days <= 1:
                G.add_edge(prev["id"], evt["id"])
    return G

def detect_irregularities(events: List[Dict], use_cache: bool = True) -> List[Dict]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    events_json = json.dumps(events, sort_keys=True)
    cache_key = hashlib.md5(events_json.encode("utf-8")).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"openai_{cache_key}.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

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
    resp = openai.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    result = json.loads(resp.choices[0].message.content)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
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
            G.nodes[evt_id]["irregularity_score"] = irr.get("score")
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
                "Score": irr.get("score"),
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
    messages = fetch_claim_emails(use_cache=not refresh)
    texts = download_and_extract(messages, use_cache=not refresh)
    events = parse_events(texts)
    G = build_graph(events)
    irregulars = detect_irregularities(events, use_cache=not refresh)
    annotate_graph(G, irregulars)
    summarize_irregularities(G, irregulars)
    visualize(G)


if __name__ == "__main__":
    main()
