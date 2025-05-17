import os
import re
import json
import base64
from datetime import datetime
from typing import List, Dict

import pdfplumber
import pandas as pd
import networkx as nx
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import openai
from pyvis.network import Network

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
EVIDENCE_DIR = "evidence"


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

def fetch_claim_emails() -> List[dict]:
    """Fetch unread emails for 'State Farm Claim' thread."""
    creds = get_credentials()
    service = build("gmail", "v1", credentials=creds)
    query = 'is:unread "State Farm Claim"'
    resp = service.users().messages().list(userId="me", q=query).execute()
    messages = []
    for item in resp.get("messages", []):
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=item["id"], format="full")
            .execute()
        )
        messages.append(msg)
    return messages

def _save_attachment(service, msg_id, part, path):
    att_id = part["body"].get("attachmentId")
    if not att_id:
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


def download_and_extract(messages: List[dict]) -> List[str]:
    """Download attachments and extract text from emails and PDFs."""
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    creds = get_credentials()
    service = build("gmail", "v1", credentials=creds)
    texts = []
    for msg in messages:
        msg_id = msg["id"]
        payload = msg.get("payload", {})
        parts = payload.get("parts", [])
        if payload.get("body", {}).get("data"):
            body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")
            texts.append(body)
        for part in parts:
            filename = part.get("filename") or "part"
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
    return texts

def parse_events(texts: List[str]) -> List[Dict]:
    """Extract events from text using regex."""
    events = []
    pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}).*?(?P<actor>[^:]+): (?P<action>[^.\n]+)\.(?P<details>.*)",
        re.DOTALL,
    )
    count = 0
    for text in texts:
        for m in pattern.finditer(text):
            count += 1
            events.append(
                {
                    "id": f"evt_{count:03d}",
                    "actor": m.group("actor").strip(),
                    "type": "Action",
                    "action": m.group("action").strip(),
                    "timestamp": m.group("timestamp"),
                    "details": m.group("details").strip(),
                }
            )
    return events

def build_graph(events: List[Dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for evt in events:
        G.add_node(evt["id"], **evt)
    for i, evt in enumerate(events):
        ts_i = datetime.fromisoformat(evt["timestamp"])
        for j in range(i):
            prev = events[j]
            ts_j = datetime.fromisoformat(prev["timestamp"])
            if any(x in evt["details"].lower() for x in ["in response to", "following"]):
                G.add_edge(prev["id"], evt["id"])
            elif 0 < (ts_i - ts_j).days <= 1:
                G.add_edge(prev["id"], evt["id"])
    return G

def detect_irregularities(events: List[Dict]) -> List[Dict]:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    system_prompt = (
        "You are a claims-compliance auditor. Given this list of events in order, "
        "identify any irregular or suspicious patterns (e.g. payments before approvals, "
        "missing communications, duplicate requests).  Return a JSON array of event IDs "
        "flagged, each with a brief reason and a severity score (1-10)."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(events)},
    ]
    resp = openai.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return json.loads(resp.choices[0].message.content)

def annotate_graph(G: nx.DiGraph, irregulars: List[Dict]) -> None:
    for irr in irregulars:
        evt_id = irr.get("id") or irr.get("event_id")
        if evt_id in G.nodes:
            G.nodes[evt_id]["irregularity_score"] = irr.get("score")
            G.nodes[evt_id]["irregularity_reason"] = irr.get("reason")

def summarize_irregularities(G: nx.DiGraph, irregulars: List[Dict]) -> pd.DataFrame:
    rows = []
    for irr in irregulars:
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
    net.show("claim_irregularity_map.html")
    nx.write_graphml(G, "claim_graph.graphml")

def main():
    messages = fetch_claim_emails()
    texts = download_and_extract(messages)
    events = parse_events(texts)
    G = build_graph(events)
    irregulars = detect_irregularities(events)
    annotate_graph(G, irregulars)
    summarize_irregularities(G, irregulars)
    visualize(G)


if __name__ == "__main__":
    main()
