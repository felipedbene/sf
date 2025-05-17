# Claim Irregularity Finder — Agents

This file describes each logical agent in the system, its responsibility, inputs, outputs, and the tools it uses.

| Agent Name                  | Responsibility                                                  | Inputs                             | Outputs                                  | Tools / Modules                        |
|-----------------------------|-----------------------------------------------------------------|------------------------------------|------------------------------------------|----------------------------------------|
| **GmailFetcherAgent**       | Connect to Gmail and pull all emails matching the claim number  | `CLAIM_NUMBER`, `use_cache` flag   | `List[Message]` (raw Gmail JSON objects) | `google-auth-oauthlib`, `gmail_v1`     |
| **AttachmentDownloaderAgent** | Download all attachments (PDF, images, EML) into `evidence/` | `List[Message]`, `use_cache` flag  | `List[file_paths]`                       | `base64`, `pdfplumber`, `pytesseract`  |
| **TextExtractorAgent**      | Turn attachments and email bodies into plain-text snippets     | `List[file_paths]`                 | `List[str]` (raw text blocks)            | `pdfplumber`, `html2text`, `pytesseract`|
| **EventParserAgent**        | Scan each text block for timestamped events, filter noise      | `List[str]`                        | `List[EventDict]`                        | `re`, timestamp normalizers            |
| **GraphBuilderAgent**       | Build a directed graph of cause→effect between events          | `List[EventDict]`                  | `nx.DiGraph`                             | `networkx`                             |
| **IrregularityDetectorAgent** | Ask OpenAI to flag suspicious events                         | `List[EventDict]`                  | `List[IrregularityDict]`                 | `openai`                               |
| **AnnotatorAgent**          | Annotate graph nodes with irregularity metadata               | `nx.DiGraph`, `List[IrregularityDict]` | `nx.DiGraph` (with node attrs)        | `networkx`                             |
| **ReporterAgent**           | Print CSV summary of flagged events                            | `nx.DiGraph`, `List[IrregularityDict]` | `pandas.DataFrame`                   | `pandas`                               |
| **VisualizerAgent**         | Render interactive HTML + export GraphML/JSON                  | `nx.DiGraph`                        | `claim_irregularity_map.html`, `*.graphml`, `graph_data.json` | `pyvis`, `networkx`                    |

---

## How to use

1. **Run** `python claim_irregularity_finder.py`  
2. Each agent will run in turn, logging its steps.  
3. Final outputs:
   - **Evidence folder** with organized attachments  
   - **CSV** report of irregularities  
   - **Interactive HTML** graph  

Keeping this `agents.md` in your repo root makes it easy for new contributors (or auditors) to see exactly what each stage does and how data flows through the system.
