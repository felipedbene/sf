# Claim Irregularity Finder

This project scans claim-related emails to identify irregularities in the claim handling process. It uses the Gmail API to download messages and attachments, extracts events from text and PDF files, and then applies the OpenAI API to flag suspicious patterns. A network graph of events is generated and saved as HTML and GraphML.

## Setup

1. **Python Dependencies**

   Install required packages with pip:

   ```bash
   pip install google-auth google-auth-oauthlib google-api-python-client pdfplumber pandas networkx pyvis openai jinja2 tqdm html2text pytesseract pillow
   ```

2. **Gmail Credentials**

   - Obtain `credentials.json` from the Google Cloud console for a Desktop application.
   - When running the script for the first time, a browser window will prompt for Gmail authorization. A `token.json` cache is then created for future use.

3. **OpenAI API Key**

Export your API key in the environment before running the script:

   ```bash
   export OPENAI_API_KEY=your-key-here
   ```

   Without a valid API key the irregularity detection step will produce an empty
   table.

## Usage

Run the main script to fetch claim emails, analyze them, and generate the visualization.
Pass the claim number followed by any additional keywords to refine the Gmail search:

```bash
python claim_irregularity_finder.py 13-83R9-01P statefarm claim brett riley
```

The script will:

1. Fetch unread Gmail messages matching a claim number and optional keywords.
2. Download any attachments to the `evidence` directory and extract text from PDFs or text files.  Common headers, page numbers, and line-item table rows are removed during preprocessing.
3. Parse events from the collected text and build a directed graph with NetworkX.
4. Send the list of events to the OpenAI API to detect irregular patterns.
5. Annotate the graph with the irregularity results, print a summary table, and produce `claim_irregularity_map.html` and `claim_graph.graphml`.
6. Display progress bars in the terminal while processing messages, parsing events, and building the graph.

The script caches downloaded emails, extracted text, and OpenAI analysis results
in the `cache` directory. Run the script with `--refresh` to ignore the cache
and fetch fresh data.

## Output Files

- **claim_irregularity_map.html** – interactive visualization of event relationships with irregularities highlighted.
- **claim_graph.graphml** – GraphML version of the event graph for further analysis.

Ensure Gmail and OpenAI credentials are configured before running the script.
