<div align="center">

# 📄 DocXtract

### End-to-end Document OCR & Structured Data Extraction Pipeline

*Turn scanned documents and PDFs into clean, structured JSON — locally, offline, and transparently.*

[![Live Demo](https://img.shields.io/badge/demo-live-4CD3F0?style=for-the-badge)](https://docxtract1.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Tesseract OCR](https://img.shields.io/badge/Tesseract-OCR-4B8BBE?style=for-the-badge&logo=googlelens&logoColor=white)](https://github.com/tesseract-ocr/tesseract)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Unspecified-lightgrey?style=for-the-badge)](#license)

**[🚀 Try the live app →](https://docxtract1.streamlit.app/)**

</div>

---

## ✨ Overview

**DocXtract** accepts document images and PDFs, extracts text using **Tesseract OCR**, identifies the document type using a **keyword-based scoring mechanism**, and applies document-specific **regular-expression extraction rules** to convert unstructured OCR text into structured fields.

It ships with an interactive **Streamlit interface** for uploading documents, running the pipeline, and reviewing extracted results — no cloud OCR API required.

<br>

<div align="center">

| 🔍 OCR | 🧭 Classification | 🧩 Extraction | 🖥️ UI |
|:---:|:---:|:---:|:---:|
| Tesseract-powered text recognition for images & PDFs | Lightweight keyword scoring, no ML model needed | Regex + heuristics tuned per document type | Streamlit workspace with JSON export |

</div>

---

## 🚀 Features

- 🔎 OCR for images and PDF documents
- 🏷️ Automatic document classification
- 🧩 Document-specific structured field extraction
- 📚 Support for multiple document categories
- 🖥️ Interactive Streamlit interface
- 💾 JSON export of extracted fields
- 📝 OCR text inspection
- 🕘 Recent processing history in the Streamlit interface

---

## 📋 Supported Documents

The current pipeline is configured to recognize the following document types. The classifier assigns scores to each supported type based on characteristic keywords found in the OCR output, then selects the highest-scoring category.

| Document Type | Example Extracted Fields |
|---|---|
| 🪪 **Driving License** | DL Number, Name, DOB |
| 🛂 **Passport** | Passport Number, Name, Country |
| 💼 **W-2** | EIN, Year, Employee Name |
| 💵 **Paystub** | Net Pay, Employee Name, Employer Name |
| 🌊 **Flood Certificate** | Borrower Name, Customer Number, Expiration Date |
| ❔ **Others** | Returned when no supported document type is detected |

---

## 🔄 How It Works

```
                    ┌──────────────────┐
                    │   PDF / Image     │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Tesseract OCR   │
                    │  Text Extraction  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │    Document       │
                    │  Classification   │
                    │ (Keyword Scoring) │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Document-Specific│
                    │  Field Extraction │
                    │  (Regex / Rules)  │
                    └─────────┬─────────┘
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
          ┌───────────────┐      ┌────────────────┐
          │  Structured    │      │   OCR Text      │
          │  JSON Fields   │      │   Output        │
          └───────────────┘      └────────────────┘
```

> ℹ️ For PDFs, the current OCR implementation converts **only the first page** to an image before sending it to Tesseract.

---

## 🏗️ Architecture

### 1️⃣ OCR Layer

The OCR layer is implemented in `main.py`.

**Supported input formats:**

`PNG` · `JPG` / `JPEG` · `BMP` · `TIFF` · `PDF`

Images are processed directly with Pillow and Tesseract. PDFs are converted to an image using `pdf2image` before OCR processing.

```python
ocr_file(path)
```
Returns the raw OCR text extracted from the document.

<br>

### 2️⃣ Document Classification

After OCR, the extracted text is normalized and passed to:

```python
classify_document(text)
```

The classifier maintains a score for each supported document category and increments the score whenever a matching keyword is found. For example:

```
Passport
    ↓
"passport" · "passport number" · "surname" · "given name" · "nationality" ...
```

The document type with the highest score is selected. If no supported category receives a score, the document is classified as `Others`.

> This logic is intentionally lightweight and does not require a machine-learning classification model.

<br>

### 3️⃣ Structured Field Extraction

Once the document type is determined, the pipeline dispatches the OCR text to a document-specific extraction function:

```python
extract_fields(doc_type, text)
```

The dispatcher routes the document to functions such as:

```
extract_driving_license()
extract_passport()
extract_w2()
extract_paystub()
extract_flood()
```

These functions use regular expressions, normalization, and document-specific heuristics to extract relevant fields.

<details>
<summary><b>🪪 Driving License</b></summary>
<br>

- DL number
- Name
- DOB

</details>

<details>
<summary><b>🛂 Passport</b></summary>
<br>

- Passport number
- Country
- Name

Passport name extraction can use the MRZ when available, with additional fallback strategies for printed passport fields.

</details>

<details>
<summary><b>💼 W-2</b></summary>
<br>

- EIN
- Year
- Employee Name

</details>

<details>
<summary><b>💵 Paystub</b></summary>
<br>

- Net Pay
- Employee Name
- Employer Name

</details>

<details>
<summary><b>🌊 Flood Certificate</b></summary>
<br>

- Borrower name
- Customer No
- Expire date

</details>

---

## 🖥️ Streamlit Application

The Streamlit application provides an interactive **Document Intelligence Workspace**. Users can:

1. 📤 Upload a PDF or image
2. ⚙️ Run OCR automatically
3. 🏷️ Detect the document type
4. 👀 View extracted fields
5. 📝 View the complete OCR text
6. 💾 Download the extracted fields as JSON
7. 🕘 Review recent processing runs

The Streamlit application directly reuses the processing functions from `main.py`.

**Run it:**

```bash
streamlit run streamlit_app.py
```

**Accepted file types:** `PNG` · `JPG` · `JPEG` · `BMP` · `TIFF` · `TIF` · `PDF`

---

## 📁 Project Structure

```
DocXtract/
│
├── main.py
├── streamlit_app.py
│
├── documents-used/
│   ├── Doc1.jpg
│   ├── Doc2.jpg
│   ├── Doc3.png
│   ├── Doc4.pdf
│   └── Doc5.pdf
│
└── smaple image.png
```

> The repository includes five sample documents representing the supported document categories.

---

## ⚙️ Requirements

### 🐍 Python

Use a recent Python 3 installation, then install dependencies:

```bash
pip install -r requirements.txt
```

### 🔤 Tesseract OCR

This project requires **Tesseract OCR** to be installed separately on the system. Python package installation alone is not enough — the `tesseract` system binary must also be installed.

**Configuration options:**

- ✅ **Recommended:** make `tesseract` available on your system `PATH`
- 🔧 **Optional:** set `TESSERACT_CMD` to the absolute path of the `tesseract` executable
- 🪟 **Windows:** default install location works out of the box — `C:\Program Files\Tesseract-OCR\tesseract.exe`

> **Streamlit Cloud:** configure the system dependency in your deployment (e.g. via apt packages) and, if needed, set `TESSERACT_CMD` in app secrets/environment.

### 📄 PDF Support

PDF processing uses:

```python
from pdf2image import convert_from_path
```

The current implementation converts the **first page** of a PDF into an image and then sends that image to Tesseract.

> `pdf2image` also requires **Poppler** system utilities (`pdftoppm`) on the host machine. Ensure Poppler is installed and available on `PATH` (including Linux/Streamlit Cloud environments).

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/ShekhawaTTiku/DocXtract.git

# 2. Move into the project
cd DocXtract

# 3. Create a virtual environment
python -m venv venv

# 4. Activate it
#    Windows:
venv\Scripts\activate
#    Linux / macOS:
source venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt
```

Finally, install and configure Tesseract OCR on the host machine. If it is not on `PATH`, set `TESSERACT_CMD` to its absolute binary path.

---

## ▶️ Usage

### Process the sample documents with Python

The command-line pipeline can process the documents configured in `DOCS_FOLDER`:

```bash
python main.py
```

The script processes each file, performs OCR, determines the document type, and prints the extracted fields.

### Launch the Streamlit UI

```bash
streamlit run streamlit_app.py
```

Then upload a supported document through the browser interface. The application displays:

```
Detected Document Type  +  Extracted Fields  +  Raw OCR Text  +  JSON Download
```

---

## 📤 Example Output

A processed document is converted into a structure similar to:

```json
{
  "Name": "John Doe",
  "DOB": "01/01/1990",
  "DL number": "D123456789"
}
```

> The exact fields depend on the detected document type. The Streamlit application also provides a button to download the extracted fields as a JSON file.

---

## 🎯 Design Decisions

<table>
<tr>
<td width="33%" valign="top">

### Why Tesseract?

Tesseract provides a lightweight local OCR engine without requiring a cloud OCR API — suitable for local development, offline processing, prototyping, and privacy-sensitive document workflows.

</td>
<td width="33%" valign="top">

### Why keyword-based classification?

The classification stage is intentionally simple and transparent. Instead of a trained model, the pipeline uses document-specific keywords and scoring rules — easy to inspect and modify for new categories.

</td>
<td width="33%" valign="top">

### Why regex-based extraction?

Each supported document type has a known structure and a small set of fields. Regular expressions and heuristics transform OCR text into structured data without a separate NLP model.

</td>
</tr>
</table>

---

## ⚠️ Limitations

The current version is primarily a **prototype / task-oriented** document extraction pipeline.

| Area | Limitation |
|---|---|
| **PDF processing** | Only the first page is processed (`convert_from_path(path, first_page=1, last_page=1)`) |
| **Document classification** | Based on predefined keywords rather than a trained ML model; poor OCR quality or unusual layouts may cause misclassification |
| **Field extraction** | Rules rely heavily on expected text patterns; OCR errors, layout changes, or different templates can cause fields to return `None` or extract incorrectly |
| **Configuration** | Some paths are hard-coded for the original Windows development environment, including the Tesseract executable path and document directory |

---

## 🛣️ Future Improvements

- 📄 Multi-page PDF processing
- 🧹 OCR preprocessing for noisy or rotated documents
- 📊 Confidence scoring for OCR and extracted fields
- 🤖 Better document classification using ML / transformer models
- 🗂️ Layout-aware document understanding
- ➕ Support for additional document types
- 🧾 Configurable extraction schemas
- 🌱 Environment-variable based configuration
- 🔌 API endpoints for programmatic document processing
- ✅ Improved validation of extracted identifiers
- 🐳 Dockerized deployment
- 🔐 Authentication and access control

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) | Core implementation |
| ![Tesseract](https://img.shields.io/badge/-Tesseract_OCR-4B8BBE?style=flat-square) | Text recognition |
| **pytesseract** | Python interface for Tesseract |
| **Pillow** | Image loading and processing |
| **pdf2image** | PDF-to-image conversion |
| ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) | Interactive document-processing UI |
| **Regular Expressions** | Structured field extraction |

---

## 🧪 Sample Documents

The repository contains a `documents-used` directory with sample documents for testing the pipeline:

```
Doc1.jpg    Doc2.jpg    Doc3.png    Doc4.pdf    Doc5.pdf
```

---

## 🧭 Pipeline Summary

```
Input Document
      │
      ▼
┌───────────────┐
│      OCR       │
│   Tesseract    │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Classification │
│ Keyword Score  │
└───────┬───────┘
        │
        ▼
┌────────────────┐
│ Field Extractor │
│  Regex + Rules  │
└───────┬────────┘
        │
        ▼
┌──────────────────────┐
│  Structured Document  │
│      Data / JSON      │
└───────────┬──────────┘
            │
            └──────────────► Streamlit UI
```

---

## 👤 Author

**Digvijay Singh Shekhawat**

[![GitHub](https://img.shields.io/badge/GitHub-@ShekhawaTTiku-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ShekhawaTTiku)

---

## 📜 License

No license is currently specified in the repository.

> Add a `LICENSE` file if you intend to distribute the project under an open-source license.

---

<div align="center">

*Built with 🐍 Python, 🔤 Tesseract, and 🖥️ Streamlit*

</div>
