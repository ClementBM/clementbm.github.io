---
layout: post
title:  "Struggling with text content extraction from pdf files"
excerpt: "How to get quality text content from pdf files without losing formating"
date:   2023-03-16
categories: [coding]
tags: [nlp, content extraction]
---

- [Requirements](#requirements)
- [Processing library](#processing-library)
- [Raw extraction](#raw-extraction)
- [Libraries for parsing pdf files](#libraries-for-parsing-pdf-files)
  - [Docling](#docling)
- [Libraries for extracting web contents](#libraries-for-extracting-web-contents)
- [Document AI](#document-ai)
- [Sources](#sources)


extracting metadata from PDF documents: Science-Parse by Allen AI


# Requirements
Get a clean textual content out of a pdf file, which means:
* Paragraphs
* Sections
* Titles
* References
* Hyperlinks

Processing time is not a requirement in my case.

# Processing library
https://github.com/hi-primus/optimus

# Raw extraction


# Libraries for parsing pdf files
* pypdfium2 (https://towardsdatascience.com/how-to-extract-text-from-any-pdf-and-image-for-large-language-model-2d17f02875e6?gi=c64a12cdc71f)
* Deep Search IBM
* [Docling](https://github.com/DS4SD/docling)

## Docling
Docling simplifies document processing, parsing diverse formats — including advanced PDF understanding — and providing seamless integrations with the gen AI ecosystem.

# Libraries for extracting web contents

* [html2text](https://github.com/Alir3z4/html2text) - Convert HTML to Markdown-formatted text.
* [lassie](https://github.com/michaelhelmick/lassie) - Web Content Retrieval for Humans.
* [micawber](https://github.com/coleifer/micawber) - A small library for extracting rich content from URLs.
* [newspaper](https://github.com/codelucas/newspaper) - News extraction, article extraction and content curation in Python.
* [python-readability](https://github.com/buriy/python-readability) - Fast Python port of arc90's readability tool.
* [requests-html](https://github.com/psf/requests-html) - Pythonic HTML Parsing for Humans.
* [textract](https://github.com/deanmalmgren/textract) - Extract text from any document, Word, PowerPoint, PDFs, etc.


* camelot: pas très bon
* tabula: pas très bon
# Document AI

https://huggingface.co/blog/document-ai
There are at least six general use cases for building document AI solutions. These use cases differ in the kind of document inputs and outputs. A combination of approaches is often necessary when solving enterprise Document AI problems.

# Sources
* https://pdfminersix.readthedocs.io/en/latest/tutorial/highlevel.html
* https://github.com/py-pdf/benchmarks
* trends/explore?q=pdf to text,pdf to text python&hl=fr
* https://github.com/chrismattmann/tika-python
* https://superuser.com/questions/198392/how-to-copy-text-out-of-a-pdf-without-losing-formatting
* https://github.com/pymupdf/PyMuPDF
* pdfplumber
* InftyReader