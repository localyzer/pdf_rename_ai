## Introduction
I almost do scan every print document (mostly with Android > OneNote > New document > Scan). Those documents normally get a name like "Scan-2023-12-31 12:33.pdf". So, you never could assume the content of file based on it's filename.
This program tries to find a reasonable name based on the content of the document.
It uses 
- Poppler's **pdf2image** to convert pdf to images
- OCR (**Tesseract**) to identify the text in the images
- OpenAI or Huggingface to identify a reasonable filename - based on the content of the file

This script is intended to run on Windows platform because for all office related tasks like scanning we work on Windows.

**Note:**  in latest versions of OpenAI, we could directly pass a pdf file. Instead, we chose to scan the pdf file first. We did that due to missing support of some models to process a pdf and to reduce overhead / cost.


## Installation

### Install Tesseract on Windows

https://github.com/UB-Mannheim/tesseract/wiki

- Execute Windows installer
- Select languages german and english during installation
- Check languages:
```shell
tesseract --list-langs
```

#### Install Tesseract's best language model

Download https://github.com/tesseract-ocr/tessdata_best/blob/main/deu.traineddata as raw
Create folder "best" in C:\Program Files\Tesseract-OCR\tessdata
Move deu.traineddata to C:\Program Files\Tesseract-OCR\tessdata\best

Select best/deu in program:

```python
lang_code = 'best/deu' if lang == 'German' else 'eng'
```

### Install Poppler on Windows

https://github.com/oschwartz10612/poppler-windows/releases/

Extract the popper folder (e.g. poppler-23.11.0) an move it to C: or any other place.


### Add to Windows PATH environment

C:\poppler-23.11.0\Library\bin
C:\poppler-23.11.0\Library\share
C:\Program Files\Tesseract-OCR

### Create .env file

copy .env.example to .env and adjust the paths to your Poppler and Tesseract installation directories.
Latest script files include the path to Poppler and Tesseract explicitly in the script file.
Those paths are specified in the .env file.

**Note:** you need an OpenAI API key or HuggingFace API Token. Please register an account with at least one of those. Then set the corresonding variables in .env file.

### Create Python venv in your project directory

```shell
python -m venv env
.\env\Scripts\Activate.ps1
```

### Install Python packages


```shell
pip install tk python-dotenv pytesseract pdf2image pdf2image openai==1.29.0 langchain==0.1.16 langchain-openai huggingface_hub langchain_huggingface
```
### Execute

```
python pdf_rename.py
```


### Quality

We got best results with OpenAI. You might play around with HuggingFace models.
But you might have to create a HuggingFace Space.

### Pyinstaller

We could create a Windows executable with Pyinstaller:

```python
pip install pyinstaller
pyinstaller --recursive-copy-metadata langchain --recursive-copy-metadata openai --recursive-copy-metadata langchain_huggingface --recursive-copy-metadata langchain_huggingface --recursive-copy-metadata langchain-openai  pdf_rename.py
```

Then cd to folder "dist/pdf_rename" and execute pdf_rename.exe
