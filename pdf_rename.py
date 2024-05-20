# pyinstaller --recursive-copy-metadata langchain --recursive-copy-metadata openai --recursive-copy-metadata langchain_huggingface --recursive-copy-metadata langchain_huggingface --recursive-copy-metadata langchain-openai  ui6_llm.py
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pdf2image import convert_from_path
import pytesseract
import threading

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env file.

# Image pre-processing
from PIL import ImageEnhance

def preprocess_image(image):
    # Convert to grayscale
    gray = image.convert('L')

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(2.0)

    # Resize for better readability
    scale_factor = 1.5  # Scale factor (can be adjusted)
    resized = enhanced.resize((int(enhanced.width * scale_factor), int(enhanced.height * scale_factor)))

    # Apply thresholding
    return resized.point(lambda x: 0 if x < 140 else 255, '1')

def extract_text_from_pdf(pdf_path, lang_code):
    images = convert_from_path(pdf_path, first_page=0, last_page=5, poppler_path=os.getenv("POPPLER_BIN_PATH"))
    text = []
    for image in images:
        image = preprocess_image(image)
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
        custom_config = r'--oem 1 --psm 4 -l ' + lang_code
        data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
        
        for i, conf in enumerate(data['conf']):
            # Filter out text with very low confidence
            if float(conf) > 25:  # Confidence greater than 25%
                text.append(data['text'][i])
    
    return ' '.join(text)

def init_openai():
    repo = os.getenv("LLM_MODEL_OPENAI")
    llm = ChatOpenAI(model=repo)
    return llm

def init_huggingface():
    # see https://huggingface.co/blog/langchain
    # repo = "meta-llama/Meta-Llama-3-8B-Instruct"  # only Instruct can be executed with huggingface inspection API
    # using ChatHuggingFace is equivalent to
    # with mistralai/Mistral-7B-Instruct-v0.2
    # llm.invoke("<s>[INST] Hugging Face is [/INST]")
    # with meta-llama/Meta-Llama-3-8B-Instruct
    # llm.invoke("""<|begin_of_text|><|start_header_id|>user<|end_header_id|>Hugging Face is<|eot_id|><|start_header_id|>assistant<|end_header_id|>""")
    
    repo = os.getenv("LLM_MODEL_HUGGINGFACE")
    llm = HuggingFaceEndpoint(
        repo_id=repo,
        task="text-generation",
        temperature=0.5,
        do_sample=False,
    )
    llm_engine_hf = ChatHuggingFace(llm=llm)
    return llm_engine_hf


def generate_filename_from_text(llm, text, lang):
    
    # we just us a multi-line prompt string as prompt, no PromptTemplate
    prompt = """
    1. After the prompt instructions i will supply a text from a pdf document. 
    2. Generate a filename for this document based on the pdf content.
    3. Use {lang} language.
    4. The filename extension should be '.pdf'.
    5. The filename should not contain any special characters.
    6. The filename should not contain any spaces.
    7. The filename should be descriptive and relevant to the content.
    8. The filename should have a maximum of 100 characters.
    9. The filename should have the current date in the format 'YYYY-MM-DD' + '_' as a prefix.  Add '.pdf' at the end.
    10. Only answer with the filename. No additional text should be added. Even not an explanation.
    11. Here is the text extracted from the pdf document:
    {text}
    """.format(text=text, lang=lang)
    result = llm.invoke(prompt)
    return result.content

def update_progress_bar(processed, total, progress_var):
    progress_var.set((processed / total) * 100)

def rename_pdfs_in_directory(directory, lang, progress_var, total_files):
    # transform language code for tesseract
    language_mapping = {
        "German": "deu",
        "Spanish": "spa",
        "French": "fra",
        "English": "eng"
    }
    lang_code = language_mapping[lang]
    processed = 0
    for filename in os.listdir(directory):
        # if you want to rename only files starting with 'Scan' and ending with '.pdf', uncomment the following line
        # and comment the next line
        # if filename.endswith('.pdf') and filename.startswith('Scan'):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            try:
                text = extract_text_from_pdf(pdf_path, lang_code)
                new_filename = generate_filename_from_text(llm, text, lang)
                new_path = os.path.join(directory, new_filename)
                os.rename(pdf_path, new_path)
                print(f"Renamed '{filename}' to '{new_filename}'")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

            processed += 1
            root.after(0, update_progress_bar, processed, total_files, progress_var)
    
    root.after(0, lambda: messagebox.showinfo("Completion", "PDF Renaming Process Completed"))

def start_renaming():
    # if you want to rename only files starting with 'Scan' and ending with '.pdf', uncomment the following line
    # and comment the next line
    # total_files = sum(1 for f in os.listdir(directory.get()) if f.endswith('.pdf') and f.startswith('Scan'))
    total_files = sum(1 for f in os.listdir(directory.get()) if f.endswith('.pdf'))
    progress_var.set(0)
    threading.Thread(target=rename_pdfs_in_directory, args=(directory.get(), language.get(), progress_var, total_files)).start()

# init an LLM
chat_service = os.getenv('CHAT')
# Conditionally import modules based on the value of "CHAT" env variable
if chat_service == "HuggingFace":
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    llm = init_huggingface()
elif chat_service == "OpenAI":
    from langchain_openai import ChatOpenAI
    llm = init_openai()
else:
    raise ValueError("Unsupported chat service specified in the 'CHAT' environment variable.")

root = tk.Tk()
root.title("PDF Renamer")

directory = tk.StringVar()
language = tk.StringVar(value='German')
progress_var = tk.DoubleVar()

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

ttk.Label(frame, text="Select Directory:").grid(row=0, column=0, sticky=tk.W)
ttk.Entry(frame, textvariable=directory, width=50).grid(row=0, column=1)
ttk.Button(frame, text="Browse", command=lambda: directory.set(filedialog.askdirectory())).grid(row=0, column=2)

ttk.Label(frame, text="Select Language:").grid(row=1, column=0, sticky=tk.W)
language_combo = ttk.Combobox(frame, textvariable=language, values=['German', 'English','Spanish', 'French'])
language_combo.grid(row=1, column=1)
language_combo.state(['readonly'])

ttk.Button(frame, text="Start Renaming", command=start_renaming).grid(row=2, column=1)

progress_bar = ttk.Progressbar(frame, orient="horizontal", length=300, mode="determinate", variable=progress_var)
progress_bar.grid(row=3, column=0, columnspan=3)

for child in frame.winfo_children():
    child.grid_configure(padx=5, pady=5)

root.mainloop()
