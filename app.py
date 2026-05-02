import gradio as gr
import torch
import re
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

MODEL_PATH = "./model_files" 

def load_model():
    try:
        # Load config from local path
        config = PeftConfig.from_pretrained(MODEL_PATH)
        
        # Load base model (downloads from HF Hub automatically)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        
        # Load adapters from local path
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

print("Loading model...")
model, tokenizer = load_model()

device = "cuda" if torch.cuda.is_available() else "cpu"
if model:
    model = model.to(device)
    model.eval()
print(f"Model loaded on {device}")

# --- Helper Functions ---
def clean_output(text):
    if not text: return ""
    text = text[0].upper() + text[1:]
    text = re.sub(r'\.([a-zA-Z])', r'. \1', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def summarize_text(input_text):
    if not model:
        return "Error: Model not found."
        
    if not input_text:
        return "Please enter text."
    
    input_text = "summarize: " + input_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"], 
            max_new_tokens=256, 
            min_length=80, 
            num_beams=5, 
            length_penalty=2.0, 
            repetition_penalty=2.5, 
            early_stopping=True
        )
    
    raw_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_output(raw_summary)

# --- Interface ---
ex_med = """BACKGROUND: Hypertension remains a leading cause of cardiovascular morbidity and mortality globally. While standard antihypertensive therapies are effective, adherence remains suboptimal. This study evaluates the efficacy and safety of a novel, long-acting angiotensin receptor blocker (ARB), Telma-X, compared to standard Losartan in patients with mild-to-moderate essential hypertension.

METHODS: We conducted a multicenter, double-blind, randomized controlled trial involving 500 patients aged 40-65 years. Patients were randomized 1:1 to receive either Telma-X 80mg daily or Losartan 50mg daily for 12 weeks. The primary endpoint was the change in mean sitting systolic blood pressure (SBP) from baseline to week 12. Secondary endpoints included diastolic blood pressure (DBP) reduction and adverse event rates.

RESULTS: At week 12, the Telma-X group showed a significantly greater reduction in SBP compared to the Losartan group (-18.4 mmHg vs -12.1 mmHg; p < 0.001). DBP reduction was also superior in the Telma-X arm (-10.2 mmHg vs -7.5 mmHg; p = 0.02). The incidence of adverse events, primarily dizziness and fatigue, was comparable between groups (4.5% vs 4.2%).

CONCLUSIONS: Telma-X demonstrated superior efficacy in lowering systolic and diastolic blood pressure compared to Losartan, with a similar safety profile. These findings suggest Telma-X could be a valuable addition to the therapeutic arsenal for hypertension management."""

ex_leg = """SECTION 1. SHORT TITLE.
This Act may be cited as the "Nonprofit Safety and Accountability Act of 2024".

SEC. 2. LIMITATION ON LIABILITY FOR BUSINESS ENTITIES.
(a) Definitions.--In this section:
    (1) Business entity.--The term "business entity" means a firm, corporation, association, partnership, consortium, joint venture, or other form of enterprise.
    (2) Facility.--The term "facility" means any real property, including any building, improvement, or appurtenance.
    (3) Nonprofit organization.--The term "nonprofit organization" means any organization described in section 501(c)(3) of the Internal Revenue Code of 1986.

(b) Liability Protection.--Subject to subsection (c), a business entity shall not be subject to civil liability relating to any injury or death occurring at a facility of the business entity in connection with a use of such facility by a nonprofit organization if:
    (A) the use occurs outside of the scope of business of the business entity;
    (B) the nonprofit organization holds a valid certificate of insurance; and
    (C) the business entity authorized the use of the facility in writing.

(c) Exception.--Subsection (b) shall not apply to an injury or death that results from an act or omission of a business entity that constitutes gross negligence or intentional misconduct, including misconduct that constitutes a hate crime or a crime of violence."""

interface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=10, label="Input Text", placeholder="Paste Medical or Legal text here..."),
    outputs=gr.Textbox(lines=5, label="Summary"),
    title="StartUp.AI Summarizer",
    description="Fine-tuned on PubMed and BillSum using LoRA.",
    theme="default",
    examples=[[ex_med], [ex_leg]] if 'ex_med' in locals() else None
)

if __name__ == "__main__":
    interface.launch()
