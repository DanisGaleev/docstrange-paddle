"""Neural Document Processor using Nanonets OCR for superior document understanding."""

import logging
import os
from typing import Optional
from pathlib import Path
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class NanonetsDocumentProcessor:
    """Neural Document Processor using Nanonets OCR model."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the Neural Document Processor with Nanonets OCR."""
        logger.info("Initializing Neural Document Processor with Nanonets OCR...")

        # Initialize models
        self._initialize_models(cache_dir)

        logger.info("Neural Document Processor initialized successfully")

    def _initialize_models(self, cache_dir: Optional[Path] = None):
        """Initialize Nanonets OCR model from local cache."""
        try:

            print("333B")
            from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(
                "nanonets/Nanonets-OCR2-3B",
                dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="sdpa",# or "eager"
            )
            self.model.eval()

            self.model = torch.compile(self.model)

            self.tokenizer = AutoTokenizer.from_pretrained(
                "nanonets/Nanonets-OCR2-3B",

            )
            self.processor = AutoProcessor.from_pretrained(
                "nanonets/Nanonets-OCR2-3B",

            )

            logger.info("Nanonets OCR model loaded successfully from local cache nanonets/Nanonets-OCR2-3B")

        except ImportError as e:
            logger.error(f"Transformers library not available: {e}")
            raise ImportError(
                "Transformers library is required for Nanonets OCR. "
                "Please install it: pip install transformers"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Nanonets OCR model: {e}")
            raise

    def extract_text(self, image_path: str) -> str:
        """Extract text from image using Nanonets OCR."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return ""

            return self._extract_text_with_nanonets(image_path)

        except Exception as e:
            logger.error(f"Nanonets OCR extraction failed: {e}")
            return ""

    def extract_text_with_layout(self, image_path: str) -> str:
        """Extract text with layout awareness using Nanonets OCR.
        
        Note: Nanonets OCR already provides layout-aware extraction,
        so this method returns the same result as extract_text().
        """
        return self.extract_text(image_path)

    def _extract_text_with_nanonets(self, image_path: str, max_new_tokens: int = 4096) -> str:
        """Extract text using Nanonets OCR model."""
        try:
            prompt = """Извлеките текст из приведённого выше документа так, как если бы вы читали его естественно. Возвращайте таблицы в формате HTML. Возвращайте уравнения в представлении LaTeX. Если в документе есть изображение и подпись к изображению отсутствует, добавьте небольшое описание изображения внутри тега <img></img>; в противном случае поместите подпись к изображению внутри <img></img>. Водяные знаки должны быть заключены в скобки. Например: <watermark>OFFICIAL COPY</watermark>. Номера страниц должны быть заключены в скобки. Например: <page_number>14</page_number> или <page_number>9/22</page_number>. Для флажков предпочтительно использовать ☐ и ☑️."""

            image = Image.open(image_path)
            messages = [
                {"role": "system", "content": "Ты полезный ассистент."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt},
                ]},
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]

            output_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            return output_text[0]

        except Exception as e:
            logger.error(f"Nanonets OCR extraction failed: {e}")
            return ""

    def __del__(self):
        """Cleanup resources."""
        pass
