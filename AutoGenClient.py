import getpass
from typing import Optional

from huggingface_hub import login
from langchain_core.messages import HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline


class LlamaAutoGenClient:

    # Class-level attributes to cache loaded tokenizer and model
    _tokenizer = None
    _model = None

    def __init__(
        self,
        model_path: str = "./Llama-3.2-1B",
        hf_token: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        do_sample: bool = True,
    ):

        # 1. Authenticate to Hugging Face
        token = hf_token or getpass.getpass("Hugging Face token: ")
        login(token=token)

        if LlamaAutoGenClient._tokenizer is None:
            # 2a. Load tokenizer and model from local folder
            LlamaAutoGenClient._tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            LlamaAutoGenClient._model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)    
            
        
        # 2b. Build a transformers text-generation pipeline
        transformers_pipe = hf_pipeline(
            "text-generation",
            model=LlamaAutoGenClient._model,
            tokenizer=LlamaAutoGenClient._tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            return_full_text=False,
            device=-1,
        )
        
        # 2c. Wrap in LangChain's HuggingFacePipeline
        self.pipeline = HuggingFacePipeline(pipeline=transformers_pipe)
        # 2d. Wrap that in ChatHuggingFace
        self.llm = ChatHuggingFace(llm=self.pipeline, model_id=model_path)
       
    

    def chat(self, prompt: str) -> str:
        """Invoke the model directly—no agents, no feedback loop."""
        # We wrap your prompt in a single HumanMessage
        out = self.llm.invoke([HumanMessage(content=prompt)])
        reply = out.content
        return reply




    
