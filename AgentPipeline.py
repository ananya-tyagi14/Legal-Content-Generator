import os
from types import SimpleNamespace
from autogen import AssistantAgent, UserProxyAgent
from AutoGenClient import LlamaAutoGenClient
from hybridRetrieval import HybridRetrieval

class RetrievalAgent(UserProxyAgent):

    """
    An agent that handles user queries by retrieving relevant document chunks
    Inherits from UserProxyAgent to receive user messages and return retrieval results.
    """

    def __init__(self, name="retrieval_agent"):
        super().__init__(name=name, code_execution_config=False)
        
        self.retriever = None
        self.confirmed_results = []  #Stores (query, chunk) pairs confirmed by user


    def init(self, **kwargs):
        """
        Here, we instantiate the HybridRetrieval object without a query.
        
        """
        self.retriever = HybridRetrieval(input_query=None)


    def on_user_message(self, message):
        
        """
        Callback triggered when the user sends a message.
        Returns:
            A dict with role="system" and content containing the joined chunks.
        """
        self.retriever.query = message.content

        # Perform hybrid retrieval: retrieve top 5 semantic and top 3 BM25 matches
        chunks = self.retriever.run(top_k_sem=5, top_k_bm25=3)
        return {"role": "system", "content": "\n\n".join(chunks)}

    def retrieve(self, query):

        """
        Helper method to run a retrieval given a plain-string query.
        Args:
            query (str): The query string to retrieve documents for.
        Returns:
            List[str]: Retrieved text chunks.
        """

        msg = SimpleNamespace(content=query)
        resp = self.on_user_message(msg)

        return resp["content"].split("\n\n")

    def confirm(self, query, chunk):
        self.confirmed_results.append({"query": query, "chunk": chunk}) #stores approved query-chunk pairs

    def get_confirmed(self):
        return self.confirmed_results  #returns a list of approved query-chunk pairs

    

class ContentAgent(AssistantAgent):
    """
    An agent responsible for generating content (e.g., introductions or statements)
    Inherits from AssistantAgent to respond to system/user messages.
    """
    def __init__(self, name="content_agent"):
        super().__init__(name=name, code_execution_config=False)
        self.client = None   #  LlamaAutoGenClient instance
        self.extracted_data = None   # Stores the retrieved text chunks

        self.topic = ""
        self.max_subsections = 3   #Maximum subsections to consider
        self.mode = "Write Intro"  #Mode of generation

        self.output = ""        # Stores the generated output text


        
    def set_generation_params(self, mode, topic, max_subsections):

        """
        Configure the generation parameters based on the mode, topic, and max_subsections.

        Initializes a LlamaAutoGenClient with appropriate settings.
        """
        
        self.mode = mode
        self.topic = topic
        self.max_subsections = max_subsections

        # Choose token budget and temperature based on mode
        if self.mode == "Write Intro":
            tokens, temp = 320, 0.7

        elif self.mode == "Write Pre-Law statement":
            tokens, temp = 210, 0.7

        else:
            tokens, temp = 480, 0.7


        self.client = LlamaAutoGenClient(
            model_path="./Llama-3.2-1B",
            hf_token="",
            max_new_tokens = tokens,
            temperature=temp,
            do_sample=True,
        )


    def on_system_message(self, message):

        """
        Callback when the system (e.g., RetrievalAgent) sends data to this agent.
    
        Args:
            message: A dict-like object with a 'content' attribute containing retrieved text.
        """

        full = message.content
        parts = full.split("\n\n") 
        self.extracted_data =  parts[0].strip() if len(parts) > 1 else full
        

    def on_user_message(self, message):

         """
        Callback when the user sends a message to the ContentAgent.

        Args:
            message: A dict-like object with a 'content' attribute (not used directly here).
        Returns:
            dict: {"role": "assistant", "content": generated_text}
        """

        lines = self.extracted_data.splitlines()
        formatted_parts = []
        
        for line in lines:

            stripped = line.strip()

            if stripped.startswith("Subsection:"):

                heading = stripped.split(":", 1)[1].strip()+ ":"
                formatted_parts.append("\n\n") 
                formatted_parts.append(heading)
                formatted_parts.append("\n")
                continue

            # List item at first level: indent with two spaces
            if stripped.startswith("•"):
                formatted_parts.append(f"  {stripped}")
                formatted_parts.append("\n")

            # Nested list item: indent with five spaces
            elif stripped.startswith("*"):
                formatted_parts.append(f"     {stripped}")
                formatted_parts.append("\n")

            # Regular line: prefix with a dash
            else:
                formatted_parts.append(f"- {stripped}")
                formatted_parts.append("\n")

        # Join all formatted parts into one prompt-ready string
        formatted_data = "".join(formatted_parts)

        #wrap formatted_data into the final instruction
        prompt = self.prompt_templates(formatted_data)

        # Call the LlamaAutoGenClient to generate the desired content
        self.output = self.client.chat(prompt)

        return {"role": "assistant", "content": self.output}
    


    #holds the 3 different types of prompts depending on the user's chosen mode
    def prompt_templates(self, formatted_data):

        intro_prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a friendly, professional copywriter for Pre-Law. Write a engaging, formal Introduction section on '{self.topic}'. Do not write the body or conclusion.          
                        
Write for a General Audience with little or no legal background:
1. **Definition/Hook** (1–2 sentences): introduce the topic and hook the reader.
2. **Overview** (1–2 sentences): explain what the article will cover in general terms.
3. **How Pre-Law can help** (Exactly 1 sentence): a concise call-out of Pre-Law’s service (“At Pre-Law, …”). If “At Pre-Law…” already appears anywhere in the context above, do NOT repeat it-skip this step.
                   
Use plain English, avoid legal jargon. Do not use first-person singular (“I”, “my”).
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Topic: Settlement Agreements
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

**Introduction**

Settlement agreements are often used when an employment relationship is coming to an end - whether through redundancy, a negotiated exit, or a workplace dispute. 
If you’re leaving your job and have been presented with a settlement agreement, it’s important to understand what it is, what it means and what you are agreeing to.

At Pre-Law, we provide clear, fixed-fee legal advice to help you understand the terms of your settlement agreement, protect your rights, and move forward.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Topic: {self.topic}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

                        """


        article_prompt = f"""

                        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
                        You are a well-polished, engaging article writer for a UK solicitor firm called Pre-Law. Do not write more than {self.max_subsections} subsections.
                        
                        Write for a General Public with no legal background:
                        - Paraphrase and use plain English, translate any legal terminology into everyday language.
                        - Avoid legal jargon or citations of case law; if you must mention a legal term, define it immediately in simple words.
                        - Don't repeat phrases.
                        - Don't use first-person singular (“I”, “my”).
                        - Don’t introduce any new legal facts or cases beyond what’s given.
                        - Ensure what you write is accurate. 
                        
                        Use subheadings to ensure a clean article structure.
                        <|eot_id|>
                        <|start_header_id|>user<|end_header_id|>
                        Here is some legal information on “{self.topic}”:

                        {formatted_data}

                        -----
                        In accordance with English law as of May 2025, write an engaging, clear, informative article about “{self.topic}” using only the legal information provided. Mention every fact in the legal information.  Make sure to:
                        1. Give the article a Title  
                        2. Include an introduction (2-3 sentences) that:  
                           - Hook the reader with a conversational first sentence. 
                           - speaks in generalities about the topic (e.g. “{self.topic} encompasses several key aspects…”),  
                           - does not enumerate the specific items you’ll list below.  
                        3. Include a simple, well structured Article body:
                            - Use bullet points where needed to list items.
                            - Do not write more than {self.max_subsections} subsections.
                            - Don't go into extensive detail.
                        4. Definitely include a Conclusion (1-2 sentences):
                            - Sub-heading: `**Conclusion**`  
                            - A short wrap-up sentence that ends with a period, **and** includes an “At Pre-Law…” statement connecting your firm to the topic (e.g. “At Pre-Law, we ensure that…”).  

                        <|eot_id|>
                        <|start_header_id|>assistant<|end_header_id|>
                        
                        """

        company_prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a friendly, professional copywriter for Pre-Law. write a brief “How Pre-Law Can Help” section:
- Start with this heading **How Pre-Law Can Help**
- Use clear, supportive language
- Describe 2–3 core services or benefits
- End with a call to action including phone, email, and online enquiry form
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Topic: Settlement Agreements and Redundancy
Phone: 01524 907100
Email: info@pre-law.co.uk
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
**How Pre-law Can Help**
                            
Whether you’re being made redundant, considering a severance package, or have been offered a settlement agreement -we’re here to support you.

We’ll explain your rights, review the terms, and, if necessary, help you negotiate the best outcome for your situation.

For more information, contact us on 01524 907100, email info@pre-law.co.uk or fill out our online enquiry form.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Topic: {self.topic}
Phone: 01524 907100
Email: info@pre-law.co.uk
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

                             """


        if self.mode == 'Write Intro':
            return intro_prompt
        elif self.mode == 'Write Pre-Law statement':
            return company_prompt
        else:
            return article_prompt
        
