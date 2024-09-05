import numpy


class Generator():

    def _init_transformers_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if self.file_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, gguf_file=self.file_name, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, gguf_file=self.file_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path)

    def _init_ollama(self) -> None:
        import ollama

    def _init_server(self) -> None:
        if not self.url:
            raise ValueError(
                'you need to pass in the url of the server alongside the model name.')

    def __init__(self, model_name_or_path: str, is_ollama: bool = False, is_server: bool = False, file_name: str = '', url: str = '', api: str = '') -> None:
        self.model_name_or_path = model_name_or_path
        self.file_name = file_name
        self.is_ollama = is_ollama
        self.is_server = is_server
        self.url = url
        self.api = api

        if not self.is_server and not self.is_ollama:
            self._init_transformers_model()

        elif self.is_ollama:
            self._init_ollama()

        elif self.is_server:
            self._init_server()

    # TODO: add generation hyperparamters

    def _generate_transformers(self, prompt: str) -> str:
        chat = [
            {"role": "user", "content": prompt},
        ]
        input_text = self.tokenizer.apply_chat_template(chat, tokenize=False)
        input_ids = self.tokenizer.encode(
            input_text, return_tensors="pt").to("cuda")
        attention_mask = input_ids.new_ones(input_ids.shape)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        output = self.model.generate(input_ids, attention_mask=attention_mask,
                                     max_length=50, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        clean_response = response[len(input_text)-2:]
        return clean_response

    def _generate_ollama(self, prompt: str) -> str:
        import ollama
        response = ollama.chat(
            model=self.model_name_or_path,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return response['message']['content']

    def _generate_server(self, prompt: str) -> str:
        import openai
        client = openai.OpenAI(
            api_key=self.api,
            base_url=self.url
        )
        completion = client.chat.completions.create(
            model=self.model_name_or_path,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        reply = completion.choices[0].message.content
        if reply:
            return reply
        else:
            return ''

    def generate(self, prompt: str) -> str:
        if self.is_ollama:
            return self._generate_ollama(prompt)
        elif self.is_server:
            return self._generate_server(prompt)
        else:
            return self._generate_transformers(prompt)


class Encoder():

    def __init__(self, model_name_or_path) -> None:
        from sentence_transformers import SentenceTransformer
        self.model_name_or_path = model_name_or_path
        self.model = SentenceTransformer(model_name_or_path)

    def encode(self, sentence: str) -> numpy.ndarray:
        embedding = self.model.encode(sentence)
        return embedding
