from transformers import AutoTokenizer, AutoModel
import torch


class AIEmbed:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
        self.model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embeddings(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, max_length=24,
                                       return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings
