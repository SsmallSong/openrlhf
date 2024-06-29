from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none, process_multi_turn_dialogue


def preprocess_data(data, input_template=None, input_key=None, apply_chat_template=None) -> str:

    print("hereherehereherh")
    prompt = data["prompt"]
    prompt = "\n<|user|>\n" +prompt+"\n<|assistant|>\n"
    input_template = None  # do not modified with input template again
    
    return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template="Human: {}\nAssistant: ",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        self.n_samples_per_prompt = getattr(self.strategy.args, "n_samples_per_prompt", 1)

        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length * self.n_samples_per_prompt

    def __getitem__(self, idx):
        return self.prompts[idx // self.n_samples_per_prompt]
