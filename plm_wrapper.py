from utils import *

from transformers import T5EncoderModel, T5Tokenizer, EsmModel, AutoTokenizer

PROTTRANS_MODEL_NAME = 'Rostlab/prot_t5_xl_uniref50'
ESM2_35M_MODEL_NAME = 'facebook/esm2_t12_35M_UR50D'
ESM2_150M_MODEL_NAME = 'facebook/esm2_t30_150M_UR50D'
ESM2_650M_MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'

def check_single_sequence(inputs):
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 1:
        raise ValueError("Expected exactly one sequence in a list, e.g. ['MKTAYIAKQR']")
    if not isinstance(inputs[0], str) or not inputs[0]:
        raise ValueError("Expected one nonempty protein sequence string")
    return inputs[0]

def resolve_device(device=None):
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

class ProtTransWrapper(nn.Module):
    def __init__(self, freeze=True):
        super(ProtTransWrapper, self).__init__()
        self.prot_trans = T5EncoderModel.from_pretrained(PROTTRANS_MODEL_NAME)
        if freeze:
            freeze_module(self.prot_trans)
        self.tokenizer = T5Tokenizer.from_pretrained(PROTTRANS_MODEL_NAME, do_lower_case=False)

    def tokenize(self, input_seqs):
        tokenized_seqs = self.tokenizer.batch_encode_plus(
            input_seqs, add_special_tokens=True,
            padding=False, truncation=False, return_tensors='pt'
        )
        for key in tokenized_seqs.keys():
            tokenized_seqs[key] = tokenized_seqs[key].to(self.prot_trans.device)
        return tokenized_seqs

    def forward(self, inputs, raw_seq=True, hidden_layer=None):
        assert(raw_seq)

        seq = check_single_sequence(inputs)
        inputs = [' '.join(x) for x in inputs]
        tokenized_seqs = self.tokenize(inputs)
        input_ids = tokenized_seqs['input_ids']
        mask = tokenized_seqs['attention_mask']

        outputs = self.prot_trans(
            input_ids=input_ids,
            attention_mask=mask,
            output_hidden_states=True
        )

        X = outputs.last_hidden_state if hidden_layer is None else outputs.hidden_states[hidden_layer]

        assert input_ids.shape == (1, len(seq) + 1)
        assert torch.all(mask == 1)
        X = X[:, :len(seq), :]

        return X
    
class ESMWrapper(nn.Module):
    def __init__(self, chkpoint, freeze=True):
        super().__init__()
        self.esm = EsmModel.from_pretrained(chkpoint, add_pooling_layer=False)
        if freeze:
            freeze_module(self.esm)
        self.tokenizer = AutoTokenizer.from_pretrained(chkpoint)

    def forward(self, inputs, raw_seq=True, hidden_layer=None, return_cls=False, return_both=False):
        assert raw_seq
        seq = check_single_sequence(inputs)

        tokenized_seqs = self.tokenizer([seq], return_tensors="pt", padding=False, truncation=False)
        input_ids = tokenized_seqs["input_ids"].to(self.esm.device)
        mask = tokenized_seqs["attention_mask"].to(self.esm.device)

        outputs = self.esm(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)

        X = (outputs.last_hidden_state if hidden_layer is None else outputs.hidden_states[hidden_layer])

        assert input_ids.shape == (1, len(seq) + 2)
        assert torch.all(mask == 1)

        residues = X[:, 1:len(seq) + 1, :]
        cls = X[:, :1, :]

        if return_both:
            return residues, cls
        if return_cls:
            return cls
        return residues


PLM = {
    "prottrans": lambda device=None: ProtTransWrapper().to(resolve_device(device)),
    "esm2-650M": lambda device=None: ESMWrapper(ESM2_650M_MODEL_NAME).to(resolve_device(device)),
}

PLM_dim = {'prottrans': 1024, 'esm2-650M': 1280}

N_LAYERS = {'prottrans': 24, 'esm2-650M': 33}
