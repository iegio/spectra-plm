from utils import *

from transformers import BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer, EsmModel, AutoTokenizer
from sequence_models.pretrained import load_model_and_alphabet

PROTBERT_MODEL_NAME = 'Rostlab/prot_bert'
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
        if raw_seq:
            seq = check_single_sequence(inputs)
            inputs = [' '.join(x) for x in inputs]
            tokenized_seqs = self.tokenize(inputs)
            input_ids = tokenized_seqs['input_ids']
            mask = tokenized_seqs['attention_mask']
        else:
            input_ids = inputs[0].to(self.prot_trans.device)
            mask = inputs[1].to(self.prot_trans.device)

        outputs = self.prot_trans(
            input_ids=input_ids,
            attention_mask=mask,
            output_hidden_states=True
        )

        X = outputs.last_hidden_state if hidden_layer is None else outputs.hidden_states[hidden_layer]

        if raw_seq:
            assert input_ids.shape == (1, len(seq) + 1)
            assert torch.all(mask == 1)
            X = X[:, :len(seq), :]

        return X
    
class ESMWrapper(nn.Module):
    def __init__(self, chkpoint, freeze=True):
        super(ESMWrapper, self).__init__()
        self.esm = EsmModel.from_pretrained(chkpoint, add_pooling_layer=False)
        if freeze:
            freeze_module(self.esm)
        self.tokenizer = AutoTokenizer.from_pretrained(chkpoint)

    def forward(self, inputs, raw_seq=True, hidden_layer=None, return_cls=False):
        assert(raw_seq)
        if raw_seq:
            seq = check_single_sequence(inputs)
            tokenized_seqs = self.tokenizer(
                [seq],
                return_tensors="pt",
                padding=False,
                truncation=False
            )
            input_ids = tokenized_seqs['input_ids'].to(self.esm.device)
            mask = tokenized_seqs['attention_mask'].to(self.esm.device)
        else:
            input_ids = inputs[0].to(self.esm.device)
            mask = inputs[1].to(self.esm.device)

        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=mask,
            output_hidden_states=True
        )

        X = outputs.last_hidden_state if hidden_layer is None else outputs.hidden_states[hidden_layer]
        if raw_seq:
            assert input_ids.shape == (1, len(seq) + 2)
            assert torch.all(mask == 1)
            X = X[:, :1, :] if return_cls else X[:, 1:len(seq) + 1, :]
        return X

class CARPWrapper(nn.Module):
    def __init__(self, chkpoint="carp_640M", freeze=True):
        super(CARPWrapper, self).__init__()

        self.carp, self.collater = load_model_and_alphabet(chkpoint)
        self.n_layers = len(self.carp.model.embedder.layers)

        if freeze:
            freeze_module(self.carp)

    def tokenize(self, input_seqs):
        device = next(self.carp.parameters()).device
        batch = [(seq,) for seq in input_seqs]
        input_ids = self.collater(batch)[0]
        return input_ids.to(device)

    def forward(self, inputs, raw_seq=True, hidden_layer=None):
        assert(raw_seq)
        if raw_seq:
            seq = check_single_sequence(inputs)
            input_ids = self.tokenize(inputs)
            assert input_ids.shape == (1, len(seq))
        else:
            input_ids = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            input_ids = input_ids.to(next(self.carp.parameters()).device)

        if hidden_layer is None:
            hidden_layer = self.n_layers

        if not 1 <= hidden_layer <= self.n_layers:
            raise ValueError(f"hidden_layer must be between 1 and {self.n_layers}")

        outputs = self.carp(input_ids, repr_layers=[hidden_layer], logits=False)
        X = outputs["representations"][hidden_layer]

        if raw_seq:
            assert X.shape[:2] == (1, len(seq))

        return X


PLM = {
    'prottrans': lambda: ProtTransWrapper().to('cuda'),
    'esm2-35M': lambda: ESMWrapper(ESM2_35M_MODEL_NAME).to('cuda'),
    'esm2-150M': lambda: ESMWrapper(ESM2_150M_MODEL_NAME).to('cuda'),
    'esm2-650M': lambda: ESMWrapper(ESM2_650M_MODEL_NAME).to('cuda'),
    'carp-640M': lambda: CARPWrapper('carp_640M').to('cuda'),
}

PLM_dim = {
    'prottrans': 1024,
    'protbert': 1024,
    'esm2-35M': 480,
    'esm2-150M': 640,
    'esm2-650M': 1280,
    'carp-640M': 1280,
}