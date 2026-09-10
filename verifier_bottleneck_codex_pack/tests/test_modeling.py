from vbexp.modeling import encode_supervised


class BoundaryTokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        class Encoded:
            input_ids = [ord(char) for char in text]

        return Encoded()


def test_supervised_labels_preserve_entire_answer():
    encoded = encode_supervised(BoundaryTokenizer(), "PROMPT", "PROGRAM: SH1", 100)
    labels = encoded["labels"].tolist()
    first_target = next(index for index, value in enumerate(labels) if value != -100)
    decoded = "".join(chr(value) for value in labels[first_target:])
    assert decoded == "PROGRAM: SH1<eos>"
