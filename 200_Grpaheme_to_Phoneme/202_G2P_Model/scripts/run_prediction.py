import sys
sys.path.append("../")
from dp.phonemizer import Phonemizer

if __name__ == '__main__':

    checkpoint_path = '../checkpoints/V1.0_EnUs_forward_6_512_2_8/latest_model.pt'
    phonemizer = Phonemizer.from_checkpoint(checkpoint_path)

    text = 'hello'

    result = phonemizer.phonemise_list([text], lang='EnUs')

    print(result.phonemes)
    for text, pred in result.predictions.items():
        tokens, probs = pred.phoneme_tokens, pred.token_probs
        for o, p in zip(tokens, probs):
            print(f'{o} {p}')
        tokens = ' '.join(tokens)
        print(f'{text} | {tokens} | {pred.confidence}')

