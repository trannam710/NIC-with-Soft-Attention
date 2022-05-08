import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import argparse

from evalfunc.bleu.bleu import Bleu
from evalfunc.rouge.rouge import Rouge
from evalfunc.cider.cider import Cider

# Data parameters
data_name = 'uitviic_5_cap_per_img_5_min_word_freq'

# Model parameters
emb_dim = 512
attention_dim = 512
decoder_dim = 512 
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
fine_tune_encoder = False
encoder_lr = 1e-4 
decoder_lr = 4e-4 
batch_size = 128
workers = 0



def main():
    global best_bleu4, epochs_since_improvement, start_epoch, fine_tune_encoder, data_name, word_map

    parser = argparse.ArgumentParser(description='Show, Attend and Tell')
    parser.add_argument('--pre_train', type=str2bool)
    parser.add_argument('--tokenize_level', type=str)
    args = parser.parse_args()

    data_folder = './caption data %s' % args.tokenize_level

    if args.pre_train:
        checkpoint = './BEST_%s_pretrain_checkpoint_' % args.tokenize_level + data_name + '.pth.tar'
    else:
        checkpoint = './BEST_%s_no_pretrain_checkpoint_' % args.tokenize_level + data_name + '.pth.tar'
    

    # Read word map
    word_map_file = os.path.join(data_folder, 'word_map_' + args.tokenize_level + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    assert checkpoint is not None

    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    
    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    valtestset(test_loader=test_loader,
                encoder=encoder,
                decoder=decoder,
                criterion=criterion)

def valtestset(test_loader, encoder, decoder, criterion):
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(tqdm(test_loader, desc='EVALUATING ...')):
            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, *_ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, *_ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU & CIDEr & ROUGE scores
        scorers = [(Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
                    (Rouge(), "ROUGE_L")]

        hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
        ref = [[' '.join(reft) for reft in reftmp] for reftmp in [[[str(x) for x in reft] for reft in reftmp]for reftmp in references]]

        score = []
        method = []
        for scorer, method_i in scorers:
            score_i, scores_i = scorer.compute_score(ref, hypo)
            
            score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
            method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        score_dict = dict(zip(method,  score))

        for method, score in score_dict.items():
            print("\n%s score is %.4f." % (method, score))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    main()