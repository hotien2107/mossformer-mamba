import torch
import sys
import os
import argparse
sys.path.append(os.path.dirname(sys.path[0]))

from utils.misc import reload_for_eval
from utils.decode import decode_one_audio
from dataloader.dataloader import DataReader
import yamlargparse
import soundfile as sf
import warnings
from networks import network_wrapper

warnings.filterwarnings("ignore")


def _preflight_mamba2(args):
    if args.recurrent_type != 'mamba2':
        return

    if not args.use_cuda:
        raise RuntimeError("Mamba2 inference requires CUDA in this integration. Set --use-cuda 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("Mamba2 inference requested CUDA, but torch.cuda.is_available() is False.")

    expanded_width = args.recurrent_inner_channels * args.mamba_expand
    if expanded_width % args.mamba_headdim != 0:
        raise ValueError(
            "Invalid Mamba2 config: recurrent_inner_channels * mamba_expand must be divisible by "
            f"mamba_headdim, got {args.recurrent_inner_channels} * {args.mamba_expand} and "
            f"mamba_headdim={args.mamba_headdim}."
        )

    conv_channels = expanded_width + 2 * args.mamba_d_state
    if conv_channels % 8 != 0:
        raise ValueError(
            "Invalid Mamba2 config: expanded recurrent width + 2 * mamba_d_state must be a multiple "
            f"of 8 for the fused conv path, got {expanded_width} + 2 * {args.mamba_d_state} = {conv_channels}."
        )

    try:
        from mamba_ssm import Mamba2  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Failed Mamba2 preflight. Install pip packages `mamba-ssm`, `causal-conv1d`, "
            "`ninja`, and `packaging` before inference with recurrent_type='mamba2'."
        ) from exc

def inference(args):
    device = torch.device('cuda') if args.use_cuda==1 else torch.device('cpu')
    _preflight_mamba2(args)
    print(device)
    print('creating model...')
    model = network_wrapper(args).ss_network
    model.to(device)

    print('loading model ...')
    reload_for_eval(model, args.checkpoint_dir, args.use_cuda)
    model.eval()
    with torch.no_grad():

        data_reader = DataReader(args)
        output_wave_dir = args.output_dir
        if not os.path.isdir(output_wave_dir):
            os.makedirs(output_wave_dir)
        num_samples = len(data_reader)
        print('Decoding...')
        for idx in range(num_samples):
            input_audio, wav_id, input_len = data_reader[idx]
            print(f'audio: {wav_id}')
            output_audios = decode_one_audio(model, device, input_audio, args) 
            for spk in range(args.num_spks):
                output_audio = output_audios[spk][:input_len]
                sf.write(os.path.join(output_wave_dir, wav_id.replace('.wav', '_s'+str(spk+1)+'.wav')), output_audio, args.sampling_rate)
    print('Done!')
if __name__ == "__main__":
    # parser = argparse.ArgumentParser('PyTorch Version ENhancement')
    parser = yamlargparse.ArgumentParser("Settings")
    parser.add_argument('--config', help='config file path', action=yamlargparse.ActionConfigFile)

    parser.add_argument('--mode', type=str, default='inference', help='run train or inference')        
    parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', type=str, default='checkpoints/MossFormer2_SS_16K', help='the checkpoint dir')
    parser.add_argument('--input-path', dest='input_path', type=str, help='input dir or scp file for saving noisy audio')
    parser.add_argument('--output-dir', dest='output_dir', type=str, help='output dir for saving processed audio')
    parser.add_argument('--use-cuda', dest='use_cuda', default=1, type=int, help='use cuda')
    parser.add_argument('--num-gpu', dest='num_gpu', type=int, default=1, help='the num gpus to use')
    parser.add_argument('--network', type=str, help='select speech enhancement models: MossFormer2_SS_16K, MossFormer2_SS_8K')
    parser.add_argument('--sampling-rate', dest='sampling_rate', type=int, default=16000)

    #MossFormer2 model parameters
    parser.add_argument('--num-spks', dest='num_spks', type=int, default=2)
    parser.add_argument('--one-time-decode-length', dest='one_time_decode_length', type=int, default=60,
                        help='the max length (second) for one-time pass decoding')
    parser.add_argument('--decode-window', dest='decode_window', type=int, default=1,
                        help='segmental decoding window length (second)')
    parser.add_argument('--encoder_kernel-size', dest='encoder_kernel_size', type=int, default=16,
                        help='the Conv1D kernel size of encoder ')
    parser.add_argument('--encoder-embedding-dim', dest='encoder_embedding_dim', type=int, default=512,
                        help='the encoder output embedding size')
    parser.add_argument('--mossformer-squence-dim', dest='mossformer_sequence_dim', type=int, default=512,
                        help='the feature dimension used in MossFormer block')
    parser.add_argument('--num-mossformer_layer', dest='num_mossformer_layer', type=int, default='24',
                        help='the number of mosssformer layers used for sequence processing')
    parser.add_argument('--recurrent-type', dest='recurrent_type', type=str, default='fsmn',
                        help='recurrent block type: fsmn or mamba2')
    parser.add_argument('--recurrent-inner-channels', dest='recurrent_inner_channels', type=int, default=256,
                        help='bottleneck channels used by the recurrent branch')
    parser.add_argument('--mamba-d-state', dest='mamba_d_state', type=int, default=64,
                        help='Mamba2 state size when recurrent_type=mamba2')
    parser.add_argument('--mamba-d-conv', dest='mamba_d_conv', type=int, default=4,
                        help='Mamba2 local convolution width when recurrent_type=mamba2')
    parser.add_argument('--mamba-expand', dest='mamba_expand', type=int, default=2,
                        help='Mamba2 expansion factor when recurrent_type=mamba2')
    parser.add_argument('--mamba-headdim', dest='mamba_headdim', type=int, default=64,
                        help='Mamba2 head dimension; should divide the expanded recurrent width')

    args = parser.parse_args()
    print(args)

    inference(args)
