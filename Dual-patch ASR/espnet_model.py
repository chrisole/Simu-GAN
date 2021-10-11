from functools import reduce
from itertools import permutations
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import logging
import pickle
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types
from espnet2.asr.kl_loss import KLLoss
from espnet2.asr.kl_loss_enc import KLLoss_enc
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.enh.abs_enh import AbsEnhancement
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel




#import pysnooper
class ESPnetEnhASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize_clean_branch: Optional[AbsNormalize],
        normalize_enh_branch: Optional[AbsNormalize],
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        ctc: CTC,
        rnnt_decoder: None,
        ctc_weight: float = 0.3,
        clean_weight: float = 0.1,
        ignore_id: int = -1,
        lsm_weight: float = 0.1,
        enh_weight: float = 0.2,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        num_spk: int = 1,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight
        assert 0.0 <= enh_weight <= 1.0, enh_weight
        assert 0.0 <= clean_weight <= 1.0, clean_weight
        assert rnnt_decoder is None, "Not implemented"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.clean_weight = clean_weight
        self.token_list = token_list.copy()
        self.frontend = frontend
        self.specaug = specaug
        self.normalize_clean_branch = normalize_clean_branch
        self.normalize_enh_branch = normalize_enh_branch
        self.encoder = encoder
        self.decoder = decoder
        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc
        self.rnnt_decoder = rnnt_decoder
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.criterionConsistencyLoss = KLLoss(vocab_size, ignore_id, length_normalized_loss, )

        self.criterionConsistencyLoss_enc = KLLoss_enc(length_normalized_loss )

        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None

        self.idx_blank = -1

    def forward(
        self, 
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        speech_ref1: torch.Tensor,
        text_ref1: torch.Tensor,
        text_ref1_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        assert text_ref1_lengths.dim() == 1
        # Check that batch_size is unified
        assert (
            speech_mix.shape[0]
            == speech_mix_lengths.shape[0]
            == text_ref1.shape[0]
            == text_ref1_lengths.shape[0]
        ), (
            speech_mix.shape,
            speech_mix_lengths.shape,
            text_ref1.shape,
            text_ref1_lengths.shape,
        )
        batch_size = speech_mix.shape[0]

        # for data-parallel
        text_length_max = text_ref1_lengths.max()
        text_ref1 = torch.cat(
            [
                text_ref1,
                torch.ones(batch_size, text_length_max, dtype=text_ref1.dtype).to(
                    text_ref1.device
                )
                * self.idx_blank,
            ],
            dim=1,
        )

        noisy_encoder_out, noisy_encoder_out_lens = self.encode_noisy_branch(speech_mix, speech_mix_lengths)
        clean_encoder_out, clean_encoder_out_lens = self.encode_clean_branch(speech_ref1, speech_mix_lengths)
        text_ref_all = text_ref1
        text_ref_lengths = text_ref1_lengths
        # 2a. Attention-decoder branch
        if self.ctc_weight == 1.0:
            enh_loss_att, enh_acc_att, enh_cer_att, enh_wer_att = None, None, None, None
        else:
            enh_loss_att, enh_acc_att, enh_cer_att, enh_wer_att, enh_decoder_out = self._calc_att_loss(
                noisy_encoder_out, noisy_encoder_out_lens, text_ref_all, text_ref_lengths
            )

        # 2b. CTC branch
        if self.ctc_weight == 0.0:
            enh_loss_ctc, enh_cer_ctc = None, None
        else:
            enh_loss_ctc, enh_cer_ctc = self._calc_ctc_loss(
                noisy_encoder_out, noisy_encoder_out_lens, text_ref_all, text_ref_lengths
            )

        # 2c. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(
                encoder_out, encoder_out_lens, text_ref_all, text_ref_lengths
            )

        # 2d loss branch
        if self.ctc_weight == 0.0:
            enh_loss_asr = enh_loss_att
        elif self.ctc_weight == 1.0:
            enh_loss_asr = enh_loss_ctc
        else:
            enh_loss_asr = self.ctc_weight * enh_loss_ctc + (1 - self.ctc_weight) * enh_loss_att

  
        # 2a1. Attention-decoder branch
        if self.ctc_weight == 1.0:
            clean_loss_att, clean_acc_att, clean_cer_att, clean_wer_att = None, None, None, None
        else:
            clean_loss_att, clean_acc_att, clean_cer_att, clean_wer_att, clean_decoder_out = self._calc_att_loss(
                clean_encoder_out, clean_encoder_out_lens, text_ref_all, text_ref_lengths
            )

        # 2b1. CTC branch
        if self.ctc_weight == 0.0:
            clean_loss_ctc, clean_cer_ctc = None, None
        else:
            clean_loss_ctc, clean_cer_ctc = self._calc_ctc_loss(
                clean_encoder_out, clean_encoder_out_lens, text_ref_all, text_ref_lengths
            )

        # 2c1. RNN-T branch
        if self.rnnt_decoder is not None:
            _ = self._calc_rnnt_loss(
                encoder_out, encoder_out_lens, text_ref_all, text_ref_lengths
            )

        # 2d1 loss branch
        if self.ctc_weight == 0.0:
            clean_loss_asr = clean_loss_att
        elif self.ctc_weight == 1.0:
            clean_loss_asr = clean_loss_ctc
        else:
            clean_loss_asr = self.ctc_weight * clean_loss_ctc + (1 - self.ctc_weight) * clean_loss_att

        # 2e
        import torch.nn.functional as F
        B, U, V = enh_decoder_out.shape
        assert clean_decoder_out.shape == (B, U, V)
        assert self.vocab_size == V
        ys_in_pad, ys_out_pad = add_sos_eos(text_ref_all, self.sos, self.eos, self.ignore_id)
        con_loss1 = self.criterionConsistencyLoss(enh_decoder_out, clean_decoder_out, ys_out_pad)
        con_loss2 = self.criterionConsistencyLoss(clean_decoder_out, enh_decoder_out, ys_out_pad)
        # kl loss of encoder output noisy_encoder_out

        con_loss = con_loss1 + con_loss2

        # 3 total loss
        if self.clean_weight == 0.0:
            loss_joint = enh_loss
        elif self.clean_weight == 1.0:
            loss_joint = clean_loss_asr
        else:
            loss_joint = self.clean_weight * clean_loss_asr + (1 - self.clean_weight) * enh_loss_asr


        # 4 final loss

        loss = loss_joint + 0.4 * con_loss



        stats = dict(
            loss=loss.detach(),
            con_loss=con_loss.detach() if con_loss is not None else None,
            loss_joint=loss_joint.detach() if loss_joint is not None else None,
            enh_loss_att=enh_loss_att.detach() if enh_loss_att is not None else None,
            enh_loss_ctc=enh_loss_ctc.detach() if enh_loss_ctc is not None else None,
            enh_loss_asr=enh_loss_asr.detach() if enh_loss_asr is not None else None,
            clean_loss_att=clean_loss_att.detach() if clean_loss_att is not None else None,
            clean_loss_ctc=clean_loss_ctc.detach() if clean_loss_ctc is not None else None,
            clean_loss_asr=clean_loss_asr.detach() if clean_loss_asr is not None else None,
            acc=enh_acc_att,
            cer=enh_cer_att,
            wer=enh_wer_att,
            cer_ctc=enh_cer_ctc,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        #logging.info(f"in the espnet_joint_model3.py, total loss is {loss}, loss_att is {loss_att} loss_ctc is {loss_ctc} and  enh_loss is {loss_enh}")
        #logging.info(f"in the espnet_joint_model3.py, total loss is {loss}, loss_att is {loss_att} loss_ctc is {loss_ctc} and no enh_loss ")
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight
    def inference(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhancement + Frontend + Encoder 

        Args:
            speech_mix: (Batch, Length, ...)
            speech_mix_lengths: (Batch, )
        """
        # Check that batch_size is unified
        assert (
            speech_mix.shape[0]
            == speech_mix_lengths.shape[0]
           
        ), (
            speech_mix.shape,
            speech_mix_lengths.shape,
        )
        batch_size = speech_mix.shape[0]

        encoder_out, encoder_out_lens = self.encode_noisy_branch(speech_mix, speech_mix_lengths)
       
        return encoder_out, encoder_out_lens


    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    )-> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]
        logging.info(f"in the collect_feats function, **kwargs is {kwargs}")
        speech_clean = kwargs["speech_ref1"]

        enh_branch_feats, enh_branch_feats_lengths = self._extract_feats(speech_mix, speech_mix_lengths)
        #enh_feats, enh_feats_lengths = enh_feats, enh_feats_lengths 
        enh_branch_feats, enh_branch_feats_lengths = enh_branch_feats, enh_branch_feats_lengths
        
        # for mix speech branch
        clean_branch_feats, clean_branch_feats_lengths = self._extract_feats(speech_clean, speech_mix_lengths)
        clean_branch_feats, clean_branch_feats_lengths = clean_branch_feats, clean_branch_feats_lengths 
        
        return {"enh_branch_feats": enh_branch_feats,
                "enh_branch_feats_lengths":  enh_branch_feats_lengths,
                "clean_branch_feats": clean_branch_feats,
                "clean_branch_feats_lengths": clean_branch_feats_lengths,
        }

    def encode_noisy_branch(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, freq)
            speech_lengths: (Batch, )
        """
        # 1. Extract feats
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        #logging.info(f"in the encode function, input of log-mel function, is {speech} and its shape is {speech.shape}")
        #logging.info(f"in the encode function, output of log-mel function, is {feats} and its shape is {feats.shape}")
        # 2. Data augmentation for spectrogram
        if self.specaug is not None and self.training:
            feats, feats_lengths = self.specaug(feats, feats_lengths)

        # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        #logging.info(f"in the encode function, input of self.normalize is {feats} and its shape is {feats.shape}")
        if self.normalize_enh_branch is not None:
            feats, feats_lengths = self.normalize_enh_branch(feats, feats_lengths)
        #logging.info(f"in the encode function, output of self.normalize is {feats} and its shape is {feats.shape}")
        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        #logging.info(f"in the espnet_joint_model3.py, ASR module encoder input is feats is {feats}")
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def encode_clean_branch(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
 
        Args:
            speech: (Batch, Length, freq)
            speech_lengths: (Batch, )
        """
        # 1. Extract feats
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        #logging.info(f"in the encode function, input of log-mel function, is {speech} and its shape is {speech.shape}")
        #logging.info(f"in the encode function, output of log-mel function, is {feats} and its shape is {feats.shape}")
        # 2. Data augmentation for spectrogram
        if self.specaug is not None and self.training:
            feats, feats_lengths = self.specaug(feats, feats_lengths)
 
        # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
        #logging.info(f"in the encode function, input of self.normalize is {feats} and its shape is {feats.shape}")
        if self.normalize_clean_branch is not None:
            feats, feats_lengths = self.normalize_clean_branch(feats, feats_lengths)
        #logging.info(f"in the encode function, output of self.normalize is {feats} and its shape is {feats.shape}")
        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        #logging.info(f"in the espnet_joint_model3.py, ASR module encoder input is feats is {feats}")
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
 
        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )
 
        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att, decoder_out

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        raise NotImplementedError

    # Enhancement related, basicly from the espnet2/enh/espnet_model1.py
    # def forward_enh(
    #     self,
    #     speech_mix: torch.Tensor,
    #     speech_mix_lengths: torch.Tensor = None,
    #     **kwargs,
    # ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    #     """Frontend + Encoder + Decoder + Calc loss
    #
    #     Args:
    #         speech_mix: (Batch, samples) or (Batch, samples, channels)
    #         speech_ref: (Batch, num_speaker, samples)
    #                     or (Batch, num_speaker, samples, channels)
    #         speech_mix_lengths: (Batch,), default None for chunk interator,
    #                         because the chunk-iterator does not have the
    #                         speech_lengths returned. see in
    #                         espnet2/iterators/chunk_iter_factory.py
    #     """
    #     # clean speech signal of each speaker
    #     speech_ref = [
    #         kwargs["speech_ref{}".format(spk + 1)] for spk in range(self.num_spk)
    #     ]
    #     #uttid = kwargs['uttid']
    #     # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
    #     speech_ref = torch.stack(speech_ref, dim=1)
    #
    #
    #
    #     batch_size = speech_mix.shape[0]
    #     speech_lengths = (
    #         speech_mix_lengths
    #         if speech_mix_lengths is not None
    #         else torch.ones(batch_size).int() * speech_mix.shape[1]
    #     )
    #     assert speech_lengths.dim() == 1, speech_lengths.shape
    #     # Check that batch_size is unified
    #     assert speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0], (
    #         speech_mix.shape,
    #         speech_ref.shape,
    #         speech_lengths.shape,
    #     )
    #     batch_size = speech_mix.shape[0]
    #
    #     # for data-parallel
    #     speech_ref = speech_ref[:, :, : speech_lengths.max()]
    #     speech_mix = speech_mix[:, : speech_lengths.max()]
    #     # predict magnidude (it is list) and masks
    #     predicted_magnitude, tf_length, mask_pre = self.enh_model(
    #       # uttid, speech_mix, speech_lengths
    #       speech_mix, speech_lengths,
    #     )
    #     #logging.info(f"in the forward_enh() function ,predicted_magnitude is {predicted_magnitude} its shape is {predicted_magnitude[0].shape}")
    #     # prepared ref magnitude, wave -> stft -> abs -> magnitude
    #     speech_ref = speech_ref.squeeze(1) # (B,1,samples) -> (B, samples)
    #     #logging.info(f"in the espnet_model1, speech_ref is {speech_ref} its shape is {speech_ref.shape}")
    #     input_spectrum, flens = self.enh_model.stft(speech_ref, speech_lengths) # it need to check speech_lengths
    #     input_spectrum = ComplexTensor(input_spectrum[..., 0], input_spectrum[..., 1])
    #     magnitude_ref = abs(input_spectrum)
    #     #logging.info(f"in the forward_enh() function, magnitude_ref is {magnitude_ref} its shape is{magnitude_ref.shape}")
    #     if self.loss_type == "magnitude3":
    #         # compute loss on magnitude spectrum
    #         # magnitude_ref  is B x T x F
    #         # magnitude_pre[0] is B x T x F
    #
    #         #logging.info(f"in the forward_enh() function ,using self.loss_type  is {self.loss_type }, magnitude_ref  shape is {magnitude_ref[0].shape}")
    #         #logging.info(f"in the forward_enh() function , predicted_magnitude[0]  shape is {predicted_magnitude[0].shape}")
    #         tf_loss, perm = self._permutation_loss3(
    #             magnitude_ref, predicted_magnitude[0], tf_length,
    #         )
    #
    #     loss = tf_loss
    #     # force_gatherable: to-device and to-tensor if scalar for DataParallel
    #     #loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
    #     #logging.info(f"final loss is {loss}, stats is {stats}, weight is {weight}")
    #
    #     return loss, perm, predicted_magnitude[0], tf_length, mask_pre


    @staticmethod
    def tf_mse_loss(ref, inf):
        """time-frequency MSE loss.

        :param ref: (Batch, T, F)
        :param inf: (Batch, T, F)
        :return: (Batch)
        """
        assert ref.dim() == inf.dim(), (ref.shape, inf.shape)
        if ref.dim() == 3:
            mseloss = (abs(ref - inf) ** 2).mean(dim=[1, 2])
        elif ref.dim() == 4:
            mseloss = (abs(ref - inf) ** 2).mean(dim=[1, 2, 3])
        else:
            raise ValueError("Invalid input shape: ref={}, inf={}".format(ref, inf))

        return mseloss

    @staticmethod
    def tf_l1_loss(ref, inf):
        """time-frequency L1 loss.

        :param ref: (Batch, T, F) or (Batch, T, C, F)
        :param inf: (Batch, T, F) or (Batch, T, C, F)
        :return: (Batch)
        """
        assert ref.dim() == inf.dim(), (ref.shape, inf.shape)
        if ref.dim() == 3:
            l1loss = abs(ref - inf).mean(dim=[1, 2])
        elif ref.dim() == 4:
            l1loss = abs(ref - inf).mean(dim=[1, 2, 3])
        else:
            raise ValueError("Invalid input shape: ref={}, inf={}".format(ref, inf))
        return l1loss

    @staticmethod
    def si_snr_loss(ref, inf):
        """si-snr loss

        :param ref: (Batch, samples)
        :param inf: (Batch, samples)
        :return: (Batch)
        """
        ref = ref / torch.norm(ref, p=2, dim=1, keepdim=True)
        inf = inf / torch.norm(inf, p=2, dim=1, keepdim=True)

        s_target = (ref * inf).sum(dim=1, keepdims=True) * ref
        e_noise = inf - s_target

        si_snr = 20 * torch.log10(
            torch.norm(s_target, p=2, dim=1) / torch.norm(e_noise, p=2, dim=1)
        )
        return -si_snr

    @staticmethod
    def si_snr_loss_zeromean(ref, inf):
        """si_snr loss with zero-mean in pre-processing.

        :param ref: (Batch, samples)
        :param inf: (Batch, samples)
        :return: (Batch)
        """
        eps = 1e-8

        assert ref.size() == inf.size()
        B, T = ref.size()
        # mask padding position along T

        # Step 1. Zero-mean norm
        mean_target = torch.sum(ref, dim=1, keepdim=True) / T
        mean_estimate = torch.sum(inf, dim=1, keepdim=True) / T
        zero_mean_target = ref - mean_target
        zero_mean_estimate = inf - mean_estimate

        # Step 2. SI-SNR with order
        # reshape to use broadcast
        s_target = zero_mean_target  # [B, T]
        s_estimate = zero_mean_estimate  # [B, T]
        # s_target = <s', s>s / ||s||^2
        pair_wise_dot = torch.sum(s_estimate * s_target, dim=1, keepdim=True)  # [B, 1]
        s_target_energy = torch.sum(s_target ** 2, dim=1, keepdim=True) + eps  # [B, 1]
        pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, T]
        # e_noise = s' - s_target
        e_noise = s_estimate - pair_wise_proj  # [B, T]

        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=1) / (
            torch.sum(e_noise ** 2, dim=1) + eps
        )
        # print('pair_si_snr',pair_wise_si_snr[0,:])
        pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + eps)  # [B]
        # print(pair_wise_si_snr)

        return -1 * pair_wise_si_snr

    @staticmethod
    def _permutation_loss(ref, inf, criterion, perm=None):
        """The basic permutation loss function.

        Args:
            ref (List[torch.Tensor]): [(batch, ...), ...]
            inf (List[torch.Tensor]): [(batch, ...), ...]
            criterion (function): Loss function
            perm: (batch)
        Returns:
            loss: torch.Tensor: (batch)
            perm: list[(num_spk)]
        """
        num_spk = len(ref)

        def pair_loss(permutation):
            return sum(
                [criterion(ref[s], inf[t]) for s, t in enumerate(permutation)]
            ) / len(permutation)

        losses = torch.stack(
            [pair_loss(p) for p in permutations(range(num_spk))], dim=1
        )
        if perm is None:
            loss, perm = torch.min(losses, dim=1)
        else:
            loss = losses[torch.arange(losses.shape[0]), perm]

        perm_list = [p for p in permutations(range(num_spk))]
        perm_detail = []
        for p in perm:
            perm_detail.append(perm_list[p])  # egs: list([1,0]) or list([0,2,1])
        return loss.mean(), perm_detail
    @staticmethod
    def _permutation_loss3(ref, inf, magnitude_lengths, perm=None):
        #logging.info(f"in _permutation_loss3, ref shape {ref.shape} and inf shape is {inf.shape}")
        #logging.info(f"in _permutation_loss3, magnitude_lengths is {magnitude_lengths}")
        input_size = magnitude_lengths
        def loss():
            loss_for_permute = []
            #logging.info(f"masks_[0]  type is {type(masks_[0])}")
            #logging.info(f"ref[0] type is {type(ref[0])}")
            # N X T X F

            inf_magnitude = inf
            #logging.info(f"in _permutation_loss3,inf_magnitude shape is {inf_magnitude.shape}")
            #  N X T X F
            ref_magnitude = ref
            #logging.info(f"in _permutation_loss3,ref_magnitude shape is {ref_magnitude.shape}")
            # N X T X F
            mse = torch.pow(inf_magnitude - ref_magnitude, 2)
            # N X T X 1
            mse_sum1 = torch.sum(mse, -1)
            # N X 1 X1
            utt_loss = torch.sum(mse_sum1, -1)
            # utt_loss = torch.sum(torch.sum(torch.pow(masks_[int(0)]*inf - ref[int(0)], 2), -1), -1)
            loss_for_permute.append(utt_loss)
            #logging.info(f"input_size device is {input_size.device}")
            #logging.info(f"")
            input_size_ = torch.tensor(input_size, dtype=torch.float32, device=inf_magnitude.device)
            #logging.info(f"input_size device again is {input_size.device}")
            loss_perutt = sum(loss_for_permute) / input_size_
            return loss_perutt

        #logging.info(f"num_utts is {ref[0].shape[0]}")
        num_utts = ref.shape[0] # batch size
        #logging.info(f"in _permutation_loss3,num_utts is {num_utts}")
        # O(N!), could be optimized
        # 1 x N
        pscore = torch.stack([loss()], dim=0)
        # pscore = torch.stack([loss(p) for p in permutations(range(num_spk))], dim=1)
        #logging.info(f"pscore is {pscore}")
        # N
        num_spk=1
        min_perutt, _ = torch.min(pscore, dim=0)
        loss = torch.sum(min_perutt) / (num_spk * num_utts)
        """
        the loss sum freq and sum time ,then average on the time axis, then average on the number of utterances
        """
        #logging.info(f"loss is {loss}")
        return loss , perm
 
