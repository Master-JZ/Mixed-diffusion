import enum
import math
import numpy as np
import torch as th
import sys
sys.path.append('.')

import torch.nn.functional as F

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param predict_xstart: the model outputs to predict x_0, else to predict eps.
    :param learn_sigmas: the model outputs to predict sigma or not. Default: False
    :param rescale_learned_sigmas, sigma_small: details setting of learned sigmas
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        predict_xstart,
        rescale_learned_sigmas,
        learn_sigmas,
        sigma_small,
        use_kl,
        rescale_timesteps=False,
    ):  #构造函数的计算与噪声类型无关
        self.rescale_timesteps = rescale_timesteps
        self.predict_xstart = predict_xstart
        self.rescale_learned_sigmas = rescale_learned_sigmas
        self.learn_sigmas = learn_sigmas
        self.sigma_small = sigma_small
        self.use_kl = use_kl

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.mapping_func = None # implement in train main()
        self.add_mask_noise = False # TODO

    def training_losses(self, model, t, input_ids, input_ids_mask, attention_mask):
        self.model = model
        return self.training_losses_seq2seq(model, t, input_ids, input_ids_mask, attention_mask)
    
    def rec_predict(self, model, *args, **kwargs):
        self.model = model
        return self.rec_predict_performance(model, *args, **kwargs)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.计算均值方差
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.生成噪声的关键位置

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)#获得高斯噪声 实际上这里的噪声不会为none，所以可改可不改

        assert noise.shape == x_start.shape
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return th.where(mask==0, x_start, x_t)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 开始反向去噪相关
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, attention_mask, clip_denoised=True, denoised_fn=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """


        B, C = x.size(0), x.size(-1)
        assert t.shape == (B,)
        # print(x.shape)
        model_output = model.model.diffusion_reverse(x, self._scale_timesteps(t), attention_mask)   #消融context-aware
        # attn = th.ones(x.size(0), x.size(1), device=x.device, dtype=th.float)
        # model_output = model.model.diffusion_reverse(x, self._scale_timesteps(t), attention_mask=attn)

        # for fixedlarge, we set the initial (log-)variance like so
        # to get a better decoder log likelihood.
        model_variance = np.append(self.posterior_variance[1], self.betas[1:])
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                # print(denoised_fn)
                x = denoised_fn(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.predict_xstart:
            pred_xstart = process_xstart(model_output)
        else:
            ### model is used to predict eps
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
            top_p=None, mask=None, x_start=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.这里的noise定义也需要改ma

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if top_p is not None and top_p > 0:
            # print('top_p sampling')
            noise = th.randn_like(x)
            replace_mask = th.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (th.abs(noise) <= top_p).all()

        else:
            noise = th.randn_like(x)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        if mask == None:
            pass
        else:
            sample = th.where(mask==0, x_start, sample)

        return {
            "sample": sample, 
            "pred_xstart": out["pred_xstart"],
            "greedy_mean": out["mean"], 
            "out": out
        }

    
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        gap=1,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param clamp_step: in clamp_first mode, choose end clamp step, otherwise starting clamp step
        :param clamp_first: bool, clamp_first mode
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = []
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            top_p=top_p,
            clamp_step=clamp_step,
            clamp_first=clamp_first,
            mask=mask,
            x_start=x_start
        ):
            final.append(sample['sample'])
        return final

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None: # custom your the start point of x_0
            sample_x = noise
        else:
            sample_x = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices: # from T to 0
            t = th.tensor([i] * shape[0], device=device)
            if not clamp_first:
                if i > clamp_step:
                    denoised_fn_cur = None
                else:
                    denoised_fn_cur = denoised_fn
            else:
                if i >= clamp_step:
                    denoised_fn_cur = denoised_fn
                else:
                    denoised_fn_cur = None
            with th.no_grad():
                out = self.p_sample(
                    model,
                    sample_x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn_cur,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                    mask=mask,
                    x_start=x_start
                )
                yield out
                sample_x = out["sample"]


    # def _get_x_start(self, x_start_mean, std):
    def _get_x_start(self, x_start_mean, std, input_ids_x=None, input_ids_mask=None):#新增后两个参数，用于传入给混合噪声生成
        '''
        Word embedding projection from {Emb(w)} to {x_0}这里也改，从embedding 到x0之前加一点点噪声
        :param x_start_mean: word embedding
        :return: x_0
        '''
        # noise = th.randn_like(x_start_mean)
        if (input_ids_x is not None) and (input_ids_mask is not None):    
            noise = self.mixed_noise(x_start_mean, input_ids_x, input_ids_mask)
            assert noise.shape == x_start_mean.shape
        else:#没有东西传入则退化成高斯噪声
            noise = th.randn_like(x_start_mean)
        # print(x_start_mean.device, noise.device)
        return (
             x_start_mean + std * noise
        )

    def _token_discrete_loss(self, x_t, get_logits, input_ids, mask=None, truncate=False, t=None):
        '''
        the loss of -log p(w|z_0)
        :param x_start_mean: word embedding
        :return: x_0
        '''
        reshaped_x_t = x_t
        logits = get_logits(reshaped_x_t)  # bsz, seqlen, vocab
        # print(logits.shape)
        loss_fct = th.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
        if mask != None:
            decoder_nll *= mask
        # print(decoder_nll.shape)
        if mask != None:
            # print("**************debug mask.sum(dim=-1)", mask.sum(dim=-1))
            decoder_nll = decoder_nll.sum(dim=-1)/(mask.sum(dim=-1)+0.00001)
        else:
            # print("========================debug input_ids", input_ids.shape)
            # print("========================debug decoder_nll", decoder_nll.shape)
            decoder_nll = decoder_nll.mean(dim=-1)

        return decoder_nll

    def _x0_helper(self, model_output, x, t):

        if self.predict_xstart:
            pred_xstart = model_output
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else: # predict eps
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        return {'pred_xprev':pred_prev, 'pred_xstart':pred_xstart}


    def get_extended_attention_mask(self, input_ids):  # future masking
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = th.triu(th.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        subsequent_mask = subsequent_mask.cuda().to(input_ids.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def training_losses_seq2seq(self, model, t, input_ids_x, input_ids_mask, attention_mask, noise=None):
        """
        Compute training losses for a single timestep.这里需要改

        :param model: the model to evaluate loss on.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        # extended_attention_mask = self.get_extended_attention_mask(input_ids_x)
        x_start_mean = model.model.get_embeds(input_ids_x)  # original seq emb  w(y)
        # #获取全部item的embedding矩阵
        # item_emb = model.model.get_all_embs()
        
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   th.tensor([0]).to(x_start_mean.device),
                                   x_start_mean.shape)
      
        # x_start = self._get_x_start(x_start_mean, std) 
        x_start = self._get_x_start(x_start_mean, std, input_ids_x, input_ids_mask)
       
        if noise is None:
            # noise = th.randn_like(x_start)
            # noise = self.sequence_guided_mixed_noise(x_start_mean, input_ids_x, input_ids_mask, item_emb)
            noise = self.mixed_noise(x_start_mean, input_ids_x, input_ids_mask)
        ################################################################################################################noise should change这里改完先保留一版
    
        x_t = self.q_sample(x_start, t, noise=noise, mask=input_ids_mask) # reparametrization trick. 
        get_logits = model.model.get_logits  

        terms = {}
        # **************mse loss*******************
        target = x_start
        model_output = model.model.diffusion_reverse(x_t, self._scale_timesteps(t), attention_mask)  #消融context-aware
        # attn = th.ones_like(input_ids_mask, dtype=th.float, device=x_t.device)
        # model_output = model.model.diffusion_reverse(x_t, self._scale_timesteps(t), attention_mask=attn)
        assert model_output.shape == target.shape == x_start.shape
        
        # terms["mse"] = mean_flat((target - model_output) ** 2)
        terms["mse"] = mean_flat((target[input_ids_mask.bool()] - model_output[input_ids_mask.bool()]) ** 2)
    
        model_out_x_start = self._x0_helper(model_output, x_t, t)['pred_xstart'] # pred_xstart = model_output 这个model_out_x_start就是前边的model_output
        
        
        
        # decoder_nll = self._token_discrete_loss(x_start, get_logits, input_ids_x) # embedding regularization
        # decoder_nll = self._token_discrete_loss(x_start[attention_mask.bool()], get_logits, input_ids_x[attention_mask.bool()]) # embedding regularization
        
        terms["nll"] = self._token_discrete_loss(model_out_x_start, get_logits, input_ids_x, mask=input_ids_mask, truncate=True, t=t) # x_0->model_out_x_start
        
        # ========================generate samples================================
        logits = self.model.model.get_logits(model_out_x_start)
        k_value = 10
        cands = th.topk(logits, k=k_value, dim=-1).indices # B,L,k
        random_element = th.randint(k_value, size=(cands.shape[0], cands.shape[1], 2)).to(input_ids_x.device)
        cands = cands.gather(dim=2, index=random_element) # B,L,2 
        original_id_expand = input_ids_x.unsqueeze(-1).repeat(1,1,2) # B,L,2 
        mask_expand = input_ids_mask.unsqueeze(-1).repeat(1,1,2) # B,L,2 
        aug_seqs = original_id_expand * (1-mask_expand) + cands * mask_expand.long() # B,L,2 
        
        
        return terms["mse"].mean(), terms["nll"].mean(), aug_seqs[:,:,0], aug_seqs[:,:,1]  #只返回其中一条增强序列用来对比学习(注释掉：：1的那个) 12.19
    
    
    def rec_predict_performance(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs. # not used unless fixing the input embeddings
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        x_start_fix = x_start # save the orignal x_0
        assert 'input_ids' in model_kwargs
        input_ids_x = model_kwargs.pop('input_ids').to(t.device)
        input_ids_mask = model_kwargs.pop('input_mask').to(t.device) 
        user_id = model_kwargs.pop('user_id').to(t.device)
        target_pos = model_kwargs.pop('target_pos').to(t.device)
        target_neg = model_kwargs.pop('target_neg').to(t.device)
        answer = model_kwargs.pop('answer').to(t.device)
        masked_indices1 = model_kwargs.pop('masked_indices1').to(t.device)
        masked_indices2 = model_kwargs.pop('masked_indices2').to(t.device)
      
        extended_attention_mask = self.get_extended_attention_mask(input_ids_x)
        input_seq_emb = model.model.module.get_embeds(input_ids_x)  # 原始seq emb
        predict_scores = model.model.module.full_sort_predict(input_seq_emb, extended_attention_mask)      

        return predict_scores

    def ddim_sample(
        self,
        model,
        x,
        t,
        attention_mask,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        langevin_fn=None,
        mask=None,
        x_start=None
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
    
        out = self.p_mean_variance(
            model,
            x,
            t,
            attention_mask,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x) ##################################################################noise enter
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        # print(sigma.mean())
        sample = mean_pred + nonzero_mask * sigma * noise
        if langevin_fn:
            print(t.shape)
            sample=langevin_fn(sample, mean_pred, sigma, self.alphas_cumprod_prev[t[0]], t, x)
        
        if mask == None:
            pass
        else:
            sample = th.where(mask==0, x_start, sample)
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        attention_mask=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        gap=1,
    ):
        """
        Generate samples from the model using DDIM.
        :param gap: compute ddim sampling for each {gap} step

        Same usage as p_sample_loop().
        """
        final = []
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            attention_mask=attention_mask,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            mask=mask,
            x_start=x_start,
            gap = gap
        ):
            final.append(sample['sample'])
        return final

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        attention_mask=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        langevin_fn=None,
        mask=None,
        x_start=None,
        gap=1
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            sample_x = noise
        else:
            sample_x = th.randn(*shape, device=device)      #############################noise
        indices = list(range(self.num_timesteps))[::-1][::gap]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    sample_x,
                    t,
                    attention_mask,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    mask=mask,
                    x_start=x_start
                )
                yield out
                sample_x = out["sample"]

    #调用混合噪声
    def mixed_noise(self, x_base, input_ids, input_mask):
        item_emb = self.model.model.get_all_embs()  #这里统一调用获取全部item的embedding

        noise = self.sequence_guided_mixed_noise(
            x_base = x_base,
            input_ids = input_ids,
            input_mask = input_mask,
            item_emb = item_emb,
        )

        # noise = th.randn_like(x_base) #注释掉混合噪声，改回高斯噪声

        return noise

    #混合噪声
    def sequence_guided_mixed_noise(
    self,
    x_base: th.Tensor,      # [B, L, D] 一个batch的数据
    input_ids: th.Tensor,   # [B, L] L是长度
    input_mask: th.Tensor,  # [B, L]
    item_emb: th.Tensor,    # [V, D] 全部物品的embedding矩阵
    lam: float = 0.1,   #0.1
    topk: int = 16,    #test 32/16
    tau: float = 0.1,
    apply_mask: bool = True,
    ) -> th.Tensor:
        """
        对一个 batch 的序列生成混合噪声:
        - 每条序列 b 先算一个 sequence-level 噪声向量 eps_seq[b] ∈ R^D
        - 再广播到该序列的各个位置，组成 noise[b, :, :]
        """
        B, L, D = x_base.shape
        V, D_emb = item_emb.shape
        assert D == D_emb, "embedding dim mismatch between x_base and item_emb"

        device = x_base.device
        noise = th.zeros_like(x_base)
        eps_den = 1e-6      #特别小的一个值，防止除以0

        for b in range(B):
            # 这一条序列的 embedding / mask， 取出其中一条序列
            seq_emb  = x_base[b]              # [L, D]
            seq_mask = input_mask[b].bool()   # [L]

            # 1. 选出有效位置的 embedding，用来构造序列级表示
            valid_emb = seq_emb[seq_mask]     # [L_valid, D]
            if valid_emb.numel() == 0:
                # 全是 padding 的极端情况：退回纯高斯
                eps_seq = th.randn(D, device=device)
            else:
                # 2. 序列级 query 向量 q_u（简单平均，可换成 last hidden 等）
                q_u = valid_emb.mean(dim=0)   # [D]

                # 3. q_u vs 全库 item 的相似度（点积）
                # sim[j] = <q_u, item_emb[j]>
                sim = th.matmul(item_emb, q_u)    # [V]

                # 4. 取 Top-K 最近邻 item，并基于相似度做 softmax 权重
                k = min(topk, sim.size(0))
                topv, topi = th.topk(sim, k=k)           # [k]
                weights = th.softmax(topv / tau, dim=0)  # [k]

                # 5. 加权得到序列级语义中心 m_u
                #    m_u = Σ w_i * W_{j_i}
                m_u = (weights.unsqueeze(1) * item_emb[topi]).sum(dim=0)  # [D]

                # 6. 方向 d_u = m_u - q_u，并做一次标准化（得到结构化方向 d_hat）
                d_u = m_u - q_u                         # [D]
                mu_d = d_u.mean()
                std_d = d_u.std(unbiased=False)
                d_hat = (d_u - mu_d) / (std_d + eps_den)

                # 7. 采一份标准高斯 g，并与 d_hat 按 λ 混合
                g = th.randn_like(d_hat)                # [D]
                u = lam * d_hat + (1.0 - lam) * g       # [D]

                # 8. 再标准化一次 u，得到最终这个序列的噪声向量 eps_seq（0 均值 / 1 方差） 1210晚注释掉，去掉最后一步标准化，使变成高斯+一个常数、且范围在-1~1
                # mu_u = u.mean()
                # std_u = u.std(unbiased=False)
                # eps_seq = (u - mu_u) / (std_u + eps_den)  # [D]
                eps_seq = u

            # 9. 把 eps_seq 广播回这一条序列的每个位置
            if apply_mask:
                # 只对 mask==1 的位置加同一份噪声，其余位置保持 0
                seq_noise = th.zeros_like(seq_emb)   # [L, D]
                num_valid = seq_mask.sum()
                if num_valid > 0:
                    seq_noise[seq_mask] = eps_seq.unsqueeze(0).expand(num_valid, D)
            else:
                # 整条序列所有位置都使用 eps_seq
                seq_noise = eps_seq.unsqueeze(0).expand(L, D)

            noise[b] = seq_noise

        return noise  # [B, L, D]


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        # print(kwargs.keys())
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, x, t, attention_mask, clip_denoised,denoised_fn
    ):  # pylint: disable=signature-differs
        # print('called p_mean_var')
        return super().p_mean_variance(self._wrap_model(model), x, t, attention_mask,clip_denoised,denoised_fn)

            # compute_losses = functools.partial(
            #     self.diffusion.training_losses,
            #     self.model,
            #     t,
            #     input_ids,
            #     input_ids_mask
            # )
    def training_losses(
        self, model,  t,
                 input_ids,
                 input_ids_mask,
                 attention_mask
):  # pylint: disable=signature-differs
        # print('called training_losses')
        return super().training_losses(self._wrap_model(model), t,
                 input_ids,
                 input_ids_mask,
                 attention_mask)
    
    def rec_predict(
        self, model, *args, **kwargs
    ): 
        return super().rec_predict(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, att_mask, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, att_mask, **kwargs)
